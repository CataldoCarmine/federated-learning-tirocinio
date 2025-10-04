import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import sys
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score, 
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score
)
warnings.filterwarnings('ignore')

# CONFIGURAZIONE SEMI PER RIPRODUCIBILITÀ
RANDOM_SEED = 42

# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==============
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = True       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = False                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False  # PCA

if ENABLE_PCA:
    ENABLE_IMPUTATION = True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 74  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 42  # Seme specifico per PCA

# Quando PCA disabilitata, disabilita rimozione feature quasi-costanti per compatibilità dei modelli
if ENABLE_PCA == False:
    ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False
    PCA_COMPONENTS = None

# ============== CONFIGURAZIONE RANDOM FOREST GLOBALE ==============
# Parametri per la selezione e aggregazione degli alberi (dal paper)
TREE_SELECTION_METHOD = 'weighted_accuracy'  # 'accuracy' o 'weighted_accuracy'
TREE_AGGREGATION_STRATEGY = 'per_forest'     # 'per_forest' o 'global'
MAX_TREES_GLOBAL = 100                       # Numero massimo alberi nel modello globale
ENSEMBLE_METHOD = 'weighted_voting'          # 'simple_voting' o 'weighted_voting'
NUM_ROUNDS = 50                              # Numero di round di addestramento federato

# Variabili globali per tracking metriche
all_federated_metrics = []  # Lista di dict, uno per round
last_confusion_matrix = None

# ============== FUNZIONI DI RIPRODUCIBILITÀ ==============
def set_reproducibility_seeds():
    """
    Imposta tutti i semi per garantire riproducibilità.
    Da chiamare all'inizio di ogni funzione critica.
    """
    # Seed per NumPy
    np.random.seed(RANDOM_SEED)
    
    # Seed per Python random (usato da scikit-learn)
    import random
    random.seed(RANDOM_SEED)
    
    # Configurazioni per determinismo
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# ============== FUNZIONI DI PREPROCESSING ==============
def fit_clip_outliers_iqr(X, k=5.0):
    """
    Calcola i limiti inferiori e superiori per ogni feature
    usando la regola dei quantili (IQR) sul dataset fornito.
    """
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def transform_clip_outliers_iqr(X, lower, upper):
    """
    Applica il clipping ai dati X usando i limiti forniti.
    """
    return np.clip(X, lower, upper)

def clip_outliers_iqr(X, k=5.0):
    """
    Calcola i limiti e applica il clipping in un'unica funzione.
    """
    lower, upper = fit_clip_outliers_iqr(X, k=k)
    return transform_clip_outliers_iqr(X, lower, upper)

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
    """
    Rimuove le feature che sono costanti almeno al 99.9%.
    """
    keep_mask = []
    n = X.shape[0]
    
    for col in range(X.shape[1]):
        col_data = X[:, col]
        
        # Conta la moda (valore più frequente)
        vals, counts = np.unique(col_data, return_counts=True)
        max_count = np.max(counts)
        ratio = max_count / n
        var = np.nanvar(col_data)
        
        # Tiene solo se NON è costante al 99.9% e varianza > threshold_var
        keep = not (ratio >= threshold_ratio or var < threshold_var)
        keep_mask.append(keep)
    
    keep_mask = np.array(keep_mask)
    return X[:, keep_mask], keep_mask

def clean_data_for_pca(X):
    """
    Pulizia robusta dei dati per prevenire problemi numerici in PCA.
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X_preprocessed):
    """
    Applica PCA con numero FISSO di componenti (identico ai client).
    """
    print(f"[Server] === APPLICAZIONE PCA ===")
    
    original_features = X_preprocessed.shape[1]
    n_samples = len(X_preprocessed)
    n_components = min(PCA_COMPONENTS, original_features, n_samples)
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_SEED)
            X_pca = pca.fit_transform(X_preprocessed)
            
            # VERIFICA: Output senza NaN/inf e dimensioni corrette
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError("PCA server ha prodotto output con NaN o inf")
            if X_pca.shape[1] != n_components:
                raise ValueError(f"PCA server output shape inconsistente: {X_pca.shape[1]} vs {n_components}")
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            print(f"[Server] ✅ PCA fissa server applicata: {X_pca.shape}")
            print(f"[Server] Varianza spiegata: {variance_explained*100:.2f}%")
            return X_pca
        
    except Exception as e:
        print(f"[Server] ERRORE PCA fissa server: {e}")
        print(f"[Server] Attivazione fallback...")
        n_fallback = min(n_components, original_features)
        X_fallback = X_preprocessed[:, :n_fallback]
        print(f"[Server] ✅ Fallback server: {X_fallback.shape}")
        return X_fallback

def apply_preprocessing_pipeline(X_global):
    """
    Applica la stessa pipeline di preprocessing dei client sui dati globali del server.
    Pipeline:
      - Pulizia inf/NaN
      - Clipping outlier per quantili (feature-wise)
      - Imputazione mediana
      - Rimozione feature quasi-costanti
      - Scaling standard
      - PCA fissa
    """
    # Imposta semi per riproducibilità PCA
    set_reproducibility_seeds()
    
    print(f"[Server] === PIPELINE PREPROCESSING SERVER ===")
    print(f"Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")
    
    # STEP 1: Pulizia inf/NaN
    if ENABLE_CLEAN_INF_NAN:
        X_cleaned = clean_data_for_pca(X_global)
    else:
        X_cleaned = X_global if hasattr(X_global, 'values') else X_global
    
    # STEP 2: Clipping outlier feature-wise
    if ENABLE_CLIPPING_OUTLIERS:
        X_clipped = clip_outliers_iqr(np.array(X_cleaned, dtype=float))
    else:
        X_clipped = X_cleaned
    
    # STEP 3: Imputazione mediana
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_clipped)
    else:
        X_imputed = X_clipped
    
    # STEP 4: Rimozione feature quasi-costanti (CONDIZIONALE)
    if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES:
        X_features_filtered, keep_mask = remove_near_constant_features(X_imputed)
        print(f"Feature conservate: {np.sum(keep_mask)}/{len(keep_mask)}")
    else:
        X_features_filtered = X_imputed
    
    # STEP 5: Scaling standard (CONDIZIONALE)
    if ENABLE_SCALING:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features_filtered)
    else:
        X_scaled = X_features_filtered
    
    # STEP 6: PCA fissa (CONDIZIONALE)
    if ENABLE_PCA:
        X_final = apply_pca(X_scaled)
    else:
        X_final = X_scaled
    
    print(f"[Server] ✅ Preprocessing completato: {X_final.shape}")
    return X_final

# ============== FUNZIONI PER DESERIALIZZAZIONE ALBERI ==============
def deserialize_tree_from_bytes(tree_bytes):
    """
    Deserializza un albero di decisione da bytes.
    
    Args:
        tree_bytes: bytes dell'albero serializzato con pickle
        
    Returns:
        DecisionTreeClassifier deserializzato
    """
    try:
        tree = pickle.loads(tree_bytes)
        return tree
    except Exception as e:
        print(f"Errore deserializzazione albero: {e}")
        raise

def deserialize_trees_from_client(parameters):
    """
    Deserializza gli alberi ricevuti da un client in formato Flower.
    
    Args:
        parameters: lista di numpy arrays dal client
        Format per ogni array: [tree_index, tree_size, accuracy, weighted_accuracy, tree_data...]
        
    Returns:
        lista di dizionari con alberi deserializzati e metadati
    """
    trees_data = []
    
    for tree_params in parameters:
        try:
            # Estrai metadati
            tree_index = int(tree_params[0])
            tree_size = int(tree_params[1])
            accuracy = float(tree_params[2])
            weighted_accuracy = float(tree_params[3])
            tree_data_array = tree_params[4:].astype(np.uint8)
            
            # Deserializza l'albero
            tree_bytes = tree_data_array.tobytes()
            tree = deserialize_tree_from_bytes(tree_bytes)
            
            tree_dict = {
                'tree_index': tree_index,
                'tree': tree,
                'accuracy': accuracy,
                'weighted_accuracy': weighted_accuracy,
                'tree_size': tree_size
            }
            
            trees_data.append(tree_dict)
            
        except Exception as e:
            print(f"⚠️ Errore deserializzazione albero {tree_index if 'tree_index' in locals() else 'unknown'}: {e}")
            continue
    
    return trees_data

# ============== FUNZIONI PER AGGREGAZIONE ALBERI ==============
def select_trees_per_forest(all_trees_by_client, method='weighted_accuracy', max_trees_per_client=None):
    """
    Selezione alberi migliori da ogni Random Forest client (S_DTs_A o S_DTs_WA).
    
    Args:
        all_trees_by_client: dict {client_id: [lista alberi]}
        method: 'accuracy' o 'weighted_accuracy'
        max_trees_per_client: numero massimo di alberi da selezionare per client
        
    Returns:
        lista di alberi selezionati
    """
    selected_trees = []
    metric_key = method  # 'accuracy' o 'weighted_accuracy'
    
    print(f"\n=== SELEZIONE ALBERI PER FOREST (metodo: {method}) ===")
    
    for client_id, trees in all_trees_by_client.items():
        if not trees:
            continue
        
        # Ordina gli alberi per metrica decrescente
        sorted_trees = sorted(trees, key=lambda x: x[metric_key], reverse=True)
        
        # Seleziona i migliori alberi
        if max_trees_per_client:
            selected = sorted_trees[:max_trees_per_client]
        else:
            selected = sorted_trees
        
        selected_trees.extend(selected)
        
        avg_metric = np.mean([t[metric_key] for t in selected])
        print(f"  Client {client_id}: selezionati {len(selected)}/{len(trees)} alberi, "
              f"avg {method}: {avg_metric:.4f}")
    
    print(f"✅ Totale alberi selezionati: {len(selected_trees)}")
    return selected_trees

def select_trees_global(all_trees_by_client, method='weighted_accuracy', max_trees_global=100):
    """
    Selezione alberi migliori globalmente tra tutti i client (S_DTs_A_All o S_DTs_WA_All).
    
    Args:
        all_trees_by_client: dict {client_id: [lista alberi]}
        method: 'accuracy' o 'weighted_accuracy'
        max_trees_global: numero massimo di alberi da selezionare globalmente
        
    Returns:
        lista di alberi selezionati
    """
    print(f"\n=== SELEZIONE ALBERI GLOBALE (metodo: {method}) ===")
    
    # Raccogli tutti gli alberi con annotazione del client di origine
    all_trees = []
    for client_id, trees in all_trees_by_client.items():
        for tree in trees:
            tree['client_id'] = client_id
            all_trees.append(tree)
    
    print(f"Totale alberi disponibili: {len(all_trees)}")
    
    # Ordina per metrica decrescente
    metric_key = method
    sorted_trees = sorted(all_trees, key=lambda x: x[metric_key], reverse=True)
    
    # Seleziona i migliori N alberi
    selected_trees = sorted_trees[:max_trees_global]
    
    # Statistiche
    avg_metric = np.mean([t[metric_key] for t in selected_trees])
    print(f"Selezionati top {len(selected_trees)} alberi globalmente")
    print(f"Media {method}: {avg_metric:.4f}")
    print(f"Range {method}: {selected_trees[-1][metric_key]:.4f} - {selected_trees[0][metric_key]:.4f}")
    
    # Distribuzione per client
    client_counts = {}
    for tree in selected_trees:
        cid = tree['client_id']
        client_counts[cid] = client_counts.get(cid, 0) + 1
    
    print(f"Distribuzione per client: {client_counts}")
    
    return selected_trees

def create_global_random_forest(selected_trees, ensemble_method='weighted_voting'):
    """
    Crea un Global Random Forest dai alberi selezionati.
    
    Args:
        selected_trees: lista di dizionari con alberi e metadati
        ensemble_method: 'simple_voting' o 'weighted_voting'
        
    Returns:
        RandomForestClassifier con alberi aggregati
    """
    print(f"\n=== COSTRUZIONE GLOBAL RANDOM FOREST ===")
    print(f"Numero alberi: {len(selected_trees)}")
    print(f"Metodo ensemble: {ensemble_method}")
    
    # Crea un nuovo RandomForestClassifier vuoto
    global_rf = RandomForestClassifier(
        n_estimators=len(selected_trees),
        random_state=RANDOM_SEED,
        warm_start=False
    )
    
    # Imposta gli alberi selezionati come estimators
    global_rf.estimators_ = [tree_dict['tree'] for tree_dict in selected_trees]
    global_rf.n_estimators = len(global_rf.estimators_)
    
    # Per weighted voting, salva i pesi (accuracy o weighted_accuracy)
    if ensemble_method == 'weighted_voting':
        # Salva i pesi come attributo custom
        weights = np.array([tree_dict['weighted_accuracy'] for tree_dict in selected_trees])
        # Normalizza i pesi
        weights = weights / np.sum(weights)
        global_rf.tree_weights_ = weights
        print(f"Pesi alberi salvati (media: {np.mean(weights):.4f})")
    
    # Imposta altri attributi necessari
    global_rf.n_classes_ = 2  # Binary classification
    global_rf.classes_ = np.array([0, 1])
    global_rf.n_outputs_ = 1
    
    print(f"✅ Global Random Forest creato con {len(global_rf.estimators_)} alberi")
    
    return global_rf

def serialize_trees_for_clients(global_rf, X_val, y_val):
    """
    Serializza il Global Random Forest per inviarlo ai client.
    
    Args:
        global_rf: RandomForestClassifier globale
        X_val: dati di validazione per calcolare metriche
        y_val: etichette di validazione
        
    Returns:
        lista di numpy arrays in formato Flower
    """
    parameters = []
    
    print(f"\n=== SERIALIZZAZIONE GLOBAL RF PER CLIENT ===")
    
    for idx, tree in enumerate(global_rf.estimators_):
        try:
            # Calcola metriche per l'albero (se abbiamo dati di validazione)
            if X_val is not None and y_val is not None and len(X_val) > 0:
                y_pred = tree.predict(X_val)
                tree_accuracy = accuracy_score(y_val, y_pred)
                tree_weighted_accuracy = balanced_accuracy_score(y_val, y_pred)
            else:
                # Usa valori di default o dalla cache se disponibile
                tree_accuracy = getattr(global_rf, 'tree_weights_', [1.0] * len(global_rf.estimators_))[idx]
                tree_weighted_accuracy = tree_accuracy
            
            # Serializza l'albero
            tree_bytes = pickle.dumps(tree)
            tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
            
            # Crea il formato Flower
            tree_params = np.concatenate([
                np.array([idx], dtype=np.float32),
                np.array([len(tree_array)], dtype=np.float32),
                np.array([tree_accuracy], dtype=np.float32),
                np.array([tree_weighted_accuracy], dtype=np.float32),
                tree_array.astype(np.float32)
            ])
            
            parameters.append(tree_params)
            
        except Exception as e:
            print(f"⚠️ Errore serializzazione albero {idx}: {e}")
            continue
    
    print(f"✅ {len(parameters)} alberi serializzati per i client")
    
    return parameters

# ============== FUNZIONE DI VALUTAZIONE GLOBALE ==============
def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid Random Forest.
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Usa preprocessing identico ai client.
        """
        # Imposta semi per riproducibilità del preprocessing
        set_reproducibility_seeds()
        
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER ===")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Costruzione path ai file CSV (client 14-15 come nel DNN)
        test_clients = [14, 15]
        df_list = []
        
        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
            
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"File data{client_id}.csv non trovato")
                continue
        
        if not df_list:
            # Fallback
            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(200, len(df_fallback)), random_state=42)]
                print(f"Usando fallback con {len(df_list[0])} campioni da data1.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per valutazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        # Statistiche distribuzione
        attack_samples = y_global.sum()
        natural_samples = (y_global == 0).sum()
        attack_ratio = y_global.mean()
        
        print(f"Dataset test globale: {len(df_global)} campioni")
        print(f"Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
        
        # Applica pipeline preprocessing
        X_global_final = apply_preprocessing_pipeline(X_global)
        
        print(f"Dataset preprocessato: {len(X_global_final)} campioni, {X_global_final.shape[1]} feature")
        
        return X_global_final, y_global, {
            'total_samples': len(df_global),
            'attack_samples': attack_samples,
            'natural_samples': natural_samples,
            'attack_ratio': attack_ratio
        }
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, dataset_info = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi
        X_global = np.random.random((100, 74 if ENABLE_PCA else 128))
        y_global = np.random.randint(0, 2, 100)
        dataset_info = {}
        print(f"Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round.
        """
        # Imposta semi per riproducibilità
        set_reproducibility_seeds()
        
        print(f"\n=== VALUTAZIONE GLOBALE RF - ROUND {server_round + 1} ===")
        
        try:
            # Deserializza il Global Random Forest
            if len(parameters) == 0:
                print("⚠️ Nessun parametro ricevuto per valutazione")
                return 1.0, {"accuracy": 0.0, "error": "no_parameters"}
            
            print(f"Deserializzazione Global RF da {len(parameters)} parametri...")
            trees_data = []
            for tree_params in parameters:
                tree_index = int(tree_params[0])
                tree_size = int(tree_params[1])
                accuracy = float(tree_params[2])
                weighted_accuracy = float(tree_params[3])
                tree_data_array = tree_params[4:].astype(np.uint8)
                
                # Deserializza l'albero
                tree_bytes = tree_data_array.tobytes()
                tree = pickle.loads(tree_bytes)
                
                tree_dict = {
                    'tree_index': tree_index,
                    'tree': tree,
                    'accuracy': accuracy,
                    'weighted_accuracy': weighted_accuracy,
                    'tree_size': tree_size
                }
                trees_data.append(tree_dict)
            
            # Ricostruisci Random Forest
            global_rf = RandomForestClassifier(
                n_estimators=len(trees_data),
                random_state=RANDOM_SEED
            )
            global_rf.estimators_ = [t['tree'] for t in trees_data]
            global_rf.n_estimators = len(global_rf.estimators_)
            global_rf.n_classes_ = 2
            global_rf.classes_ = np.array([0, 1])
            global_rf.n_outputs_ = 1
            
            print(f"✅ Global RF ricostruito con {len(global_rf.estimators_)} alberi")
            
            # Valutazione sul dataset test globale
            y_pred = global_rf.predict(X_global)
            y_pred_proba = global_rf.predict_proba(X_global)[:, 1]
            
            # Calcola metriche
            accuracy = accuracy_score(y_global, y_pred)
            precision = precision_score(y_global, y_pred, zero_division=0)
            recall = recall_score(y_global, y_pred, zero_division=0)
            f1 = f1_score(y_global, y_pred, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_global, y_pred)
            
            # AUC
            try:
                auc = roc_auc_score(y_global, y_pred_proba)
            except:
                auc = 0.0
            
            # Loss simulata (1 - accuracy)
            loss = 1.0 - accuracy
            
            # Classification report e confusion matrix
            report = classification_report(
                y_global, y_pred, 
                target_names=["natural", "attack"], 
                output_dict=True, 
                zero_division=0
            )
            conf_matrix = confusion_matrix(y_global, y_pred)
            
            print(f"RISULTATI VALUTAZIONE:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1:.4f} ({f1*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  Campioni test: {len(X_global)}")
            
            print(f"Classification report (per classe):")
            print(classification_report(
                y_global, y_pred, 
                target_names=["natural", "attack"], 
                zero_division=0
            ))
            print(f"Confusion matrix:")
            print(conf_matrix)
            
            # Raccolta metriche per report
            metric_row = {
                "round": server_round + 1,
                "loss_distribuita": float(loss),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "auc": float(auc),
                "f1_score": float(f1),
                "f1_natural": report["natural"]["f1-score"],
                "f1_attack": report["attack"]["f1-score"],
                "precision": float(precision),
                "precision_natural": report["natural"]["precision"],
                "precision_attack": report["attack"]["precision"],
                "recall": float(recall),
                "recall_natural": report["natural"]["recall"],
                "recall_attack": report["attack"]["recall"],
            }
            
            # Salva metriche globali
            global all_federated_metrics, last_confusion_matrix
            all_federated_metrics.append(metric_row)
            last_confusion_matrix = conf_matrix
            
            return float(loss), {
                # Metriche base
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "f1_score": float(f1),
                "balanced_accuracy": float(balanced_acc),
                
                # Metriche per classe
                "precision_natural": report["natural"]["precision"],
                "recall_natural": report["natural"]["recall"],
                "f1_natural": report["natural"]["f1-score"],
                "precision_attack": report["attack"]["precision"],
                "recall_attack": report["attack"]["recall"],
                "f1_attack": report["attack"]["f1-score"],
                "support_natural": report["natural"]["support"],
                "support_attack": report["attack"]["support"],
                
                # Confusion matrix
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1]),
                
                # Info dataset
                "global_test_samples": int(len(X_global)),
                "attack_samples": int(dataset_info.get('attack_samples', 0)),
                "natural_samples": int(dataset_info.get('natural_samples', 0)),
                "attack_ratio": float(dataset_info.get('attack_ratio', 0)),
                "n_trees": len(global_rf.estimators_),
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {
                "accuracy": 0.0, 
                "error": str(e), 
                "global_test_samples": 0
            }
    
    return evaluate

def save_federated_metrics_report(metrics_list):
    """
    Salva un report completo delle metriche federate in un file di testo.
    """
    if not metrics_list:
        print("[SERVER] ⚠️ Nessuna metrica da salvare.")
        return
    
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, f"metrics_RF_complete_report_{timestamp}.txt")
    
    cols = [
        ("round", "Round", 6),
        ("loss_distribuita", "Loss", 11),
        ("accuracy", "Accuracy", 11),
        ("balanced_accuracy", "BalancedAcc", 13),
        ("auc", "AUC", 9),
        ("f1_score", "F1_Score", 11),
        ("f1_natural", "F1_Natural", 11),
        ("f1_attack", "F1_Attack", 11),
        ("precision", "Precision", 11),
        ("precision_natural", "Precision_Nat", 14),
        ("precision_attack", "Precision_Att", 14),
        ("recall", "Recall", 11),
        ("recall_natural", "Recall_Nat", 11),
        ("recall_attack", "Recall_Att", 11),
    ]
    
    with open(report_path, 'w') as f:
        f.write("="*150 + "\n")
        f.write("RANDOM FOREST FEDERATED LEARNING - METRICHE COMPLETE\n")
        f.write("="*150 + "\n\n")
        
        # Header
        header = " | ".join([f"{label:^{width}}" for _, label, width in cols])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Righe
        for metric_row in metrics_list:
            row = " | ".join([
                f"{metric_row.get(key, 0):^{width}.4f}" if isinstance(metric_row.get(key, 0), float) 
                else f"{metric_row.get(key, 0):^{width}d}"
                for key, _, width in cols
            ])
            f.write(row + "\n")
        
        # Ultima confusion matrix
        if last_confusion_matrix is not None:
            f.write("\n" + "="*150 + "\n")
            f.write("CONFUSION MATRIX (ULTIMO ROUND)\n")
            f.write("="*150 + "\n")
            f.write(str(last_confusion_matrix) + "\n")
        
        f.write("\n" + "="*150 + "\n")
        f.write(f"Report generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*150 + "\n")
    
    print(f"\n[SERVER] ✅ Tabella metriche federate RF salvata in: {report_path}")

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round.
    """
    print(f"\n=== METRICHE CLIENT ===")
    
    for _, fit_res in fit_results:
        metrics = fit_res.metrics
        client_id = metrics.get('client_id', '?')
        train_acc = metrics.get('train_accuracy', 0.0)
        train_f1 = metrics.get('train_f1_score', 0.0)
        n_trees = metrics.get('n_trees_sent', 0)
        train_samples = metrics.get('train_samples', 0)
        
        print(f"  Client {client_id}: Accuracy={train_acc:.4f}, F1={train_f1:.4f}, "
              f"Trees={n_trees}, Samples={train_samples}")

# ============== STRATEGIA FEDERATED AVERAGING PER RANDOM FOREST ==============
class SmartGridRandomForestFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid Random Forest.
    Implementa l'aggregazione degli alberi basata sul paper.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega gli alberi di Random Forest dai client.
        Implementa i metodi di selezione del paper.
        """
        # Imposta semi per riproducibilità
        set_reproducibility_seeds()
        
        print(f"\n=== AGGREGAZIONE TRAINING RF - ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("ERRORE: Nessun client ha fornito risultati validi")
            return None
        
        # Stampa metriche dei client
        print_client_metrics(results)
        
        try:
            # Deserializza alberi da tutti i client
            all_trees_by_client = {}
            
            for client_idx, (client_proxy, fit_res) in enumerate(results):
                client_id = fit_res.metrics.get('client_id', client_idx)
                parameters = fit_res.parameters
                
                # Converti Parameters in lista di numpy arrays se necessario
                if isinstance(parameters, Parameters):
                    parameters = parameters.tensors
                
                print(f"\nDeserializzazione alberi da Client {client_id}...")
                trees = deserialize_trees_from_client(parameters)
                
                if trees:
                    all_trees_by_client[client_id] = trees
                    avg_acc = np.mean([t['accuracy'] for t in trees])
                    avg_wacc = np.mean([t['weighted_accuracy'] for t in trees])
                    print(f"  ✅ {len(trees)} alberi ricevuti, avg accuracy: {avg_acc:.4f}, "
                          f"avg weighted_accuracy: {avg_wacc:.4f}")
                else:
                    print(f"  ⚠️ Nessun albero valido ricevuto")
            
            if not all_trees_by_client:
                print("ERRORE: Nessun albero valido ricevuto da alcun client")
                return None
            
            # Selezione alberi basata sulla strategia configurata
            if TREE_AGGREGATION_STRATEGY == 'per_forest':
                # Seleziona i migliori alberi da ogni client
                trees_per_client = MAX_TREES_GLOBAL // len(all_trees_by_client)
                selected_trees = select_trees_per_forest(
                    all_trees_by_client, 
                    method=TREE_SELECTION_METHOD,
                    max_trees_per_client=trees_per_client
                )
            else:  # 'global'
                # Seleziona i migliori alberi globalmente
                selected_trees = select_trees_global(
                    all_trees_by_client,
                    method=TREE_SELECTION_METHOD,
                    max_trees_global=MAX_TREES_GLOBAL
                )
            
            if not selected_trees:
                print("ERRORE: Nessun albero selezionato")
                return None
            
            # Crea il Global Random Forest
            global_rf = create_global_random_forest(selected_trees, ENSEMBLE_METHOD)
            
            # Serializza il Global RF per inviarlo ai client
            # Usa dataset di test globale per calcolare metriche
            try:
                X_val, y_val, _ = self._get_test_data()
            except:
                X_val, y_val = None, None
            
            parameters = serialize_trees_for_clients(global_rf, X_val, y_val)
            
            # Converti in formato Parameters di Flower
            aggregated_parameters = Parameters(tensors=parameters, tensor_type="numpy.ndarray")
            
            # Calcola metriche aggregate
            aggregated_metrics = {}
            
            print(f"✅ Aggregazione completata per round {server_round}")
            print(f"✅ Global RF con {len(selected_trees)} alberi creato e serializzato")
            
            return aggregated_parameters, aggregated_metrics
            
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione Random Forest.
        """
        # Imposta semi per riproducibilità
        set_reproducibility_seeds()
        
        print(f"\n=== AGGREGAZIONE VALUTAZIONE RF - ROUND {server_round} ===")
        print(f"Client che hanno valutato: {len(results)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione completata per round {server_round}")
            else:
                print(f"Aggregazione valutazione non riuscita per round {server_round}")
            
        except Exception as e:
            print(f"ERRORE durante aggregazione valutazione: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result
    
    def _get_test_data(self):
        """
        Helper per ottenere dataset di test (usato per metriche alberi).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_clients = [14, 15]
        df_list = []
        
        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
            except:
                continue
        
        if df_list:
            df_global = pd.concat(df_list, ignore_index=True)
            X = df_global.drop(columns=["marker"])
            y = (df_global["marker"] != "Natural").astype(int)
            X_processed = apply_preprocessing_pipeline(X)
            return X_processed, y, {}
        else:
            return None, None, {}

# ============== MAIN ==============
def main():
    """
    Funzione principale per avviare il server SmartGrid Random Forest federato.
    """
    # Imposta semi per riproducibilità
    set_reproducibility_seeds()
    
    print("=== SERVER FEDERATO SMARTGRID RANDOM FOREST ===")
    print("Configurazione:")
    print("  - Rounds: " + str(NUM_ROUNDS))
    print("  - Client minimi: 2")
    print("  - Strategia: Random Forest Aggregation")
    print("  - Valutazione: Dataset globale " + ("con PCA fissa (client 14-15)" if ENABLE_PCA else "senza PCA (client 14-15)"))
    print("  - Pipeline: Pulizia → Imputazione → Normalizzazione → " + ("PCA fissa" if ENABLE_PCA else "nessuna riduzione"))
    print(f"  - Tree Selection Method: {TREE_SELECTION_METHOD}")
    print(f"  - Tree Aggregation Strategy: {TREE_AGGREGATION_STRATEGY}")
    print(f"  - Max Trees Global: {MAX_TREES_GLOBAL}")
    print(f"  - Ensemble Method: {ENSEMBLE_METHOD}")
    
    # Configurazione del server
    config = fl.server.ServerConfig(NUM_ROUNDS)
    
    # Strategia Random Forest Federated Averaging
    strategy = SmartGridRandomForestFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_smartgrid_evaluate_fn()
    )
    
    print(f"\nServer Random Forest in attesa di client su localhost:8080...")
    print("Per connettere i client, esegui:")
    print("  python clientRF.py 1")
    print("  python clientRF.py 2")
    print("  ...")
    print("  python clientRF.py 13")
    print("\nClient 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    print("")
    
    try:
        # Avvia il server Flower
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
        
        global all_federated_metrics
        if all_federated_metrics:
            save_federated_metrics_report(all_federated_metrics)
        else:
            print("[SERVER] ⚠️ Nessuna metrica federata disponibile per il report finale.")
        
    except Exception as e:
        print(f"Errore durante l'avvio del server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
