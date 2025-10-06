import flwr as fl
import pandas as pd
import numpy as np
import sys
import os
import warnings
import pickle
import joblib
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
warnings.filterwarnings('ignore')

# CONFIGURAZIONE SEMI PER RIPRODUCIBILITÀ
RANDOM_SEED = 42

# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==============
ENABLE_CLEAN_INF_NAN = False           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = False       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = False              # Imputazione mediana
ENABLE_SCALING = False                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False  # PCA per riduzione dimensionalità

if ENABLE_PCA:
    ENABLE_IMPUTATION = True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 74  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 42  # Seme specifico per PCA
  
# Quando PCA disabilitata, disabilita rimozione feature quasi-costanti per compatibilità dei modelli
if ENABLE_PCA == False:
    ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False
    PCA_COMPONENTS = None

# CONFIGURAZIONE MODELLO RANDOM FOREST
# Basato sui risultati del paper: hyperparameter tuning per ottimizzare performance
RF_N_ESTIMATORS = 65      # Numero di alberi nella foresta (dal paper: ottimo tra 65-93)
RF_MAX_DEPTH = None       # Profondità massima degli alberi (None = illimitata)
RF_MIN_SAMPLES_SPLIT = 2  # Campioni minimi per effettuare uno split
RF_MIN_SAMPLES_LEAF = 1   # Campioni minimi in una foglia
RF_MAX_FEATURES = 'sqrt'  # Feature da considerare per ogni split ('sqrt' dal paper)
RF_BOOTSTRAP = True       # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced'  # Gestione automatica dello sbilanciamento
RF_CRITERION = 'entropy'  # Criterio di splitting (dal paper: entropy migliore di gini per molti dataset)

# CONFIGURAZIONE ENSEMBLE PER FEDERATED RANDOM FOREST
# Basato sulla metodologia del paper per aggregazione degli alberi
ENSEMBLE_METHOD = 'weighted_voting'  # 'simple_voting' o 'weighted_voting'
TREE_SELECTION_METHOD = 'accuracy_based'  # Come selezionare i migliori alberi per l'aggregazione

NUM_ROUNDS = 100  # Numero di round di addestramento federato

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

def fit_clip_outliers_iqr(X, k=5.0):
    """
    Calcola i limiti inferiori e superiori per ogni feature
    usando la regola dei quantili (IQR) sul dataset fornito (tipicamente il training).
    Ritorna due array: lower e upper.
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

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
    """
    Rimuove le feature che sono costanti almeno al 99.9% (tutte uguali tranne lo 0.1%).
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
    Pulizia robusta dei dati per prevenire problemi numerici in PCA:
    - Sostituisce inf/-inf con NaN
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X_preprocessed, client_id=None):
    """
    Applica PCA con numero FISSO di componenti.
    """
    print(f"[Client {client_id}] === APPLICAZIONE PCA ===")

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
                raise ValueError(f"PCA client {client_id} ha prodotto output con NaN o inf")
            if X_pca.shape[1] != n_components:
                raise ValueError(f"PCA output shape inconsistente: {X_pca.shape[1]} vs {n_components}")
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            print(f"[Client {client_id}] ✅ PCA fissa applicata: {X_pca.shape}")
            print(f"[Client {client_id}] Varianza spiegata: {variance_explained*100:.2f}%")
            return X_pca
        
    except Exception as e:
        print(f"[Client {client_id}] ERRORE PCA: {e}")
        print(f"[Client {client_id}] Attivazione fallback semplificato...")
        n_fallback = min(n_components, original_features)
        X_fallback = X_preprocessed[:, :n_fallback]
        print(f"[Client {client_id}] ✅ Fallback: {X_fallback.shape}")
        return X_fallback

def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico.
    Applica preprocessing minimale per Random Forest (che gestisce bene dati raw).
    """
    # Imposta semi per riproducibilità del preprocessing
    set_reproducibility_seeds()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")
    
    df = pd.read_csv(file_path)
    print(f"=== PREPROCESSING FEDERATO RANDOM FOREST ===")
    print(f"Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")

    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"[Client {client_id}] Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
    
    # Random Forest gestisce bene i dati raw, quindi preprocessing minimale
    # STEP 1: Pulizia inf/NaN (solo se abilitata)
    if ENABLE_CLEAN_INF_NAN:
        X_cleaned = clean_data_for_pca(X)
    else:
        X_cleaned = X.values if hasattr(X, 'values') else X

    # Suddivisione train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_cleaned, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"[Client {client_id}] Suddivisione: {len(X_train_raw)} training, {len(X_val_raw)} validation")

    # STEP 2: Clipping outlier per quantili (solo se abilitato)
    if ENABLE_CLIPPING_OUTLIERS:
        X_train_np = np.array(X_train_raw, dtype=float)
        X_val_np = np.array(X_val_raw, dtype=float)
        lower, upper = fit_clip_outliers_iqr(X_train_np, k=5.0)
        X_train_clipped = transform_clip_outliers_iqr(X_train_np, lower, upper)
        X_val_clipped = transform_clip_outliers_iqr(X_val_np, lower, upper)
    else:
        X_train_clipped = X_train_raw
        X_val_clipped = X_val_raw

    # STEP 3: Imputazione mediana (solo se abilitata)
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_val_imputed = imputer.transform(X_val_clipped)
    else:
        X_train_imputed = X_train_clipped
        X_val_imputed = X_val_clipped

    # STEP 4: Rimozione feature quasi-costanti (solo se abilitata)
    if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES:
        X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
        X_val_reduced = X_val_imputed[:, keep_mask]
        print(f"[Client {client_id}] Feature dopo rimozione quasi-costanti: {X_train_reduced.shape[1]} (da {X_train_imputed.shape[1]})")
    else:
        X_train_reduced = X_train_imputed
        X_val_reduced = X_val_imputed
        print(f"[Client {client_id}] Rimozione feature quasi-costanti DISABILITATA - mantenute {X_train_reduced.shape[1]} feature")

    # STEP 5: Scaling standard (solo se abilitato)
    if ENABLE_SCALING:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reduced)
        X_val_scaled = scaler.transform(X_val_reduced)
        print(f"[Client {client_id}] Preprocessing completato (clipping, imputazione, {('costanti, ' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else '')}scaling)")
    else:
        X_train_scaled = X_train_reduced
        X_val_scaled = X_val_reduced
        print(f"[Client {client_id}] Scaling DISABILITATO - usando dati preprocessati direttamente: {X_train_scaled.shape}")

    # STEP 6: PCA (solo se abilitata)
    if ENABLE_PCA:
        X_train_final = apply_pca(X_train_scaled, client_id=client_id)
        X_val_final = apply_pca(X_val_scaled, client_id=client_id)
        expected_features = PCA_COMPONENTS
        if X_train_final.shape[1] != expected_features:
            raise RuntimeError(f"Client {client_id}: PCA output shape inconsistente: {X_train_final.shape} vs {expected_features}")
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        print(f"[Client {client_id}] PCA DISABILITATA - usando dati scalati direttamente: {X_train_final.shape}")
        
    # Info dataset
    dataset_info = {
        'client_id': client_id,
        'total_samples': len(df),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio,
        'train_attack_ratio': y_train.mean(),
        'val_attack_ratio': y_val.mean(),
        'original_features': X.shape[1],
        'final_features': X_train_final.shape[1],
        'pca_enabled': ENABLE_PCA,
        'remove_near_constant_enabled': ENABLE_REMOVE_NEAR_CONSTANT_FEATURES,
        'pca_components_fixed': PCA_COMPONENTS if ENABLE_PCA else None,
        'preprocessing_method': f"minimal_for_rf{'_pca' if ENABLE_PCA else ''}",
        'compatibility_guaranteed': True
    }
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO ===")
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_random_forest_model():
    """
    Crea il modello Random Forest per SmartGrid.
    Implementa la configurazione ottimale basata sul paper.
    
    Returns:
        Modello RandomForestClassifier configurato secondo i risultati del paper
    """
    # Imposta semi per riproducibilità del modello
    set_reproducibility_seeds()

    print(f"[Client] === CREAZIONE RANDOM FOREST ===")
    print(f"[Client] Modello: Random Forest con {RF_N_ESTIMATORS} alberi")
    print(f"[Client] Criterio: {RF_CRITERION} (dal paper: migliore per molti dataset)")
    print(f"[Client] Max features: {RF_MAX_FEATURES} (feature selection automatica)")
    print(f"[Client] Class weight: {RF_CLASS_WEIGHT} (gestione sbilanciamento)")
    
    # PARAMETRI OTTIMIZZATI BASATI SUL PAPER
    # Il paper mostra che entropy come criterio e sqrt per max_features 
    # danno risultati migliori sui dataset di intrusion detection
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,           # Numero di alberi (dal paper: 65-93 range ottimo)
        criterion=RF_CRITERION,                 # Criterio di splitting (entropy vs gini)
        max_depth=RF_MAX_DEPTH,                 # Profondità massima degli alberi
        min_samples_split=RF_MIN_SAMPLES_SPLIT, # Campioni minimi per split
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,   # Campioni minimi per foglia
        max_features=RF_MAX_FEATURES,           # Feature da considerare per ogni split
        bootstrap=RF_BOOTSTRAP,                 # Bootstrap sampling
        random_state=RANDOM_SEED,               # Per riproducibilità
        n_jobs=-1,                              # Usa tutti i core disponibili
        class_weight=RF_CLASS_WEIGHT,           # Gestione automatica dello sbilanciamento
        oob_score=True                          # Calcola out-of-bag score per validazione
    )
    
    print(f"[Client] Parametri Random Forest:")
    print(f"  - N. estimatori: {RF_N_ESTIMATORS}")
    print(f"  - Criterio: {RF_CRITERION}")
    print(f"  - Max depth: {RF_MAX_DEPTH}")
    print(f"  - Min samples split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  - Min samples leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  - Max features: {RF_MAX_FEATURES}")
    print(f"  - Bootstrap: {RF_BOOTSTRAP}")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED}")
    print(f"  - OOB Score: True")
    
    return model

def extract_trees_from_forest(model, X_val, y_val):
    """
    Estrae gli alberi dal Random Forest e calcola le loro performance individuali.
    Implementa la metodologia del paper per la selezione degli alberi migliori.
    
    Args:
        model: Random Forest addestrato
        X_val: Dati di validazione
        y_val: Etichette di validazione
        
    Returns:
        Lista di tuple (tree, accuracy, weighted_accuracy) per ogni albero
    """
    print(f"[Client] === ESTRAZIONE ALBERI DA RANDOM FOREST ===")
    
    trees_performance = []
    
    for i, tree in enumerate(model.estimators_):
        # Predizioni dell'albero singolo
        tree_predictions = tree.predict(X_val)
        
        # Calcola accuracy standard
        accuracy = accuracy_score(y_val, tree_predictions)
        
        # Calcola weighted accuracy (come nel paper)
        # Weighted accuracy considera la distribuzione delle classi
        class_counts = np.bincount(y_val)
        weights = 1.0 / class_counts  # Peso inversamente proporzionale alla frequenza
        class_weights_norm = weights / weights.sum()  # Normalizza i pesi
        
        # Calcola accuracy pesata per classe
        weighted_acc = 0.0
        for class_label in np.unique(y_val):
            class_mask = (y_val == class_label)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_val[class_mask], tree_predictions[class_mask])
                weighted_acc += class_accuracy * class_weights_norm[class_label]
        
        trees_performance.append((tree, accuracy, weighted_acc))
        
        if i < 5:  # Stampa info per i primi 5 alberi
            print(f"[Client] Albero {i+1}: Accuracy={accuracy:.4f}, Weighted Accuracy={weighted_acc:.4f}")
    
    # Ordina gli alberi per performance (weighted accuracy come nel paper)
    trees_performance.sort(key=lambda x: x[2], reverse=True)  # Ordina per weighted accuracy
    
    print(f"[Client] Migliore albero: Accuracy={trees_performance[0][1]:.4f}, Weighted Accuracy={trees_performance[0][2]:.4f}")
    print(f"[Client] Peggiore albero: Accuracy={trees_performance[-1][1]:.4f}, Weighted Accuracy={trees_performance[-1][2]:.4f}")
    
    return trees_performance

def serialize_trees_for_aggregation(trees_performance, max_trees=None):
    """
    Serializza gli alberi per l'invio al server.
    Implementa la metodologia del paper per la selezione degli alberi da inviare.
    
    Args:
        trees_performance: Lista di tuple (tree, accuracy, weighted_accuracy)
        max_trees: Numero massimo di alberi da inviare (None = tutti)
        
    Returns:
        Lista di alberi serializzati con le loro performance
    """
    print(f"[Client] === SERIALIZZAZIONE ALBERI ===")
    
    if max_trees is not None:
        # Seleziona solo i migliori alberi basandosi sulla weighted accuracy
        selected_trees = trees_performance[:max_trees]
        print(f"[Client] Selezionati {len(selected_trees)} migliori alberi su {len(trees_performance)}")
    else:
        selected_trees = trees_performance
        print(f"[Client] Invio tutti i {len(selected_trees)} alberi")
    
    serialized_trees = []
    
    for i, (tree, accuracy, weighted_accuracy) in enumerate(selected_trees):
        try:
            # Serializza l'albero usando joblib (più efficiente di pickle per scikit-learn)
            buf = BytesIO()
            joblib.dump(tree, buf)
            tree_bytes = buf.getvalue()
            serialized_trees.append(tree_bytes)
            
        except Exception as e:
            print(f"[Client] Errore serializzazione albero {i}: {e}")
            continue
    
    print(f"[Client] Serializzati {len(serialized_trees)} alberi")
    return serialized_trees

# Variabili globali per il client
client_id = None
model = None
X_train = None
y_train = None
X_val = None
y_val = None
dataset_info = None

class SmartGridRandomForestClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con Random Forest.
    Implementa la metodologia del paper per l'aggregazione federata di Random Forest.
    """
    
    def get_parameters(self, config):
        """
        Per Random Forest, non restituiamo pesi neurali ma gli alberi del modello.
        Questo è diverso dalle reti neurali - implementiamo la metodologia del paper.
        """
        global model, X_val, y_val
    
        if model is None:
            print(f"[Client {client_id}] Modello non ancora addestrato, restituisco parametri vuoti")
            return []

        try:
            # Estrai e valuta le performance degli alberi
            trees_performance = extract_trees_from_forest(model, X_val, y_val)

            # Serializza gli alberi per l'invio (invia tutti gli alberi)
            serialized_trees = serialize_trees_for_aggregation(trees_performance)
        
            # Converti in formato compatibile con Flower
            # Flower si aspetta una lista di array numpy
            parameters = []
            for tree_bytes in serialized_trees:
                # Convertiamo i bytes in numpy array per compatibilità
                tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
                parameters.append(tree_array)
        
            print(f"[Client {client_id}] Invio {len(parameters)} alberi al server")
            return parameters
        
        except Exception as e:
            print(f"[Client {client_id}] Errore nell'estrazione parametri: {e}")
            return []

    def set_parameters(self, parameters):
        """
        Per Random Forest federato, questo metodo riceve il modello aggregato dal server.
        Il server invia un nuovo Random Forest composto dai migliori alberi di tutti i client.
        """
        global model
    
        if not parameters or len(parameters) == 0:
            print(f"[Client {client_id}] Nessun parametro ricevuto dal server")
            return
    
        try:
            # Il server invia il modello Random Forest aggregato
            # Assumiamo che il primo parametro sia il modello serializzato
            if len(parameters) > 0:
                aggregated_model_bytes = parameters[0].tobytes()
                model = joblib.load(BytesIO(aggregated_model_bytes))
            
                print(f"[Client {client_id}] ✅ Modello aggregato ricevuto dal server")
                print(f"[Client {client_id}] Nuovo modello ha {model.n_estimators} alberi")
        
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore nell'impostazione parametri: {e}")
            # Mantieni il modello corrente in caso di errore
            pass

    def fit(self, parameters, config):
        """
        Addestra il modello Random Forest locale.
        """
        global model, X_train, y_train, dataset_info

        # Imposta semi per riproducibilità dell'addestramento
        set_reproducibility_seeds()
        
        print(f"[Client {client_id}] Round di addestramento Random Forest...")
        
        # Imposta parametri se ricevuti dal server
        if parameters:
            self.set_parameters(parameters)
        
        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training!")
            return [], 0, {}
        
        try:
            # Addestra il Random Forest locale
            print(f"[Client {client_id}] Addestramento Random Forest su {len(X_train)} campioni...")
            model.fit(X_train, y_train)
            
            # Calcola metriche di training
            train_predictions = model.predict(X_train)
            train_prob = model.predict_proba(X_train)[:, 1]  # Probabilità classe positiva
            
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_precision = precision_score(y_train, train_predictions, zero_division=0)
            train_recall = recall_score(y_train, train_predictions, zero_division=0)
            train_f1 = f1_score(y_train, train_predictions, zero_division=0)
            train_balanced_acc = balanced_accuracy_score(y_train, train_predictions)
            
            # AUC se abbiamo probabilità
            try:
                train_auc = roc_auc_score(y_train, train_prob)
            except:
                train_auc = 0.0
            
            # Out-of-bag score se disponibile
            oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else 0.0
            
            print(f"[Client {client_id}] Training completato!")
            print(f"[Client {client_id}] Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
            print(f"[Client {client_id}] Balanced Acc: {train_balanced_acc:.4f}, OOB Score: {oob_score:.4f}")
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante addestramento: {e}")
            return [], 0, {'error': f'training_failed: {str(e)}'}
        
        # Metriche da inviare al server
        metrics = {
            # Metriche base
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            'train_balanced_accuracy': float(train_balanced_acc),
            'train_auc': float(train_auc),
            'oob_score': float(oob_score),
            
            # Info modello
            'n_estimators': int(model.n_estimators),
            'n_features': int(model.n_features_in_),
            
            # Dataset info
            'client_id': int(dataset_info['client_id']),
            'train_samples': int(dataset_info['train_samples']),
        }
        
        # Restituisce gli alberi del modello addestrato
        return self.get_parameters(config), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello Random Forest.
        """
        global model, X_val, y_val

        # Imposta semi per riproducibilità della valutazione
        set_reproducibility_seeds()
        
        # Imposta parametri se ricevuti dal server
        if parameters:
            self.set_parameters(parameters)
        
        if model is None:
            print(f"[Client {client_id}] Modello non disponibile per valutazione")
            return 1.0, 0, {"accuracy": 0.0}
        
        if len(X_val) == 0:
            return 0.0, 0, {"accuracy": 0.0}
        
        try:
            # Valutazione Random Forest
            val_predictions = model.predict(X_val)
            val_prob = model.predict_proba(X_val)[:, 1]  # Probabilità classe positiva
            
            # Calcola metriche
            accuracy = accuracy_score(y_val, val_predictions)
            precision = precision_score(y_val, val_predictions, zero_division=0)
            recall = recall_score(y_val, val_predictions, zero_division=0)
            f1_score_val = f1_score(y_val, val_predictions, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_val, val_predictions)
            
            # AUC
            try:
                auc = roc_auc_score(y_val, val_prob)
            except:
                auc = 0.0
            
            # Metriche per classe
            report = classification_report(y_val, val_predictions, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_val, val_predictions)

            print(f"[Client {client_id}] Val Accuracy: {accuracy:.4f}, Val F1: {f1_score_val:.4f}")
            print(f"[Client {client_id}] Val Balanced Acc: {balanced_acc:.4f}, Val AUC: {auc:.4f}")
            print(f"[Client {client_id}] Classification report (per classe):")
            print(classification_report(y_val, val_predictions, target_names=["natural", "attack"], zero_division=0))
            print(f"[Client {client_id}] Confusion matrix:")
            print(f"tn: {conf_matrix[0, 0]}, fp: {conf_matrix[0, 1]}, fn: {conf_matrix[1, 0]}, tp: {conf_matrix[1, 1]}")
            
            # Simula loss per compatibilità (Random Forest non ha loss)
            loss = 1 - accuracy  # Loss simulata
            
            # Metriche
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "f1_score": f1_score_val,
                "balanced_accuracy": balanced_acc,
                "val_samples": len(X_val),
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
                "tp": int(conf_matrix[1, 1])
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante valutazione: {e}")
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

def main():
    """
    Funzione principale per avviare il client SmartGrid Random Forest.
    """

    global client_id, model, X_train, y_train, X_val, y_val, dataset_info

    # Imposta semi all'avvio del client
    set_reproducibility_seeds()
    
    if len(sys.argv) != 2:
        print("Uso: python clientRF.py <client_id>")
        print("Esempio: python clientRF.py 1")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("Client ID deve essere tra 1 e 13")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"=== AVVIO CLIENT RANDOM FOREST {client_id} ===")
    
    try:
        # Carica i dati con preprocessing minimale per Random Forest
        print(f"[Client {client_id}] Caricamento dati per Random Forest...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
        
        # Imposta semi all'avvio del client
        set_reproducibility_seeds()

        # Crea il modello Random Forest
        model = create_random_forest_model()

        print(f"[Client {client_id}] === RIASSUNTO CLIENT RANDOM FOREST ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['final_features']}")
        print(f"[Client {client_id}] Modello: Random Forest con {model.n_estimators} alberi")
        print(f"[Client {client_id}] Criterio: {model.criterion}, Max features: {model.max_features}")
        print(f"[Client {client_id}] Connessione al server su localhost:8080...")
        
        # Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridRandomForestClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()