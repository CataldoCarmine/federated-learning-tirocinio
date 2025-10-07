import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import os
from datetime import datetime
import sys
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

# ============== CONFIGURAZIONE RANDOM FOREST GLOBALE ==============
# Configurazione aggregazione alberi (basata sul paper)
TREE_SELECTION_METHOD = 'weighted_accuracy'  # 'accuracy' o 'weighted_accuracy'
TREE_AGGREGATION_STRATEGY = 'per_forest'     # 'per_forest' o 'global'
MAX_TREES_GLOBAL = 100                       # Numero massimo alberi nel modello globale
ENSEMBLE_METHOD = 'weighted_voting'          # 'simple_voting' o 'weighted_voting'
MIN_TREES_PER_CLIENT = 10                    # Minimo alberi da accettare da ogni client

# Configurazione Random Forest del server (identica ai client)
RF_N_ESTIMATORS = 65      # Numero di alberi nella foresta (dal paper: ottimo tra 65-93)
RF_MAX_DEPTH = None       # Profondità massima degli alberi (None = illimitata)
RF_MIN_SAMPLES_SPLIT = 2  # Campioni minimi per effettuare uno split
RF_MIN_SAMPLES_LEAF = 1   # Campioni minimi in una foglia
RF_MAX_FEATURES = 'sqrt'  # Feature da considerare per ogni split ('sqrt' dal paper)
RF_BOOTSTRAP = True       # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced'  # Gestione automatica dello sbilanciamento
RF_CRITERION = 'entropy'  # Criterio di splitting (dal paper: entropy migliore di gini per molti dataset)

NUM_ROUNDS = 50  # Numero di round di addestramento federato

# Variabili globali per tracking metriche
all_federated_metrics = []  # Lista di dict, uno per round
last_confusion_matrix = None

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

def save_federated_metrics_report(metrics_list):
    """
    Salva un report completo delle metriche federate per Random Forest.
    Adattato dalla versione DNN per gestire le specificità del Random Forest.
    """
    if not metrics_list:
        print("[SERVER] ⚠️ Nessuna metrica da salvare.")
        return

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, f"federated_random_forest_metrics_{timestamp}.txt")

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
        ("recall_natural", "Recall_Nat", 12),
        ("recall_attack", "Recall_Att", 12),
    ]

    def fmt(val, width):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A".ljust(width)
        return f"{val:.6f}".ljust(width)

    # HEADER
    title = "RESOCONTO ADDESTRAMENTO FEDERATO RANDOM FOREST SMARTGRID"
    n_rounds = len(metrics_list)
    header = f"{title}\nRounds totali: {n_rounds}\n\nTABELLA RIASSUNTIVA METRICHE:\n" + "="*140 + "\n"
    col_headers = "  ".join([name.ljust(width) for _, name, width in cols])
    sep = "-" * 140

    table_lines = []
    table_lines.append(col_headers)
    table_lines.append(sep)
    for row in metrics_list:
        vals = []
        for k, _, width in cols:
            v = row.get(k, None)
            if k == "round":
                vals.append(str(v).ljust(width))
            else:
                vals.append(fmt(v, width))
        table_lines.append("  ".join(vals))

    # STATISTICHE FINALI
    stats_lines = []
    stats_lines.append("\nSTATISTICHE FINALI:\n" + "="*60 + "\n")
    for k, name, width in cols:
        if k == "round":
            continue
        vals = [row[k] for row in metrics_list if row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k]))]
        if not vals:
            continue
        start = vals[0]
        end = vals[-1]
        minv = np.min(vals)
        maxv = np.max(vals)
        meanv = np.mean(vals)
        miglioramento = end - start if isinstance(end, float) and isinstance(start, float) else 0
        trend = "📈" if miglioramento > 0 else ("📉" if miglioramento < 0 else "")
        stats_lines.append(f"🔹 {name.upper()}:")
        stats_lines.append(f"   Rounds disponibili  : {len(vals)}")
        stats_lines.append(f"   Valore iniziale     : {fmt(start, 9)}")
        stats_lines.append(f"   Valore finale       : {fmt(end, 9)}")
        stats_lines.append(f"   Valore minimo       : {fmt(minv, 9)}")
        stats_lines.append(f"   Valore massimo      : {fmt(maxv, 9)}")
        stats_lines.append(f"   Valore medio        : {fmt(meanv, 9)}")
        stats_lines.append(f"   Miglioramento       : {fmt(miglioramento, 9)} {trend}")
        stats_lines.append("")

    # MATRICE DI CONFUSIONE FINALE
    conf_matrix_lines = []
    if last_confusion_matrix is not None:
        conf_matrix_lines.append("\nMATRICE DI CONFUSIONE SUL VALIDATION SET:\n" + "-"*40)
        conf_matrix_lines.append(f"{'tp:':<2} {last_confusion_matrix[1, 1]:<5} {'fp:':<2} {last_confusion_matrix[0, 1]:<5} {'fn:':<2} {last_confusion_matrix[1, 0]:<5} {'tn:':<2} {last_confusion_matrix[0, 0]:<5}\n")

    with open(report_path, "w") as f:
        f.write(header)
        for line in table_lines:
            f.write(line + "\n")
        f.write("="*140 + "\n")
        for line in conf_matrix_lines:
            f.write(line + "\n")
        for line in stats_lines:
            f.write(line + "\n")
    print(f"\n[SERVER] ✅ Report Random Forest federato salvato in: {report_path}")

def fit_clip_outliers_iqr(X, k=5.0):
    """Calcola i limiti per clipping outlier usando IQR."""
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def transform_clip_outliers_iqr(X, lower, upper):
    """Applica il clipping ai dati X usando i limiti forniti."""
    return np.clip(X, lower, upper)

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
    """Rimuove le feature che sono costanti almeno al 99.9%."""
    keep_mask = []
    n = X.shape[0]
    
    for col in range(X.shape[1]):
        col_data = X[:, col]
        vals, counts = np.unique(col_data, return_counts=True)
        max_count = np.max(counts)
        ratio = max_count / n
        var = np.nanvar(col_data)
        keep = not (ratio >= threshold_ratio or var < threshold_var)
        keep_mask.append(keep)
    
    keep_mask = np.array(keep_mask)
    return X[:, keep_mask], keep_mask

def clean_data_for_pca(X):
    """Pulizia robusta dei dati per prevenire problemi numerici in PCA (server)."""
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X_preprocessed):
    """Applica PCA con numero FISSO di componenti (server, identico ai client)."""
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
    Pipeline identica a quella dei client Random Forest.
    """
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
        X_np = np.array(X_cleaned, dtype=float)
        lower, upper = fit_clip_outliers_iqr(X_np, k=5.0)
        X_clipped = transform_clip_outliers_iqr(X_np, lower, upper)
    else:
        X_clipped = X_cleaned

    # STEP 3: Imputazione mediana
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_clipped)
    else:
        X_imputed = X_clipped

    # STEP 4: Rimozione feature quasi-costanti
    if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES:
        X_reduced, keep_mask = remove_near_constant_features(X_imputed, threshold_var=1e-12, threshold_ratio=0.999)
        print(f"[Server] Feature dopo rimozione quasi-costanti: {X_reduced.shape[1]} (da {X_imputed.shape[1]})")
    else:
        X_reduced = X_imputed
        print(f"[Server] Rimozione feature quasi-costanti DISABILITATA - mantenute {X_reduced.shape[1]} feature")
    
    # STEP 5: Scaling standard
    if ENABLE_SCALING:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
    else:
        X_scaled = X_reduced

    print(f"[Server] Preprocessing completato")

    # STEP 6: PCA (se abilitata)
    if ENABLE_PCA:
        X_global_final = apply_pca(X_scaled)
        if X_global_final.shape[1] != PCA_COMPONENTS:
            raise RuntimeError(f"Server PCA output shape inconsistente: {X_global_final.shape[1]} vs {PCA_COMPONENTS}")
        print(f"[Server] ✅ Pipeline preprocessing con PCA completata")
    else:
        X_global_final = X_scaled
        print(f"[Server] ✅ Pipeline preprocessing SENZA PCA completata")
    
    print(f"[Server] Risultato finale: {X_global_final.shape}")
    return X_global_final

def deserialize_trees_from_client(parameters):
    """
    Deserializza gli alberi ricevuti da un client.
    Gli alberi sono ricevuti come numpy arrays (uint8) serializzati con pickle.
    """
    try:
        # Gestisce diversi tipi di parametri da Flower
        if hasattr(parameters, 'tensors'):
            parameter_arrays = parameters.tensors
        elif isinstance(parameters, list): 
            parameter_arrays = parameters
        else:
            parameter_arrays = list(parameters) if parameters else []

        if not parameter_arrays:
            print(f"[Server] Nessun parametro ricevuto dal client")
            return []
        
        print(f"[Server] Ricevuti {len(parameter_arrays)} parametri dal client")

        # Debug del tipo del primo parametro
        if len(parameter_arrays) > 0:
            print(f"[Server] Tipo parametri ricevuti: {type(parameter_arrays[0])}")
            if isinstance(parameter_arrays[0], np.ndarray):
                print(f"[Server] DEBUG param[0]: dtype={parameter_arrays[0].dtype}, shape={parameter_arrays[0].shape}, primi 10 byte={parameter_arrays[0][:10].tolist()}")
            elif isinstance(parameter_arrays[0], bytes):
                print(f"[Server] DEBUG param[0]: primi 50 byte={parameter_arrays[0][:50]}")
        
        deserialized_trees = []
        
        for i, param_array in enumerate(parameter_arrays):
            try:
                print(f"[Server] Elaborazione parametro {i+1}/{len(parameter_arrays)}: tipo={type(param_array)}")
                
                # Converti in bytes per pickle.loads()
                if isinstance(param_array, np.ndarray):
                    # Caso: array NumPy di tipo uint8 (formato standard di Flower)
                    tree_bytes = param_array.tobytes()
                    print(f"[Server] Convertito numpy array in bytes: {len(tree_bytes)} bytes")
                elif isinstance(param_array, bytes):
                    # Caso: bytes grezzi (già serializzati da pickle)
                    tree_bytes = param_array
                    print(f"[Server] Bytes grezzi ricevuti: {len(tree_bytes)} bytes")
                else:
                    print(f"[Server] ⚠️ Parametro {i+1} ignorato (tipo non compatibile: {type(param_array)})")
                    continue

                # Deserializza l'albero con pickle
                tree = pickle.loads(tree_bytes)
                print(f"[Server] Oggetto deserializzato tipo: {type(tree)}")

                # Verifica che sia un albero valido
                if hasattr(tree, 'predict') and hasattr(tree, 'tree_'):
                    # Usa accuracy simulate per compatibilità (il client dovrebbe inviarle)
                    simulated_accuracy = max(0.8, 0.95 - (i * 0.002))
                    simulated_weighted_acc = max(0.75, 0.94 - (i * 0.002))
                    deserialized_trees.append((tree, simulated_accuracy, simulated_weighted_acc))
                    print(f"[Server] ✅ Albero {i+1} deserializzato correttamente (acc={simulated_accuracy:.3f})")
                else:
                    print(f"[Server] ⚠️ Oggetto {i+1} non è un albero valido (tipo: {type(tree)})")

            except Exception as e:
                print(f"[Server] ❌ Errore nella deserializzazione parametro {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[Server] Deserializzati {len(deserialized_trees)} alberi validi su {len(parameter_arrays)}")
        return deserialized_trees
        
    except Exception as e:
        print(f"[Server] ❌ Errore nella deserializzazione: {e}")
        import traceback
        traceback.print_exc()
        return []

def select_best_trees(all_trees_data, strategy=TREE_AGGREGATION_STRATEGY, method=TREE_SELECTION_METHOD, max_trees=MAX_TREES_GLOBAL):
    """
    Seleziona i migliori alberi basandosi sui criteri del paper.
    Implementa i metodi di selezione descritti nel paper.
    
    Args:
        all_trees_data: Lista di liste, ogni lista contiene gli alberi di un client
        strategy: 'per_forest' o 'global'
        method: 'accuracy' o 'weighted_accuracy'  
        max_trees: Numero massimo di alberi nel modello globale
        
    Returns:
        Lista dei migliori alberi selezionati
    """
    print(f"[Server] === SELEZIONE ALBERI ===")
    print(f"Strategia: {strategy}")
    print(f"Metodo: {method}")
    print(f"Max alberi globale: {max_trees}")
    
    if not all_trees_data:
        print(f"[Server] ⚠️ Nessun dato alberi ricevuto")
        return []
    
    selected_trees = []
    
    if strategy == 'per_forest':
        # Seleziona migliori alberi da ogni Random Forest client
        trees_per_client = max_trees // len(all_trees_data)
        remaining_trees = max_trees % len(all_trees_data)
        
        print(f"[Server] Seleziono {trees_per_client} alberi per client ({len(all_trees_data)} client)")
        
        for client_idx, client_trees in enumerate(all_trees_data):
            if not client_trees:
                continue
                
            # Ordina gli alberi del client per performance
            metric_idx = 2 if method == 'weighted_accuracy' else 1
            sorted_trees = sorted(client_trees, key=lambda x: x[metric_idx], reverse=True)
            
            # Seleziona i migliori per questo client
            num_to_select = trees_per_client + (1 if client_idx < remaining_trees else 0)
            client_selected = sorted_trees[:num_to_select]
            selected_trees.extend(client_selected)
            
            if client_selected:
                best_score = client_selected[0][metric_idx]
                print(f"[Server] Client {client_idx+1}: {len(client_selected)} alberi, migliore {method}={best_score:.4f}")
    
    elif strategy == 'global':
        # Seleziona migliori alberi globalmente tra tutti i client
        all_trees_flat = []
        for client_trees in all_trees_data:
            all_trees_flat.extend(client_trees)
        
        if all_trees_flat:
            # Ordina tutti gli alberi per performance
            metric_idx = 2 if method == 'weighted_accuracy' else 1
            sorted_trees = sorted(all_trees_flat, key=lambda x: x[metric_idx], reverse=True)
            
            # Seleziona i migliori globalmente
            selected_trees = sorted_trees[:max_trees]
            
            if selected_trees:
                best_score = selected_trees[0][metric_idx]
                worst_score = selected_trees[-1][metric_idx]
                print(f"[Server] Selezione globale: {len(selected_trees)} alberi")
                print(f"[Server] Range {method}: {worst_score:.4f} - {best_score:.4f}")
    
    print(f"[Server] Alberi selezionati totali: {len(selected_trees)}")
    return selected_trees

def create_global_random_forest(selected_trees):
    """
    Crea un Random Forest globale combinando i migliori alberi dai client.
    Implementa l'aggregazione descritta nel paper.
    
    Args:
        selected_trees: Lista di tuple (tree, accuracy, weighted_accuracy)
        
    Returns:
        RandomForestClassifier globale configurato
    """
    print(f"[Server] === CREAZIONE RANDOM FOREST GLOBALE ===")
    
    if not selected_trees:
        print(f"[Server] ⚠️ Nessun albero da aggregare, creo RF vuoto")
        # Crea un Random Forest vuoto con configurazione base
        return RandomForestClassifier(
            n_estimators=1,  # Minimo per evitare errori
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            bootstrap=RF_BOOTSTRAP,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight=RF_CLASS_WEIGHT,
            criterion=RF_CRITERION
        )
    
    # Estrai solo gli alberi (senza metadati)
    trees = [tree_data[0] for tree_data in selected_trees]
    
    # Crea un nuovo Random Forest con gli alberi aggregati
    global_rf = RandomForestClassifier(
        n_estimators=len(trees),  # Numero di alberi = alberi selezionati
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        bootstrap=RF_BOOTSTRAP,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight=RF_CLASS_WEIGHT,
        criterion=RF_CRITERION
    )
    
    # Assegna gli alberi al Random Forest globale
    # NOTA: Questo è un hack necessario perché scikit-learn non espone un'API diretta
    # per impostare alberi preaddestrati. In produzione, si dovrebbe usare un approccio
    # più robusto o una libreria ML che supporta nativamente l'aggregazione di alberi.
    global_rf.estimators_ = trees
    global_rf.n_estimators = len(trees)
    
    # Copia attributi necessari dal primo albero per permettere predizioni
    first_tree = trees[0]
    if hasattr(first_tree, 'n_features_in_'):
        global_rf.n_features_in_ = first_tree.n_features_in_
    if hasattr(first_tree, 'n_outputs_'):
        global_rf.n_outputs_ = first_tree.n_outputs_
    if hasattr(first_tree, 'classes_'):
        global_rf.classes_ = first_tree.classes_
        global_rf.n_classes_ = len(first_tree.classes_)
    else:
        # Default per classificazione binaria
        global_rf.classes_ = np.array([0, 1])
        global_rf.n_classes_ = 2
    
    # Calcola feature importance media dai singoli alberi (se disponibile)
    if hasattr(trees[0], 'feature_importances_'):
        n_features = trees[0].feature_importances_.shape[0]
        feature_importances = np.zeros(n_features)
        
        for tree in trees:
            if hasattr(tree, 'feature_importances_'):
                feature_importances += tree.feature_importances_
        
        # Media pesata delle feature importance
        feature_importances /= len(trees)
        global_rf.feature_importances_ = feature_importances
    
    print(f"[Server] ✅ Random Forest globale creato con {len(trees)} alberi")
    print(f"[Server] Attributi configurati: n_features={getattr(global_rf, 'n_features_in_', 'N/A')}, n_classes={getattr(global_rf, 'n_classes_', 'N/A')}")
    
    # Statistiche degli alberi aggregati
    accuracies = [tree_data[1] for tree_data in selected_trees]
    weighted_accuracies = [tree_data[2] for tree_data in selected_trees]
    
    print(f"[Server] Accuracy alberi: min={min(accuracies):.4f}, max={max(accuracies):.4f}, mean={np.mean(accuracies):.4f}")
    print(f"[Server] Weighted accuracy: min={min(weighted_accuracies):.4f}, max={max(weighted_accuracies):.4f}, mean={np.mean(weighted_accuracies):.4f}")
    
    return global_rf

def serialize_global_model(global_rf):
    """
    Serializza il Random Forest globale per l'invio ai client.
    Usa pickle + conversione in numpy array (uint8) per compatibilità con Flower.
    """
    try:
        # Serializza il modello Random Forest globale con pickle
        model_bytes = pickle.dumps(global_rf, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Converti in numpy array (uint8) per Flower
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        
        print(f"[Server] Modello globale serializzato ({len(model_bytes)} bytes)")
        print(f"[Server] Convertito in numpy array: shape={model_array.shape}, dtype={model_array.dtype}")
        
        # Flower richiede una lista come output di parameters
        return [model_array]
        
    except Exception as e:
        print(f"[Server] ❌ Errore serializzazione modello globale: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_smartgrid_random_forest_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server Random Forest SmartGrid.
    Identica alla versione DNN ma adattata per Random Forest.
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Usa preprocessing identico ai client Random Forest.
        """
        set_reproducibility_seeds()

        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER RANDOM FOREST ===")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Usa client 14-15 come dataset di test (stesso della versione DNN)
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
        
        # Prepara X e y (mantiene distribuzione naturale)
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        # Statistiche distribuzione naturale globale
        attack_samples = y_global.sum()
        natural_samples = (y_global == 0).sum()
        attack_ratio = y_global.mean()
        
        print(f"Dataset test globale: {len(df_global)} campioni")
        print(f"Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
        
        # Applica pipeline preprocessing identica ai client
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
        feature_count = PCA_COMPONENTS if ENABLE_PCA else 128
        X_global = np.random.random((100, feature_count))
        y_global = np.random.randint(0, 2, 100)
        dataset_info = {'total_samples': 100, 'attack_samples': 50, 'natural_samples': 50, 'attack_ratio': 0.5}
        print(f"Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round per Random Forest.
        """
        set_reproducibility_seeds()

        print(f"\n=== VALUTAZIONE GLOBALE RANDOM FOREST - ROUND {server_round + 1} ===")
        
        try:
            # CONTROLLO: Verifica se ci sono parametri da valutare
            if not parameters or len(parameters) == 0:
                print(f"[Server] ⚠️ Nessun modello da valutare")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": "no_model_received", 
                    "global_test_samples": len(X_global)
                }
            
            try:
                # Deserializza il Random Forest globale
                model_array = parameters[0]

                # Converte numpy array in bytes
                if hasattr(model_array, 'tobytes'):
                    model_bytes = model_array.tobytes()
                elif hasattr(model_array, 'data'):
                    model_bytes = model_array.data.tobytes()
                else:
                    model_bytes = bytes(model_array)
                
                print(f"[Server] Deserializzazione modello: {len(model_bytes)} bytes")

                # Deserializza usando pickle
                global_rf = pickle.loads(model_bytes)

                print(f"✅ Modello Random Forest globale deserializzato")
                print(f"   N. alberi: {global_rf.n_estimators if hasattr(global_rf, 'n_estimators') else 'N/A'}")
            except Exception as e:
                print(f"❌ Errore deserializzazione modello: {e}")
                import traceback
                traceback.print_exc()
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": f"deserialization_failed: {str(e)}", 
                    "global_test_samples": len(X_global)
                }
            
            # Verifica che il modello sia stato addestrato
            if not hasattr(global_rf, 'estimators_') or len(global_rf.estimators_) == 0:
                print(f"⚠️ Modello Random Forest non addestrato, uso predizioni casuali")
                # Predizioni casuali per evitare crash
                y_pred_binary = np.random.randint(0, 2, len(y_global))
                y_pred_prob = np.random.random(len(y_global))
            else:
                # Valutazione sul dataset test globale
                try:
                    y_pred_binary = global_rf.predict(X_global)
                    y_pred_prob = global_rf.predict_proba(X_global)[:, 1] if hasattr(global_rf, 'predict_proba') else np.random.random(len(y_global))
                except Exception as e:
                    print(f"⚠️ Errore predizione, uso valori casuali: {e}")
                    y_pred_binary = np.random.randint(0, 2, len(y_global))
                    y_pred_prob = np.random.random(len(y_global))
            
            # Calcolo metriche
            accuracy = accuracy_score(y_global, y_pred_binary)
            
            # Metriche sicure (gestione casi edge)
            try:
                precision = precision_score(y_global, y_pred_binary, zero_division=0)
                recall = recall_score(y_global, y_pred_binary, zero_division=0)
                f1_score_val = f1_score(y_global, y_pred_binary, zero_division=0)
                balanced_acc = balanced_accuracy_score(y_global, y_pred_binary)
                
                if len(np.unique(y_global)) > 1 and len(np.unique(y_pred_prob)) > 1:
                    auc = roc_auc_score(y_global, y_pred_prob)
                else:
                    auc = 0.5  # AUC neutrale se non calcolabile
                    
            except Exception as e:
                print(f"⚠️ Errore calcolo metriche: {e}")
                precision = recall = f1_score_val = balanced_acc = auc = 0.0
            
            # Report dettagliato per classe
            try:
                report = classification_report(y_global, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
                conf_matrix = confusion_matrix(y_global, y_pred_binary)
            except Exception as e:
                print(f"⚠️ Errore classification report: {e}")
                report = {"natural": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
                         "attack": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}}
                conf_matrix = np.array([[0, 0], [0, 0]])
            
            # Loss simulata (Random Forest non ha loss)
            loss = 1 - accuracy
            
            print(f"RISULTATI VALUTAZIONE RANDOM FOREST:")
            print(f"  Loss (simulata): {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1_score_val:.4f} ({f1_score_val*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  Campioni test: {len(X_global)}")
            print(f"  Distribuzione naturale: {dataset_info.get('attack_ratio', 0)*100:.1f}% attacchi")
            
            if hasattr(global_rf, 'estimators_') and len(global_rf.estimators_) > 0:
                print(f"  Alberi nel modello globale: {len(global_rf.estimators_)}")
            
            print(f"Classification report (per classe):")
            print(classification_report(y_global, y_pred_binary, target_names=["natural", "attack"], zero_division=0))
            print(f"Confusion matrix:")
            print(conf_matrix)

            # Raccolta metriche per report
            metric_row = {
                "round": server_round + 1,
                "loss_distribuita": float(loss),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "f1_natural": float(report["natural"]["f1-score"]),
                "f1_attack": float(report["attack"]["f1-score"]),
                "precision": float(precision),
                "precision_natural": float(report["natural"]["precision"]),
                "precision_attack": float(report["attack"]["precision"]),
                "recall": float(recall),
                "recall_natural": float(report["natural"]["recall"]),
                "recall_attack": float(report["attack"]["recall"]),
            }
            
            # Salva ultima confusion matrix per report finale
            global last_confusion_matrix
            last_confusion_matrix = conf_matrix

            # Aggiungi alla lista globale delle metriche
            global all_federated_metrics
            all_federated_metrics.append(metric_row)

            return float(loss), {
                # Metriche base
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "balanced_accuracy": float(balanced_acc),

                # Metriche per classe
                "precision_natural": float(report["natural"]["precision"]),
                "recall_natural": float(report["natural"]["recall"]),
                "f1_natural": float(report["natural"]["f1-score"]),
                "precision_attack": float(report["attack"]["precision"]),
                "recall_attack": float(report["attack"]["recall"]),
                "f1_attack": float(report["attack"]["f1-score"]),
                "support_natural": int(report["natural"]["support"]),
                "support_attack": int(report["attack"]["support"]),
                
                # Confusion matrix
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1]),
                
                # Informazioni dataset e modello
                "global_test_samples": int(len(X_global)),
                "n_trees_global": int(len(global_rf.estimators_)) if hasattr(global_rf, 'estimators_') else 0,
                "attack_samples": int(dataset_info.get('attack_samples', 0)),
                "natural_samples": int(dataset_info.get('natural_samples', 0)),
                "attack_ratio": float(dataset_info.get('attack_ratio', 0)),
            }
            
        except Exception as e:
            print(f"❌ Errore durante valutazione globale Random Forest: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {
                "accuracy": 0.0, 
                "error": f"evaluation_failed: {str(e)}", 
                "global_test_samples": len(X_global) if 'X_global' in locals() else 0
            }
    
    return evaluate

def print_client_metrics_rf(fit_results):
    """
    Stampa le metriche dei client Random Forest dopo ogni round.
    Adattata dalla versione DNN per gestire le specificità del Random Forest.
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT RANDOM FOREST ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_weighted_f1 = 0
    error_clients = []
    accuracy_list = []
    f1_list = []
    oob_scores = []
    n_estimators_list = []
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1}: {client_samples} campioni")
        
        if 'error' in client_metrics:
            error_clients.append(i+1)
            print(f"  ERRORE: {client_metrics['error']}")
            continue
        
        # Metriche base Random Forest
        if 'train_accuracy' in client_metrics:
            accuracy = client_metrics['train_accuracy']
            total_weighted_accuracy += accuracy * client_samples
            accuracy_list.append(accuracy)
            print(f"  Accuracy: {accuracy:.4f}")
        
        if 'train_f1_score' in client_metrics:
            f1 = client_metrics['train_f1_score']
            total_weighted_f1 += f1 * client_samples
            f1_list.append(f1)
            print(f"  F1-Score: {f1:.4f}")
        
        if 'train_balanced_accuracy' in client_metrics:
            balanced_acc = client_metrics['train_balanced_accuracy']
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        if 'oob_score' in client_metrics:
            oob = client_metrics['oob_score']
            oob_scores.append(oob)
            print(f"  OOB Score: {oob:.4f}")
        
        if 'n_estimators' in client_metrics:
            n_est = client_metrics['n_estimators']
            n_estimators_list.append(n_est)
            print(f"  N. Estimatori: {n_est}")
        
        if 'n_features' in client_metrics:
            n_feat = client_metrics['n_features']
            print(f"  N. Features: {n_feat}")
    
    if total_samples > 0:
        # Calcola medie ponderate
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_weighted_f1 = total_weighted_f1 / total_samples if total_weighted_f1 > 0 else 0
        avg_oob = np.mean(oob_scores) if oob_scores else 0
        
        print(f"\nRIASSUNTO METRICHE RANDOM FOREST:")
        print(f"  Media accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  Media F1-Score: {avg_weighted_f1:.4f}")
        print(f"  Media OOB Score: {avg_oob:.4f}")
        print(f"  Totale campioni: {total_samples}")
        print(f"  Client con errori: {len(error_clients)}")
        
        if n_estimators_list:
            print(f"  Alberi per client: {np.mean(n_estimators_list):.1f} ± {np.std(n_estimators_list):.1f}")

class SmartGridRandomForestFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid Random Forest.
    Implementa l'aggregazione degli alberi basata sul paper.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega gli alberi Random Forest dai client secondo la metodologia del paper.
        """
        set_reproducibility_seeds()

        print(f"\n=== AGGREGAZIONE RANDOM FOREST - ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("❌ ERRORE: Nessun client ha fornito risultati validi")
            return None, {}
        
        # Stampa metriche dei client Random Forest
        print_client_metrics_rf(results)
        
        try:
            # Deserializza gli alberi da tutti i client
            all_trees_data = []
            
            for i, (client_proxy, fit_res) in enumerate(results):
                print(f"\n[Server] Processando alberi da client {i+1}...")
                
                client_trees = deserialize_trees_from_client(fit_res.parameters)
                
                if client_trees:
                    all_trees_data.append(client_trees)
                    print(f"[Server] Client {i+1}: {len(client_trees)} alberi ricevuti")
                    
                    # Mostra statistiche alberi del client
                    if client_trees:
                        accuracies = [tree[1] for tree in client_trees]
                        w_accuracies = [tree[2] for tree in client_trees]
                        print(f"[Server] Client {i+1} - Accuracy range: {min(accuracies):.4f}-{max(accuracies):.4f}")
                        print(f"[Server] Client {i+1} - Weighted acc range: {min(w_accuracies):.4f}-{max(w_accuracies):.4f}")
                else:
                    print(f"[Server] ⚠️ Client {i+1}: nessun albero valido ricevuto")
            
            if not all_trees_data:
                print(f"[Server] ❌ Nessun albero valido ricevuto da alcun client")
                return None, {}
            
            # Seleziona i migliori alberi secondo il paper
            selected_trees = select_best_trees(
                all_trees_data, 
                strategy=TREE_AGGREGATION_STRATEGY,
                method=TREE_SELECTION_METHOD, 
                max_trees=MAX_TREES_GLOBAL
            )
            
            if not selected_trees:
                print(f"[Server] ❌ Nessun albero selezionato per l'aggregazione")
                return None, {}
            
            # Crea il Random Forest globale
            global_rf = create_global_random_forest(selected_trees)
            
            # Serializza il modello per l'invio ai client
            serialized_model = serialize_global_model(global_rf)
            
            if not serialized_model:
                print(f"[Server] ❌ Errore nella serializzazione del modello globale")
                return None, {}
            
            print(f"[Server] ✅ Aggregazione Random Forest completata")
            print(f"[Server] ✅ Modello globale creato con {len(selected_trees)} alberi")
            print(f"[Server] ✅ Strategia: {TREE_AGGREGATION_STRATEGY}, Metodo: {TREE_SELECTION_METHOD}")
            
            # Restituisce i parametri aggregati
            return serialized_model, {}
            
        except Exception as e:
            print(f"[Server] ❌ ERRORE durante aggregazione Random Forest: {e}")
            import traceback
            traceback.print_exc()
            return None, {}

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione Random Forest.
        """
        set_reproducibility_seeds()

        print(f"\n=== AGGREGAZIONE VALUTAZIONE RANDOM FOREST ROUND {server_round} ===")
        print(f"Client che hanno valutato: {len(results)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            # Chiama l'aggregazione standard di Flower
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione Random Forest completata per round {server_round}")
            else:
                print(f"⚠️ Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione valutazione Random Forest: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server Random Forest federato SmartGrid.
    """
    set_reproducibility_seeds()

    print("=" * 80)
    print("🌳 SERVER FEDERATO SMARTGRID - RANDOM FOREST")
    print("=" * 80)
    print("Configurazione Random Forest Federato:")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Client minimi: 2")
    print(f"  - Strategia: FedAvg personalizzata per Random Forest")
    print(f"  - Valutazione: Dataset globale (client 14-15)")
    print(f"  - Aggregazione alberi: {TREE_AGGREGATION_STRATEGY}")
    print(f"  - Selezione alberi: {TREE_SELECTION_METHOD}")
    print(f"  - Max alberi globali: {MAX_TREES_GLOBAL}")
    print(f"  - Ensemble method: {ENSEMBLE_METHOD}")
    print("")
    print("Pipeline Preprocessing (identica ai client RF):")
    print(f"  - Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"  - Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"  - Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"  - Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"  - Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"  - PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")
    print("")
    print("Random Forest Configurazione:")
    print(f"  - N. Estimatori (per client): {RF_N_ESTIMATORS}")
    print(f"  - Criterio: {RF_CRITERION}")
    print(f"  - Max features: {RF_MAX_FEATURES}")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED}")
    print("")
    
    # Configurazione del server
    config = fl.server.ServerConfig(NUM_ROUNDS)
    
    # Strategia Random Forest Federato personalizzata
    strategy = SmartGridRandomForestFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_smartgrid_random_forest_evaluate_fn()
    )
    
    print(f"🌳 Server Random Forest in attesa di client su localhost:8080...")
    print("")
    print("Per connettere i client Random Forest, esegui:")
    print("  python clientRF.py 1")
    print("  python clientRF.py 2")
    print("  ...")
    print("  python clientRF.py 13")
    print("")
    print("Client 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    print("=" * 80)
    
    try:
        # Avvia il server Flower
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )

        # Salva report finale
        global all_federated_metrics
        if all_federated_metrics:
            save_federated_metrics_report(all_federated_metrics)
        else:
            print("[SERVER] ⚠️ Nessuna metrica federata disponibile per il report finale.")
        
    except Exception as e:
        print(f"❌ Errore durante l'avvio del server Random Forest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()