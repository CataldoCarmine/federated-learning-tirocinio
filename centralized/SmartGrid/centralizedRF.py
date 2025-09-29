import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ========== CONFIGURAZIONE RIPRODUCIBILITÀ ==========
RANDOM_SEED = 42

# ========== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==========
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = True       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = True                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = True  # Cambia a False per disabilitare rimozione feature quasi-costanti
ENABLE_PCA = True  # Cambia a False per disabilitare la PCA

if ENABLE_PCA:
    ENABLE_IMPUTATION= True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 71  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 42  # Seme specifico per PCA

# ========== CONFIGURAZIONE MODELLO RANDOM FOREST ==========
RF_N_ESTIMATORS = 100          # Numero di alberi nella foresta
RF_MAX_DEPTH = None            # Profondità massima degli alberi (None = illimitata)
RF_MIN_SAMPLES_SPLIT = 2       # Campioni minimi per effettuare uno split
RF_MIN_SAMPLES_LEAF = 1        # Campioni minimi in una foglia
RF_MAX_FEATURES = 'sqrt'       # Feature da considerare per ogni split
RF_BOOTSTRAP = True            # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced'   # Gestione automatica dello sbilanciamento

# ========== FUNZIONI DI PREPROCESSING (identiche al federato) ==========
def set_reproducibility_seeds():
    """Imposta tutti i semi per garantire riproducibilità."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    import random
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()

def fit_clip_outliers_iqr(X, k=5.0):
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def transform_clip_outliers_iqr(X, lower, upper):
    return np.clip(X, lower, upper)

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
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
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X, pca_obj=None):
    """Applica PCA con numero FISSO di componenti."""
    if pca_obj is None:
        n_components = min(PCA_COMPONENTS, X.shape[1], len(X))
        pca = PCA(n_components=n_components, random_state=PCA_RANDOM_SEED)
        X_pca = pca.fit_transform(X)
        return X_pca, pca
    else:
        X_pca = pca_obj.transform(X)
        return X_pca

def compute_class_weights(y):
    """Calcola i pesi delle classi per compensare lo sbilanciamento."""
    try:
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        return class_weight_dict
    except Exception as e:
        unique_classes = np.unique(y)
        return {cls: 1.0 for cls in unique_classes}

def load_centralized_smartgrid_data():
    """Carica e unisce tutti i dati SmartGrid per l'addestramento centralizzato."""
    set_reproducibility_seeds()
    print("=== CARICAMENTO DATASET SMARTGRID CENTRALIZZATO ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data", "SmartGrid")
    df_list = []
    files_loaded = []
    for file_id in range(1, 16):
        file_path = os.path.join(data_dir, f"data{file_id}.csv")
        if os.path.exists(file_path):
            try:
                df_file = pd.read_csv(file_path)
                df_list.append(df_file)
                files_loaded.append(file_id)
                print(f"  - Caricato data{file_id}.csv: {len(df_file)} campioni")
            except Exception as e:
                print(f"  - Errore nel caricamento di data{file_id}.csv: {e}")
        else:
            print(f"  - File data{file_id}.csv non trovato")
    if not df_list:
        raise FileNotFoundError("Nessun file di dati SmartGrid trovato nella cartella data/SmartGrid/")
    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"\nDataset centralizzato combinato:")
    print(f"  - File caricati: {len(files_loaded)} ({files_loaded})")
    print(f"  - Totale campioni: {len(df_combined)}")
    print(f"  - Feature totali: {df_combined.shape[1] - 1}")  # -1 per escludere 'marker'
    X = df_combined.drop(columns=["marker"])
    y = (df_combined["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    marker_distribution = df_combined["marker"].value_counts()
    print(f"\nDistribuzione per tipo di scenario:")
    for marker, count in marker_distribution.items():
        percentage = (count / len(df_combined)) * 100
        print(f"  - {marker}: {count} campioni ({percentage:.2f}%)")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset_info = {
        'files_loaded': files_loaded,
        'total_files': len(files_loaded),
        'total_samples': len(df_combined),
        'features': X.shape[1],
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio
    }
    print("=" * 60)
    return X, y, dataset_info

def split_train_validation_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    print(f"=== STEP 1: SUDDIVISIONE TRAIN/VALIDATION/TEST (PRIMA DEL PREPROCESSING) ===")
    total_size = train_size + val_size + test_size
    if abs(total_size - 1.0) > 0.001:
        raise ValueError(f"Le proporzioni devono sommare a 1.0, ricevuto: {total_size}")
    temp_val_test_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_val_test_size, random_state=random_state, stratify=y
    )
    relative_test_size = test_size / temp_val_test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )
    print(f"  - Training set: {len(X_train)} campioni ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Validation set: {len(X_val)} campioni ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  - Test set: {len(X_test)} campioni ({len(X_test)/len(X)*100:.1f}%)")
    train_attack_ratio = y_train.mean()
    val_attack_ratio = y_val.mean()
    test_attack_ratio = y_test.mean()
    print(f"  - Proporzione attacchi training: {train_attack_ratio*100:.2f}%")
    print(f"  - Proporzione attacchi validation: {val_attack_ratio*100:.2f}%")
    print(f"  - Proporzione attacchi test: {test_attack_ratio*100:.2f}%")
    print("=" * 60)
    return X_train, X_val, X_test, y_train, y_val, y_test

def centralized_preprocessing(X_train_raw, X_val_raw, X_test_raw):
    """Pipeline identica a quella federata, con flag dinamiche."""
    set_reproducibility_seeds()
    print(f"=== PREPROCESSING CENTRALIZZATO ===")
    print(f"Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")

    # STEP 1: Pulizia dei dati
    if ENABLE_CLEAN_INF_NAN:
        X_train_clean = clean_data_for_pca(X_train_raw)
        X_val_clean = clean_data_for_pca(X_val_raw)
        X_test_clean = clean_data_for_pca(X_test_raw)
    else:
        X_train_clean = X_train_raw.values if hasattr(X_train_raw, 'values') else X_train_raw
        X_val_clean = X_val_raw.values if hasattr(X_val_raw, 'values') else X_val_raw
        X_test_clean = X_test_raw.values if hasattr(X_test_raw, 'values') else X_test_raw

    # STEP 2: Clipping outlier per quantili
    if ENABLE_CLIPPING_OUTLIERS:
        lower, upper = fit_clip_outliers_iqr(X_train_clean, k=5.0)
        X_train_clipped = transform_clip_outliers_iqr(X_train_clean, lower, upper)
        X_val_clipped = transform_clip_outliers_iqr(X_val_clean, lower, upper)
        X_test_clipped = transform_clip_outliers_iqr(X_test_clean, lower, upper)
    else:
        X_train_clipped = X_train_clean
        X_val_clipped = X_val_clean
        X_test_clipped = X_test_clean

    # STEP 3: Imputazione dei valori mancanti
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_val_imputed = imputer.transform(X_val_clipped)
        X_test_imputed = imputer.transform(X_test_clipped)
    else:
        X_train_imputed = X_train_clipped
        X_val_imputed = X_val_clipped
        X_test_imputed = X_test_clipped

    # STEP 4: Rimozione delle feature quasi-costanti
    if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES:
        X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
        X_val_reduced = X_val_imputed[:, keep_mask]
        X_test_reduced = X_test_imputed[:, keep_mask]
        print(f"Feature dopo rimozione quasi-costanti: {X_train_reduced.shape[1]} (da {X_train_imputed.shape[1]})")
    else:
        X_train_reduced = X_train_imputed
        X_val_reduced = X_val_imputed
        X_test_reduced = X_test_imputed
        print(f"Rimozione feature quasi-costanti DISABILITATA - mantenute {X_train_reduced.shape[1]} feature")

    # STEP 5: Scaling standard 
    if ENABLE_SCALING:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reduced)
        X_val_scaled = scaler.transform(X_val_reduced)
        X_test_scaled = scaler.transform(X_test_reduced)
        print("Scaling standard applicato")
    else:
        X_train_scaled = X_train_reduced
        X_val_scaled = X_val_reduced
        X_test_scaled = X_test_reduced
        print("Scaling DISABILITATO")

    # STEP 6: PCA
    if ENABLE_PCA:
        X_train_final, pca_obj = apply_pca(X_train_scaled)
        X_val_final = apply_pca(X_val_scaled, pca_obj=pca_obj)
        X_test_final = apply_pca(X_test_scaled, pca_obj=pca_obj)
        expected_features = PCA_COMPONENTS
        if X_train_final.shape[1] != expected_features:
            raise RuntimeError(f"PCA output shape inconsistente: {X_train_final.shape} vs {expected_features}")
        print(f"PCA applicata: {X_train_final.shape[1]} componenti")
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        X_test_final = X_test_scaled
        print(f"PCA DISABILITATA - usando dati scalati direttamente: {X_train_final.shape}")

    return X_train_final, X_val_final, X_test_final

def create_smartgrid_random_forest_model(input_features):
    """
    Crea il modello Random Forest per SmartGrid.
    
    Args:
        input_features: Numero di feature in input (per coerenza con la funzione precedente)
        
    Returns:
        Modello Random Forest configurato
    """
    set_reproducibility_seeds()
    print(f"[Centralizzato] === CREAZIONE RANDOM FOREST ===")
    print(f"[Centralizzato] Input features: {input_features}")
    print(f"[Centralizzato] Architettura: Random Forest con {RF_N_ESTIMATORS} alberi")
    
    # Crea il modello Random Forest con i parametri configurati
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,           # Numero di alberi nella foresta
        max_depth=RF_MAX_DEPTH,                 # Profondità massima degli alberi
        min_samples_split=RF_MIN_SAMPLES_SPLIT, # Campioni minimi per split
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,   # Campioni minimi per foglia
        max_features=RF_MAX_FEATURES,           # Feature da considerare per ogni split
        bootstrap=RF_BOOTSTRAP,                 # Bootstrap sampling
        random_state=RANDOM_SEED,               # Per riproducibilità
        n_jobs=-1,                              # Usa tutti i core disponibili
        class_weight=RF_CLASS_WEIGHT            # Gestione automatica dello sbilanciamento
    )
    
    print(f"[Centralizzato] Parametri Random Forest:")
    print(f"  - N. estimatori: {RF_N_ESTIMATORS}")
    print(f"  - Max depth: {RF_MAX_DEPTH}")
    print(f"  - Min samples split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  - Min samples leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  - Max features: {RF_MAX_FEATURES}")
    print(f"  - Bootstrap: {RF_BOOTSTRAP}")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED}")
    
    return model

def train_smartgrid_random_forest_model(model, X_train, y_train, X_val, y_val):
    """
    Addestra il modello Random Forest.
    
    Args:
        model: Modello Random Forest da addestrare
        X_train: Dati di training
        y_train: Etichette di training
        X_val: Dati di validation
        y_val: Etichette di validation
        
    Returns:
        None (Random Forest non ha history come le reti neurali)
    """
    print("=== ADDESTRAMENTO RANDOM FOREST CENTRALIZZATO ===")
    
    print(f"Training su {len(X_train)} campioni")
    print(f"Validation su {len(X_val)} campioni")
    print(f"Distribuzione training - Attacchi: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"Distribuzione validation - Attacchi: {y_val.sum()}/{len(y_val)} ({y_val.mean()*100:.1f}%)")
    
    print("Inizio addestramento Random Forest...")
    model.fit(X_train, y_train)
    
    print("✅ Addestramento Random Forest completato")
    
    # Valutazione rapida sul training set per dare feedback
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"Accuracy training: {train_score:.4f}")
    print(f"Accuracy validation: {val_score:.4f}")
    
    # Random Forest non ha history come le reti neurali, restituiamo None
    return None

def evaluate_smartgrid_random_forest_model(model, X_test, y_test, set_name="Test", threshold=0.5):
    """
    Valuta il modello Random Forest.
    
    Args:
        model: Modello Random Forest addestrato
        X_test: Dati di test
        y_test: Etichette di test
        set_name: Nome del set per il logging
        threshold: Soglia per la classificazione (per compatibilità, non usata)
        
    Returns:
        Tuple con (loss_simulata, accuracy, metriche_dict)
    """
    print(f"=== VALUTAZIONE FINALE RANDOM FOREST SMARTGRID - {set_name.upper()} SET ===")
    
    # Random Forest produce direttamente predizioni binarie e probabilità
    y_pred_binary = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilità della classe positiva (attacco)
    
    # Calcolo delle metriche base
    accuracy = (y_pred_binary == y_test).mean()
    
    # Metriche che richiedono gestione dei casi edge
    if len(np.unique(y_test)) > 1:
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob)
    else:
        precision = 0.0
        recall = 0.0
        auc = 0.0
    
    # F1-score e Balanced Accuracy
    f1_score_val = f1_score(y_test, y_pred_binary, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_binary)
    
    # Report dettagliato per classe
    report = classification_report(
        y_test, y_pred_binary, 
        target_names=["natural", "attack"], 
        output_dict=True, 
        zero_division=0
    )
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    
    # Loss simulata (Random Forest non ha una loss specifica, usiamo 1 - accuracy)
    loss = 1 - accuracy
    
    # Stampa risultati
    print(f"  Loss (simulata): {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1_score_val:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    print(f"Classification report (per classe):")
    print(classification_report(y_test, y_pred_binary, target_names=["natural", "attack"], zero_division=0))
    print(f"Confusion matrix:")
    print(conf_matrix)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        top_features = np.argsort(feature_importances)[::-1][:10]
        print(f"\nTop 10 Feature Importance:")
        for i, idx in enumerate(top_features):
            print(f"  Feature {idx}: {feature_importances[idx]:.4f}")
    
    # Restituisce le metriche in formato compatibile con il codice esistente
    return loss, accuracy, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "f1_score": f1_score_val,
        "balanced_accuracy": balanced_acc,
        "precision_natural": report["natural"]["precision"],
        "recall_natural": report["natural"]["recall"],
        "f1_natural": report["natural"]["f1-score"],
        "precision_attack": report["attack"]["precision"],
        "recall_attack": report["attack"]["recall"],
        "f1_attack": report["attack"]["f1-score"],
        "support_natural": report["natural"]["support"],
        "support_attack": report["attack"]["support"],
        "tn": int(conf_matrix[0, 0]),
        "fp": int(conf_matrix[0, 1]),
        "fn": int(conf_matrix[1, 0]),
        "tp": int(conf_matrix[1, 1])
    }

# ========== FUNZIONE FEATURE IMPORTANCE (modificata per restituire i valori) ==========

def feature_importance_analysis(X, y, feature_names=None, n_estimators=100, title="Feature Importance", max_show=20):
    """
    Calcola la feature importance con RandomForestClassifier.
    Restituisce una lista di tuple (feature_name, importance).
    """
    print(f"=== ANALISI FEATURE IMPORTANCE (RandomForest) ===")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    results = []
    print(f"{'Feature':<20} {'Importance':<12}")
    print("-" * 35)
    for i in range(min(max_show, len(importances))):
        fname = f"F{i+1}" if feature_names is None else feature_names[indices[i]]
        importance = importances[indices[i]]
        print(f"{fname:<20} {importance:.6f}")
        results.append((fname, float(importance)))
    print()
    return results

# ========== FUNZIONE SALVATAGGIO REPORT ==========

def save_centralized_random_forest_report(X_val, y_val, model, final_metrics, feature_importance_before=None, feature_importance_after=None):
    """
    Salva un report per Random Forest centralizzato.
    Diverso dalla versione DNN perché Random Forest non ha epoche di training.
    
    Args:
        X_val: Dati di validation
        y_val: Etichette di validation
        model: Modello Random Forest addestrato
        final_metrics: Metriche finali del modello
        feature_importance_before: Feature importance prima del preprocessing
        feature_importance_after: Feature importance dopo il preprocessing
    """
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, f"centralized_random_forest_report_{timestamp}.txt")

    # Predizioni per matrice di confusione
    y_pred_binary = model.predict(X_val)
    conf_matrix = confusion_matrix(y_val, y_pred_binary)
    
    # Header del report
    title = "RESOCONTO ADDESTRAMENTO CENTRALIZZATO SMARTGRID - RANDOM FOREST"
    header_lines = []
    header_lines.append(title)
    header_lines.append("=" * len(title))
    header_lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header_lines.append(f"Modello: Random Forest")
    header_lines.append(f"N. Estimatori: {model.n_estimators}")
    header_lines.append(f"Max Depth: {model.max_depth}")
    header_lines.append(f"Random State: {model.random_state}")
    header_lines.append("")

    # Metriche finali
    metrics_lines = []
    metrics_lines.append("METRICHE FINALI:")
    metrics_lines.append("=" * 40)
    metrics_lines.append(f"Accuracy: {final_metrics['accuracy']:.6f}")
    metrics_lines.append(f"F1-Score: {final_metrics['f1_score']:.6f}")
    metrics_lines.append(f"Balanced Accuracy: {final_metrics['balanced_accuracy']:.6f}")
    metrics_lines.append(f"Precision: {final_metrics['precision']:.6f}")
    metrics_lines.append(f"Recall: {final_metrics['recall']:.6f}")
    metrics_lines.append(f"AUC: {final_metrics['auc']:.6f}")
    metrics_lines.append("")
    
    # Metriche per classe
    metrics_lines.append("METRICHE PER CLASSE:")
    metrics_lines.append("-" * 30)
    metrics_lines.append("CLASSE NATURAL:")
    metrics_lines.append(f"  Precision: {final_metrics['precision_natural']:.6f}")
    metrics_lines.append(f"  Recall: {final_metrics['recall_natural']:.6f}")
    metrics_lines.append(f"  F1-Score: {final_metrics['f1_natural']:.6f}")
    metrics_lines.append(f"  Support: {final_metrics['support_natural']}")
    metrics_lines.append("")
    metrics_lines.append("CLASSE ATTACK:")
    metrics_lines.append(f"  Precision: {final_metrics['precision_attack']:.6f}")
    metrics_lines.append(f"  Recall: {final_metrics['recall_attack']:.6f}")
    metrics_lines.append(f"  F1-Score: {final_metrics['f1_attack']:.6f}")
    metrics_lines.append(f"  Support: {final_metrics['support_attack']}")
    metrics_lines.append("")

    # Matrice di confusione
    conf_matrix_lines = []
    conf_matrix_lines.append("MATRICE DI CONFUSIONE SUL VALIDATION SET:")
    conf_matrix_lines.append("-" * 40)
    conf_matrix_lines.append(f"True Positive (TP):  {conf_matrix[1, 1]}")
    conf_matrix_lines.append(f"False Positive (FP): {conf_matrix[0, 1]}")
    conf_matrix_lines.append(f"False Negative (FN): {conf_matrix[1, 0]}")
    conf_matrix_lines.append(f"True Negative (TN):  {conf_matrix[0, 0]}")
    conf_matrix_lines.append("")

    # Feature importance del modello Random Forest
    model_fi_lines = []
    model_fi_lines.append("FEATURE IMPORTANCE DEL MODELLO RANDOM FOREST:")
    model_fi_lines.append("-" * 60)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        model_fi_lines.append(f"{'Rank':<6} {'Feature':<15} {'Importance':<12}")
        model_fi_lines.append("-" * 35)
        for i in range(min(20, len(importances))):
            idx = indices[i]
            model_fi_lines.append(f"{i+1:<6} Feature_{idx:<10} {importances[idx]:.6f}")
    else:
        model_fi_lines.append("Feature importance non disponibile")
    model_fi_lines.append("")

    # Feature importance prima e dopo preprocessing 
    fi_lines = []
    fi_lines.append("FEATURE IMPORTANCE PRIMA DELLA PREPROCESSING:")
    fi_lines.append("-" * 60)
    if feature_importance_before is not None:
        fi_lines.append(f"{'Feature':<25} {'Importance':<12}")
        fi_lines.append("-" * 40)
        for fname, imp in feature_importance_before:
            fi_lines.append(f"{fname:<25} {imp:.6f}")
    else:
        fi_lines.append("Non disponibile")
    fi_lines.append("")
    
    fi_lines.append("FEATURE IMPORTANCE DOPO LA PREPROCESSING:")
    fi_lines.append("-" * 60)
    if feature_importance_after is not None:
        fi_lines.append(f"{'Feature/Componente':<25} {'Importance':<12}")
        fi_lines.append("-" * 40)
        for fname, imp in feature_importance_after:
            fi_lines.append(f"{fname:<25} {imp:.6f}")
    else:
        fi_lines.append("Non disponibile")
    fi_lines.append("")

    # Scrivi il file
    with open(report_path, "w") as f:
        # Scrivi tutte le sezioni
        for line in header_lines:
            f.write(line + "\n")
        for line in metrics_lines:
            f.write(line + "\n")
        for line in conf_matrix_lines:
            f.write(line + "\n")
        for line in model_fi_lines:
            f.write(line + "\n")
        for line in fi_lines:
            f.write(line + "\n")
    
    print(f"\n[SERVER] ✅ Report Random Forest centralizzato salvato in: {report_path}")

# ========== MAIN ==========

def main():
    print("INIZIO ADDESTRAMENTO RANDOM FOREST CENTRALIZZATO SMARTGRID")
    try:
        # Carica e prepara i dati
        X, y, dataset_info = load_centralized_smartgrid_data()
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_validation_test(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
        )
        X_train_final, X_val_final, X_test_final = centralized_preprocessing(X_train_raw, X_val_raw, X_test_raw)

        # Determina dinamicamente il numero di feature in input
        input_features = X_train_final.shape[1]
        print(f"\n[Centralizzato] Feature: {dataset_info['features']} → {input_features}")

        # Random Forest gestisce automaticamente i class weights con class_weight='balanced'
        print(f"\n[Centralizzato] Class weights: gestiti automaticamente da Random Forest")

        # Crea e addestra il modello Random Forest
        model = create_smartgrid_random_forest_model(input_features)
        history = train_smartgrid_random_forest_model(model, X_train_final, y_train, X_val_final, y_val)
        
        print("\n" + "=" * 80)
        
        # Valutazione finale
        final_loss, final_accuracy, final_metrics = evaluate_smartgrid_random_forest_model(
            model, X_test_final, y_test, "Test", threshold=0.5
        )

        # Calcola feature importance PRIMA e DOPO la preprocessing
        print("\n" + "=" * 60)
        feature_importance_before = feature_importance_analysis(
            X_train_raw.values, y_train, feature_names=list(X_train_raw.columns), 
            title="Feature Importance (prima del preprocessing)", max_show=20
        )
        
        feature_importance_after = feature_importance_analysis(
            X_train_final, y_train, 
            feature_names=[f"F{i+1}" for i in range(X_train_final.shape[1])],
            title="Feature Importance (dopo preprocessing/PCA)", max_show=20
        )

        # Salva il report 
        save_centralized_random_forest_report(
            X_val_final, y_val, model, final_metrics, 
            feature_importance_before, feature_importance_after
        )

        print("\nPipeline Random Forest centralizzata completata.\n")
        
        # Riassunto finale
        print("=" * 80)
        print("RIASSUNTO FINALE:")
        print(f"  Modello: Random Forest ({model.n_estimators} alberi)")
        print(f"  Campioni training: {len(X_train_final)}")
        print(f"  Campioni test: {len(X_test_final)}")
        print(f"  Feature finali: {input_features}")
        print(f"  Accuracy test: {final_accuracy:.4f}")
        print(f"  F1-Score test: {final_metrics['f1_score']:.4f}")
        print(f"  Balanced Accuracy test: {final_metrics['balanced_accuracy']:.4f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()