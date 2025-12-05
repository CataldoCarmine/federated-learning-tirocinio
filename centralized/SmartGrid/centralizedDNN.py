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
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
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
ENABLE_CLIPPING_OUTLIERS = False       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = True                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Cambia a False per disabilitare rimozione feature quasi-costanti
ENABLE_PCA = True  # Cambia a False per disabilitare la PCA

if ENABLE_PCA:
    ENABLE_IMPUTATION= True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 71  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 21  # Seme specifico per PCA

# ========== CONFIGURAZIONE MODELLO ==========
ACTIVATION_FUNCTION = 'leaky_relu'  # Ottimizzabile: 'leaky_relu', 'selu', 'relu'
USE_ADAMW = False  # Ottimizzabile: True per AdamW, False per Adam
EXTENDED_DROPOUT = True  # Ottimizzabile: True per dropout esteso

LEARNING_RATE = 0.00033732651610264363
DROPOUT_RATE = 0.4
DROPOUT_FINAL = DROPOUT_RATE * 0.75
L2_REG = 0.002063680713812367
BATCH_SIZE = 32
EPOCHS = 100

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

    # Pulizia dei dati
    if ENABLE_CLEAN_INF_NAN:
        X_train_clean = clean_data_for_pca(X_train_raw)
        X_val_clean = clean_data_for_pca(X_val_raw)
        X_test_clean = clean_data_for_pca(X_test_raw)
    else:
        X_train_clean = X_train_raw.values if hasattr(X_train_raw, 'values') else X_train_raw
        X_val_clean = X_val_raw.values if hasattr(X_val_raw, 'values') else X_val_raw
        X_test_clean = X_test_raw.values if hasattr(X_test_raw, 'values') else X_test_raw

    # Clipping outlier per quantili
    if ENABLE_CLIPPING_OUTLIERS:
        lower, upper = fit_clip_outliers_iqr(X_train_clean, k=5.0)
        X_train_clipped = transform_clip_outliers_iqr(X_train_clean, lower, upper)
        X_val_clipped = transform_clip_outliers_iqr(X_val_clean, lower, upper)
        X_test_clipped = transform_clip_outliers_iqr(X_test_clean, lower, upper)
    else:
        X_train_clipped = X_train_clean
        X_val_clipped = X_val_clean
        X_test_clipped = X_test_clean

    # Imputazione dei valori mancanti
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_val_imputed = imputer.transform(X_val_clipped)
        X_test_imputed = imputer.transform(X_test_clipped)
    else:
        X_train_imputed = X_train_clipped
        X_val_imputed = X_val_clipped
        X_test_imputed = X_test_clipped

    # Rimozione delle feature quasi-costanti (se abilitata)
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

    # Scaling standard (mean=0, std=1)
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

    # PCA (se abilitata)
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

def create_smartgrid_dnn_model(input_features):
    """Crea il modello DNN identico a quello federato, dinamico sul numero di input."""
    set_reproducibility_seeds()
    print(f"[Centralizzato] === CREAZIONE DNN ===")
    print(f"[Centralizzato] Input features: {input_features}")
    print(f"[Centralizzato] Architettura: {input_features} → ... → 1")
    print(f"[Centralizzato] Attivazione: {ACTIVATION_FUNCTION}")
    print(f"[Centralizzato] Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"[Centralizzato] Dropout esteso: {EXTENDED_DROPOUT}")

    if ACTIVATION_FUNCTION == 'leaky_relu':
        activation_layer = lambda: layers.LeakyReLU(alpha=0.01)
        initializer = 'he_normal'
    elif ACTIVATION_FUNCTION == 'selu':
        activation_layer = lambda: layers.Activation('selu')
        initializer = 'lecun_normal'
    else:  # relu default
        activation_layer = lambda: layers.Activation('relu')
        initializer = 'he_normal'
    
    model = keras.Sequential([
        layers.Input(shape=(input_features,), name='input_layer'),
        layers.Dense(256, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_1'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(DROPOUT_RATE, name='dropout_1'),

        layers.Dense(128, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_2'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(DROPOUT_RATE if EXTENDED_DROPOUT else 0.0, name='dropout_2'),

        layers.Dense(64, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_3'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(DROPOUT_RATE, name='dropout_3'),

        layers.Dense(32, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_4'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(DROPOUT_FINAL, name='dropout_4'),

        layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='output_layer')
    ])

    if USE_ADAMW:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Centralizzato] Ottimizzatore: AdamW")
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Centralizzato] Ottimizzatore: Adam")

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    model.summary()
    return model

def create_training_callbacks():
    """Crea i callback di training ottimizzati."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            mode='min',
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=4,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
    ]
    return callbacks

def train_smartgrid_dnn_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("=== ADDESTRAMENTO DNN CENTRALIZZATO ===")
    callbacks = create_training_callbacks()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
        class_weight=class_weights
    )
    return history

def evaluate_smartgrid_model(model, X_test, y_test, set_name="Test", threshold=0.5):
    """
    VALUTAZIONE FINALE DEL MODELLO DNN SUL SET SPECIFICATO.

    Restituisce:
        loss, accuracy, metrics_dict

    Nota:
    - model.evaluate restituisce la loss e le metriche compilate in model.compile.
    - Qui calcoliamo F1, balanced accuracy e confusion matrix usando model.predict.
    - Modifica minima: il dizionario delle metriche ora include anche 'loss' per poter essere salvato nel report.
    """
    print(f"=== VALUTAZIONE FINALE DNN SMARTGRID - {set_name.upper()} SET ===")
    # Esegui la valutazione Keras (ritorna loss + metrics definiti in compile)
    results = model.evaluate(X_test, y_test, verbose=0)
    # Scompone i risultati (assumiamo ordine: loss, accuracy, precision, recall, auc)
    try:
        loss, accuracy, precision, recall, auc = results
    except Exception:
        # Fallback robusto se l'ordine/numero metriche è diverso
        # Assumiamo almeno loss e accuracy
        loss = results[0] if len(results) > 0 else 0.0
        accuracy = results[1] if len(results) > 1 else 0.0
        precision = results[2] if len(results) > 2 else 0.0
        recall = results[3] if len(results) > 3 else 0.0
        auc = results[4] if len(results) > 4 else 0.0

    # Calcolo F1 e altre metriche non fornite direttamente da Keras
    f1_score_val = 0.0
    balanced_acc = 0.0
    try:
        y_pred_prob = model.predict(X_test, verbose=0)
        # gestisci output probabilità o vettore
        if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
            y_prob_pos = y_pred_prob[:, 1]
        else:
            y_prob_pos = y_pred_prob.flatten()
        y_pred_binary = (y_prob_pos > threshold).astype(int).flatten()
        # metriche
        from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
        f1_score_val = f1_score(y_test, y_pred_binary, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred_binary)
        report = classification_report(y_test, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
    except Exception as e:
        print(f"⚠️ Errore nel calcolo metriche test dettagliate: {e}")
        report = {"natural": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
                  "attack":  {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}}
        conf_matrix = np.array([[0, 0], [0, 0]])

    # Stampa risultati principali su console
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1_score_val:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"Classification report (per classe):")
    try:
        print(classification_report(y_test, y_pred_binary, target_names=["natural", "attack"], zero_division=0))
    except Exception:
        print("  (classification_report non disponibile)")
    print(f"Confusion matrix:")
    print(conf_matrix)

    # Restituisce loss, accuracy e un dizionario con tutte le metriche (ora include 'loss')
    return loss, accuracy, {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "f1_score": float(f1_score_val),
        "balanced_accuracy": float(balanced_acc),
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

# ========== NUOVA FUNZIONE: SALVA METRICHE E FEATURE IMPORTANCE IN FILE TXT ==========

def save_centralized_training_report(history, X_val, y_val, model, feature_importance_before=None, feature_importance_after=None, final_test_metrics=None):
    """
    Salva un file txt con una tabella delle metriche ad ogni epoca, statistiche per metrica,
    e una sezione con la feature importance prima/dopo PCA.

    Modifica minima: se final_test_metrics contiene 'loss', la stampa nella sezione Test.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, f"centralized_training_report_{timestamp}.txt")

    # Definizione colonne (come prima)
    cols = [
        ("epoch", "Epoch", 6),
        ("loss", "Loss", 11),
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

    # Numero epoche (compatibilità Keras)
    n_epochs = len(history.history.get("loss", []))
    metric_rows = []
    for i in range(n_epochs):
        # Prendiamo i valori di loss/accuracy/auc da history (validation)
        loss = history.history.get("val_loss", [np.nan]*n_epochs)[i] if "val_loss" in history.history else np.nan
        accuracy = history.history.get("val_accuracy", [np.nan]*n_epochs)[i] if "val_accuracy" in history.history else np.nan
        auc = history.history.get("val_auc", [np.nan]*n_epochs)[i] if "val_auc" in history.history else np.nan

        # Metriche custom calcolate con il modello finale sul validation (approssimazione)
        try:
            y_pred_prob = model.predict(X_val, verbose=0)
            if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
                y_prob_pos = y_pred_prob[:, 1]
            else:
                y_prob_pos = y_pred_prob.flatten()
            y_pred_binary = (y_prob_pos > 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_val, y_pred_binary)
            f1_val = f1_score(y_val, y_pred_binary, zero_division=0)
            report_dict = classification_report(y_val, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            precision_weighted = report_dict.get("weighted avg", {}).get("precision", np.nan)
            recall_weighted = report_dict.get("weighted avg", {}).get("recall", np.nan)
        except Exception:
            balanced_acc = np.nan
            f1_val = np.nan
            report_dict = {}

        metric_rows.append({
            "epoch": i+1,
            "loss": loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "auc": auc,
            "f1_score": report_dict.get("weighted avg", {}).get("f1-score", np.nan),
            "f1_natural": report_dict.get("natural", {}).get("f1-score", np.nan),
            "f1_attack": report_dict.get("attack", {}).get("f1-score", np.nan),
            "precision": precision_weighted,
            "precision_natural": report_dict.get("natural", {}).get("precision", np.nan),
            "precision_attack": report_dict.get("attack", {}).get("precision", np.nan),
            "recall": recall_weighted,
            "recall_natural": report_dict.get("natural", {}).get("recall", np.nan),
            "recall_attack": report_dict.get("attack", {}).get("recall", np.nan),
        })

    # Header tabella, statistiche ecc. (stessa logica precedente)
    title = "RESOCONTO ADDESTRAMENTO CENTRALIZZATO SMARTGRID"
    n_epochs = len(metric_rows)
    header = f"{title}\nEpoche totali: {n_epochs}\n\nTABELLA RIASSUNTIVA METRICHE:\n" + "="*140 + "\n"
    col_headers = "  ".join([name.ljust(width) for _, name, width in cols])
    sep = "-" * 140

    table_lines = [col_headers, sep]
    for row in metric_rows:
        vals = []
        for k, _, width in cols:
            v = row.get(k, None)
            if k == "epoch":
                vals.append(str(v).ljust(width))
            else:
                vals.append(fmt(v, width))
        table_lines.append("  ".join(vals))

    # Statistiche finali (come prima)
    stats_lines = ["\nSTATISTICHE FINALI:\n" + "="*60 + "\n"]
    for k, name, width in cols:
        if k == "epoch":
            continue
        vals = [row[k] for row in metric_rows if row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k]))]
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
        stats_lines.append(f"   Epoche disponibili  : {len(vals)}")
        stats_lines.append(f"   Valore iniziale     : {fmt(start, 9)}")
        stats_lines.append(f"   Valore finale       : {fmt(end, 9)}")
        stats_lines.append(f"   Valore minimo       : {fmt(minv, 9)}")
        stats_lines.append(f"   Valore massimo      : {fmt(maxv, 9)}")
        stats_lines.append(f"   Valore medio        : {fmt(meanv, 9)}")
        stats_lines.append(f"   Miglioramento       : {fmt(miglioramento, 9)} {trend}")
        stats_lines.append("")

    # Matrice di confusione sul validation (modello finale)
    try:
        conf_matrix = confusion_matrix(y_val, (model.predict(X_val, verbose=0).flatten() > 0.5).astype(int))
        conf_matrix_lines = []
        conf_matrix_lines.append("\nMATRICE DI CONFUSIONE SUL VALIDATION SET:\n" + "-"*40)
        conf_matrix_lines.append(f"{'tp:':<2} {conf_matrix[1, 1]:<5} {'fp:':<2} {conf_matrix[0, 1]:<5} {'fn:':<2} {conf_matrix[1, 0]:<5} {'tn:':<2} {conf_matrix[0, 0]:<5}\n")
    except Exception:
        conf_matrix_lines = ["\nMATRICE DI CONFUSIONE SUL VALIDATION SET: Non disponibile\n"]

    # ====== SEZIONE TEST FINALI (MINIMA) ======
    test_lines = []
    if final_test_metrics is not None:
        fm = final_test_metrics
        test_lines.append("\nMETRICHE TEST FINALI (Valutazione indipendente):\n" + "="*60)
        # Stampa la loss del test se presente
        if 'loss' in fm:
            test_lines.append(f"  Loss (test): {fm.get('loss', 0):.6f}")
        test_lines.append(f"  Accuracy (test): {fm.get('accuracy', 0):.6f}")
        test_lines.append(f"  F1-Score (test): {fm.get('f1_score', 0):.6f}")
        test_lines.append(f"  Balanced Accuracy (test): {fm.get('balanced_accuracy', 0):.6f}")
        test_lines.append(f"  Precision (test): {fm.get('precision', 0):.6f}")
        test_lines.append(f"  Recall (test): {fm.get('recall', 0):.6f}")
        test_lines.append(f"  AUC (test): {fm.get('auc', 0):.6f}")
        test_lines.append("")
        # Per-class metrics se presenti
        if 'precision_natural' in fm:
            test_lines.append("  METRICHE PER CLASSE (TEST):")
            test_lines.append(f"    Natural - Precision: {fm.get('precision_natural', 0):.6f}, Recall: {fm.get('recall_natural', 0):.6f}, F1: {fm.get('f1_natural', 0):.6f}, Support: {fm.get('support_natural', 0)}")
            test_lines.append(f"    Attack  - Precision: {fm.get('precision_attack', 0):.6f}, Recall: {fm.get('recall_attack', 0):.6f}, F1: {fm.get('f1_attack', 0):.6f}, Support: {fm.get('support_attack', 0)}")
            test_lines.append("")
        # Confusion matrix test se disponibile
        if all(k in fm for k in ('tn', 'fp', 'fn', 'tp')):
            test_lines.append("  MATRICE DI CONFUSIONE SUL TEST SET:")
            test_lines.append(f"    TP: {fm['tp']}, FP: {fm['fp']}, FN: {fm['fn']}, TN: {fm['tn']}")
            test_lines.append("")
        else:
            test_lines.append("  MATRICE DI CONFUSIONE SUL TEST SET: Non disponibile (dati mancanti)\n")
    else:
        test_lines.append("\nMETRICHE TEST FINALI: Non fornite\n")

    # SEZIONE FEATURE IMPORTANCE (PRIMA E DOPO PCA)
    fi_lines = []
    fi_lines.append("\nFEATURE IMPORTANCE PRIMA DELLA PCA (RandomForest):\n" + "-"*60)
    if feature_importance_before is not None:
        fi_lines.append(f"{'Feature':<25} {'Importance':<12}")
        fi_lines.append("-" * 40)
        for fname, imp in feature_importance_before:
            fi_lines.append(f"{fname:<25} {imp:.6f}")
    else:
        fi_lines.append("Non disponibile")
    fi_lines.append("\nFEATURE IMPORTANCE DOPO LA PCA (RandomForest):\n" + "-"*60)
    if feature_importance_after is not None:
        fi_lines.append(f"{'PCA_Component':<25} {'Importance':<12}")
        fi_lines.append("-" * 40)
        for fname, imp in feature_importance_after:
            fi_lines.append(f"{fname:<25} {imp:.6f}")
    else:
        fi_lines.append("Non disponibile")

    # Scrivi il file
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(header)
        for line in table_lines:
            f.write(line + "\n")
        f.write("="*140 + "\n")
        for line in stats_lines:
            f.write(line + "\n")
        for line in conf_matrix_lines:
            f.write(line + "\n")
        for line in test_lines:
            f.write(line + "\n")
        for line in fi_lines:
            f.write(line + "\n")
    print(f"\n[SERVER] ✅ Report addestramento centralizzato salvato in: {report_path}")

# ========== MAIN ==========

def main():
    print("INIZIO ADDESTRAMENTO DNN CENTRALIZZATO SMARTGRID (PIPELINE FEDERATA + FEATURE IMPORTANCE)")
    try:
        X, y, dataset_info = load_centralized_smartgrid_data()
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_validation_test(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
        )
        X_train_final, X_val_final, X_test_final = centralized_preprocessing(X_train_raw, X_val_raw, X_test_raw)

        # Determina dinamicamente il numero di feature in input
        input_features = X_train_final.shape[1]
        print(f"\n[Centralizzato] Feature: {dataset_info['features']} → {input_features}")

        # Calcola class weights
        class_weights = compute_class_weights(y_train)
        print(f"\n[Centralizzato] Class weights: {class_weights}")

        # Crea modello
        model = create_smartgrid_dnn_model(input_features)
        history = train_smartgrid_dnn_model(model, X_train_final, y_train, X_val_final, y_val, class_weights)
        print("\n" + "=" * 80)
        # Valutazione finale SUL TEST SET (stampa su terminale) - CATTURA final_metrics
        final_loss, final_accuracy, final_metrics = evaluate_smartgrid_model(model, X_test_final, y_test, "Test", threshold=0.5)

        # Calcola feature importance PRIMA e DOPO la PCA (se vuoi, puoi modificarlo secondo le tue esigenze)
        feature_importance_before = feature_importance_analysis(
            X_train_raw.values, y_train, feature_names=list(X_train_raw.columns), 
            title="Feature Importance (prima del preprocessing)", max_show=20
        )
        feature_importance_after = feature_importance_analysis(
            X_train_final, y_train, feature_names=[f"F{i+1}" for i in range(X_train_final.shape[1])],
            title="Feature Importance (dopo preprocessing/PCA)", max_show=20
        )
        # Passiamo le metriche finali del test al report
        save_centralized_training_report(
            history, X_val_final, y_val, model,
            feature_importance_before=feature_importance_before,
            feature_importance_after=feature_importance_after,
            final_test_metrics=final_metrics  # <-- aggiunto, minima modifica
        )

        print("\nPipeline centralizzata completata.\n")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()