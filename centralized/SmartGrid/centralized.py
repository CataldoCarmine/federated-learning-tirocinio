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
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ========== PARAMETRI GLOBALI (identici al federato) ==========
# PCA Statici
PCA_COMPONENTS = 74
PCA_RANDOM_STATE = 42

# Configurazione modello DNN 
ACTIVATION_FUNCTION = 'leaky_relu'  # 'relu', 'leaky_relu', 'selu'
USE_ADAMW = False
EXTENDED_DROPOUT = True

# Parametri ottimizzati (Optuna)
DROPOUT_RATE = 0.4
DROPOUT_FINAL = DROPOUT_RATE * 0.75
L2_REG = 0.002063680713812367
LEARNING_RATE = 0.00033732651610264363
BATCH_SIZE = 32
EPOCHS = 100

# ========== FUNZIONI DI PREPROCESSING (identiche al federato) ==========

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
    if pca_obj is None:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=PCA_RANDOM_STATE)
        X_pca = pca.fit_transform(X)
        return X_pca, pca
    else:
        X_pca = pca_obj.transform(X)
        return X_pca

def load_centralized_smartgrid_data():
    """
    Carica e unisce tutti i dati SmartGrid per l'addestramento centralizzato.
    """
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
    """
    Pipeline identica a quella federata: clipping, imputazione, rimozione quasi-costanti, scaling.
    """
    # Pulizia dei dati
    X_train_clean = clean_data_for_pca(X_train_raw)
    X_val_clean = clean_data_for_pca(X_val_raw)
    X_test_clean = clean_data_for_pca(X_test_raw)


    lower, upper = fit_clip_outliers_iqr(X_train_clean, k=5.0)
    X_train_clipped = transform_clip_outliers_iqr(X_train_clean, lower, upper)
    X_val_clipped = transform_clip_outliers_iqr(X_val_clean, lower, upper)
    X_test_clipped = transform_clip_outliers_iqr(X_test_clean, lower, upper)

    # Imputazione dei valori mancanti
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clipped)
    X_val_imputed = imputer.transform(X_val_clipped)
    X_test_imputed = imputer.transform(X_test_clipped)

    # Rimozione delle feature quasi-costanti
    X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
    X_val_reduced = X_val_imputed[:, keep_mask]
    X_test_reduced = X_test_imputed[:, keep_mask]

    # Scaling standard (mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_val_scaled = scaler.transform(X_val_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    return X_train_scaled, X_val_scaled, X_test_scaled

# ========== FUNZIONI MODELLO/ CALLBACK (identici federato) ==========

def create_smartgrid_dnn_model():
    """
    Modello DNN identico a quello federato, parametri globali.
    """
    print(f"[Centralizzato] === CREAZIONE DNN ===")
    print(f"[Centralizzato] Input features: {PCA_COMPONENTS}")
    print(f"[Centralizzato] Architettura: {PCA_COMPONENTS} → ... → 1")
    print(f"[Centralizzato] Attivazione: {ACTIVATION_FUNCTION}")
    print(f"[Centralizzato] Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"[Centralizzato] Dropout esteso: {EXTENDED_DROPOUT}")

    # Selezione funzione di attivazione
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
        layers.Input(shape=(PCA_COMPONENTS,), name='input_layer'),
        layers.Dense(32, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_1'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(DROPOUT_RATE, name='dropout_1'),

        layers.Dense(48, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_2'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(DROPOUT_RATE if EXTENDED_DROPOUT else 0.0, name='dropout_2'),

        layers.Dense(16, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_3'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(DROPOUT_RATE, name='dropout_3'),

        layers.Dense(4, kernel_regularizer=regularizers.l2(L2_REG), kernel_initializer=initializer, name='dense_4'),
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
    """
    Callback identici al federato.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8, #Aumentato, l'addestramento si ferma troppo presto
            restore_best_weights=True,
            verbose=1,
            mode='min',
            min_delta=0.001 #Si può aumentare un po' il delta per evitare early stopping troppo sensibile
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=4, #Aumentato, l'addestramento si ferma troppo presto
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
    ]
    return callbacks

def train_smartgrid_dnn_model(model, X_train, y_train, X_val, y_val):
    print("=== ADDESTRAMENTO DNN CENTRALIZZATO ===")
    callbacks = create_training_callbacks()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    return history

def evaluate_smartgrid_model(model, X_test, y_test, set_name="Test", threshold=0.5):
    print(f"=== VALUTAZIONE FINALE DNN SMARTGRID - {set_name.upper()} SET ===")
    results = model.evaluate(X_test, y_test, verbose=0)
    loss, accuracy, precision, recall, auc = results
    f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_binary = (y_pred_prob > threshold).astype(int).flatten()
    balanced_acc = balanced_accuracy_score(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print(f"  Loss: {loss:.4f}")
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

# ========== NUOVA FUNZIONE: SALVA METRICHE E FEATURE IMPORTANCE IN FILE TXT ==========

def save_centralized_training_report(history, X_val, y_val, model, feature_importance_before=None, feature_importance_after=None):
    """
    Salva un file txt con una tabella delle metriche ad ogni epoca, statistiche per metrica,
    e una sezione con la feature importance prima/dopo PCA.
    """

    results_dir = os.path.join("centralized", "SmartGrid", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, f"centralized_training_report_{timestamp}.txt")

    # Definisci le colonne e la larghezza
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

    # Raccogli metriche per ogni epoca
    n_epochs = len(history.history["loss"])
    metric_rows = []
    for i in range(n_epochs):
        # Valuta su validation set il modello ai pesi salvati di quell'epoca
        # In keras, model.fit non salva il modello ad ogni epoca, quindi dobbiamo stimare le metriche "extra" (f1, balanced acc, ecc.)
        # usando le predizioni corrispondenti, se disponibili
        # Per semplicità e didattica, usiamo le metriche di Keras per loss, accuracy, auc
        # e calcoliamo le metriche custom (f1, balanced acc, ecc) usando il validation set e il modello finale
        # NOTA: in keras, history salva solo le metriche standard; per metriche custom, le calcoliamo sul modello migliore

        # Prendiamo i valori di loss, accuracy, auc da history
        loss = history.history["val_loss"][i] if "val_loss" in history.history else np.nan
        accuracy = history.history["val_accuracy"][i] if "val_accuracy" in history.history else np.nan
        auc = history.history["val_auc"][i] if "val_auc" in history.history else np.nan

        # Per metriche custom, usiamo il modello finale (approssimazione didattica)
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred_binary = (y_pred_prob > 0.5).astype(int).flatten()
        balanced_acc = balanced_accuracy_score(y_val, y_pred_binary)
        f1_score_val = f1_score(y_val, y_pred_binary)
        report = classification_report(y_val, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1_score_macro = report["weighted avg"]["f1-score"]

        metric_rows.append({
            "epoch": i+1,
            "loss": loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "auc": auc,
            "f1_score": f1_score_macro,
            "f1_natural": report["natural"]["f1-score"],
            "f1_attack": report["attack"]["f1-score"],
            "precision": precision,
            "precision_natural": report["natural"]["precision"],
            "precision_attack": report["attack"]["precision"],
            "recall": recall,
            "recall_natural": report["natural"]["recall"],
            "recall_attack": report["attack"]["recall"],
        })

    # Header della tabella
    title = "RESOCONTO ADDESTRAMENTO CENTRALIZZATO SMARTGRID"
    n_epochs = len(metric_rows)
    header = f"{title}\nEpoche totali: {n_epochs}\n\nTABELLA RIASSUNTIVA METRICHE:\n" + "="*140 + "\n"
    col_headers = "  ".join([name.ljust(width) for _, name, width in cols])
    sep = "-" * 140

    table_lines = []
    table_lines.append(col_headers)
    table_lines.append(sep)
    for row in metric_rows:
        vals = []
        for k, _, width in cols:
            v = row.get(k, None)
            if k == "epoch":
                vals.append(str(v).ljust(width))
            else:
                vals.append(fmt(v, width))
        table_lines.append("  ".join(vals))

    # STATISTICHE FINALI
    stats_lines = []
    stats_lines.append("\nSTATISTICHE FINALI:\n" + "="*60 + "\n")
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
        
    conf_matrix = confusion_matrix(y_val, (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten())
    conf_matrix_lines = []
    conf_matrix_lines.append("\nMATRICE DI CONFUSIONE SUL VALIDATION SET:\n" + "-"*40)
    conf_matrix_lines.append(f"{'tp:':<2} {conf_matrix[1, 1]:<5} {'fp:':<2} {conf_matrix[0, 1]:<5} {'fn:':<2} {conf_matrix[1, 0]:<5} {'tn:':<2} {conf_matrix[0, 0]:<5}\n")

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

    with open(report_path, "w") as f:
        f.write(header)
        for line in table_lines:
            f.write(line + "\n")
        f.write("="*140 + "\n")
        for line in stats_lines:
            f.write(line + "\n")
        for line in conf_matrix_lines:
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
        X_train_scaled, X_val_scaled, X_test_scaled = centralized_preprocessing(X_train_raw, X_val_raw, X_test_raw)

        print("\n[Centralizzato] Feature importance PRIMA della PCA:")
        feature_names = list(X_train_raw.columns)
        feature_importance_before = feature_importance_analysis(X_train_scaled, y_train, feature_names=feature_names, n_estimators=100, title="Prima della PCA", max_show=20)

        X_train_pca, pca_object = apply_pca(X_train_scaled)
        X_val_pca = apply_pca(X_val_scaled, pca_obj=pca_object)
        X_test_pca = apply_pca(X_test_scaled, pca_obj=pca_object)

        print("\n[Centralizzato] Feature importance DOPO la PCA (componenti PCA):")
        pca_feature_names = [f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]
        feature_importance_after = feature_importance_analysis(X_train_pca, y_train, feature_names=pca_feature_names, n_estimators=100, title="Dopo la PCA", max_show=20)

        model = create_smartgrid_dnn_model()
        history = train_smartgrid_dnn_model(model, X_train_pca, y_train, X_val_pca, y_val)
        # Salva report addestramento per tutte le epoche su validation set + feature importance
        save_centralized_training_report(history, X_val_pca, y_val, model, feature_importance_before=feature_importance_before, feature_importance_after=feature_importance_after)

        print("\n" + "=" * 80)
        final_loss, final_accuracy, final_metrics = evaluate_smartgrid_model(model, X_test_pca, y_test, "Test", threshold=0.5)

        print("\nPipeline centralizzata completata.\n")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()