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
import time
import sys
import os

# == CONFIGURAZIONI IDENTICHE AL FEDERATO ==
PCA_COMPONENTS = 21
PCA_RANDOM_STATE = 42

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
    # Se viene passato un oggetto PCA già fit, usa transform, altrimenti fit_transform
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
    Pipeline identica a quella federata: clipping, imputazione, rimozione quasi-costanti, scaling, PCA.
    """
    # Pulizia inf/NaN
    X_train_clean = clean_data_for_pca(X_train_raw)
    X_val_clean = clean_data_for_pca(X_val_raw)
    X_test_clean = clean_data_for_pca(X_test_raw)
    # Clipping IQR
    lower, upper = fit_clip_outliers_iqr(X_train_clean, k=5.0)
    X_train_clipped = transform_clip_outliers_iqr(X_train_clean, lower, upper)
    X_val_clipped = transform_clip_outliers_iqr(X_val_clean, lower, upper)
    X_test_clipped = transform_clip_outliers_iqr(X_test_clean, lower, upper)
    # Imputazione mediana
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clipped)
    X_val_imputed = imputer.transform(X_val_clipped)
    X_test_imputed = imputer.transform(X_test_clipped)
    # Rimozione quasi-costanti
    X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
    X_val_reduced = X_val_imputed[:, keep_mask]
    X_test_reduced = X_test_imputed[:, keep_mask]
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_val_scaled = scaler.transform(X_val_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    return X_train_scaled, X_val_scaled, X_test_scaled

def feature_importance_analysis(X, y, feature_names=None, n_estimators=100, title="Feature Importance", max_show=20):
    """
    Calcola e stampa la feature importance con RandomForest.
    """
    print("=== ANALISI FEATURE IMPORTANCE (RandomForest) ===")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Stampa le prime max_show feature
    print(f"{'Feature':<20} {'Importance':<12}")
    print("-" * 35)
    for i in range(min(max_show, len(importances))):
        fname = f"F{i+1}" if feature_names is None else feature_names[indices[i]]
        print(f"{fname:<20} {importances[indices[i]]:.6f}")
    print()
    return importances, indices

def create_smartgrid_dnn_model(input_shape):
    print("=== CREAZIONE MODELLO DNN CENTRALIZZATO ===")
    dropout_rate = 0.2
    l2_reg = 0.0001
    model = keras.Sequential([
        layers.Input(shape=(input_shape,), name='input_layer'),
        layers.Dense(112, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer='he_normal', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer='he_normal', name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer='he_normal', name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate, name='dropout_3'),
        layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer='he_normal', name='dense_4'),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(0.15, name='dropout_4'),
        layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='output_layer')
    ])
    optimizer = keras.optimizers.Adam(
        learning_rate=0.006, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0
    )
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
    print(f"Modello DNN creato con input shape: {input_shape}")
    model.summary()
    return model

def train_smartgrid_dnn_model(model, X_train, y_train, X_val, y_val):
    print("=== ADDESTRAMENTO DNN CENTRALIZZATO ===")
    epochs = 20
    batch_size = 32
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
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

def main():
    print("INIZIO ADDESTRAMENTO DNN CENTRALIZZATO SMARTGRID (PIPELINE FEDERATA + FEATURE IMPORTANCE)")
    try:
        # 1. Carica i dati grezzi
        X, y, dataset_info = load_centralized_smartgrid_data()
        # 2. Split train/val/test
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_validation_test(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
        )
        # 3. Pipeline identica al federato (clipping, imputazione, rimozione quasi-costanti, scaling)
        X_train_scaled, X_val_scaled, X_test_scaled = centralized_preprocessing(X_train_raw, X_val_raw, X_test_raw)

        # === FEATURE IMPORTANCE PRIMA DELLA PCA ===
        print("\n[Centralizzato] Feature importance PRIMA della PCA:")
        feature_names = list(X_train_raw.columns)
        feature_importance_analysis(X_train_scaled, y_train, feature_names=feature_names, n_estimators=100, title="Prima della PCA", max_show=20)

        # 4. PCA (fit solo sul train)
        X_train_pca, pca_object = apply_pca(X_train_scaled)
        X_val_pca = apply_pca(X_val_scaled, pca_obj=pca_object)
        X_test_pca = apply_pca(X_test_scaled, pca_obj=pca_object)

        # === FEATURE IMPORTANCE DOPO LA PCA ===
        print("\n[Centralizzato] Feature importance DOPO la PCA (componenti PCA):")
        pca_feature_names = [f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]
        feature_importance_analysis(X_train_pca, y_train, feature_names=pca_feature_names, n_estimators=100, title="Dopo la PCA", max_show=20)

        # 5. Crea e addestra il modello DNN
        model = create_smartgrid_dnn_model(X_train_pca.shape[1])
        history = train_smartgrid_dnn_model(model, X_train_pca, y_train, X_val_pca, y_val)

        # 6. Valuta sul test set
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