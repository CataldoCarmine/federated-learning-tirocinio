import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import sys
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

# CONFIGURAZIONE PCA STATICA FISSA
PCA_COMPONENTS = 21  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_STATE = 42

# CONFIGURAZIONE MODELLO DNN - PARAMETRI OTTIMIZZABILI CON OPTUNA
ACTIVATION_FUNCTION = 'relu'  # Ottimizzabile: 'leaky_relu', 'selu', 'relu'
USE_ADAMW = False  # Ottimizzabile: True per AdamW, False per Adam
EXTENDED_DROPOUT = True  # Ottimizzabile: True per dropout esteso

def clip_outliers_iqr(X, k=5.0):
    """
    Clippa gli outlier per ogni feature usando la regola dei quantili (IQR).
    Limiti: [Q1 - k*IQR, Q3 + k*IQR] (default k=5.0).
    """
    X_clipped = X.copy()
    for col in range(X_clipped.shape[1]):
        col_data = X_clipped[:, col]
        q1 = np.nanpercentile(col_data, 25)
        q3 = np.nanpercentile(col_data, 75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        X_clipped[:, col] = np.clip(col_data, lower, upper)
    return X_clipped

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
    - NON usa threshold fissi
    - NON azzera valori piccoli
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
    GARANZIA: Output sempre con PCA_COMPONENTS dimensioni.
    """
    print(f"[Client {client_id}] === APPLICAZIONE PCA FISSA (SEMPLIFICATA) ===")
    original_features = X_preprocessed.shape[1]
    n_samples = len(X_preprocessed)
    n_components = min(PCA_COMPONENTS, original_features, n_samples)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
            X_pca = pca.fit_transform(X_preprocessed)
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

def compute_class_weights(y_train):
    """
    Calcola i pesi delle classi per compensare lo sbilanciamento.
    Versione semplificata.
    """
    try:
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        return class_weight_dict
    except Exception as e:
        print(f"Errore nel calcolo class weights: {e}")
        unique_classes = np.unique(y_train)
        return {cls: 1.0 for cls in unique_classes}

def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico con PCA FISSA.
    Applica:
      - Pulizia inf/NaN
      - Clipping outlier per quantili (feature-wise)
      - Imputazione mediana
      - Rimozione feature quasi-costanti
      - Scaling standard
      - PCA
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")
    df = pd.read_csv(file_path)
    print(f"[Client {client_id}] === CARICAMENTO CON PCA FISSA E PREPROCESSING ROBUSTO ===")
    print(f"[Client {client_id}] Dataset caricato: {len(df)} campioni")
    print(f"[Client {client_id}] PCA fissa: {PCA_COMPONENTS} componenti (compatibilità garantita)")
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"[Client {client_id}] Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
    # Pulizia preliminare: solo inf/NaN
    X_cleaned = clean_data_for_pca(X)
    # STEP 1: Suddivisione train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_cleaned, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"[Client {client_id}] Suddivisione: {len(X_train_raw)} training, {len(X_val_raw)} validation")
    # STEP 2: Clipping outlier per quantili SOLO su training, applicato anche a validation usando limiti del train
    X_train_np = np.array(X_train_raw, dtype=float)
    X_val_np = np.array(X_val_raw, dtype=float)
    # Calcola limiti clipping su training
    q1 = np.nanpercentile(X_train_np, 25, axis=0)
    q3 = np.nanpercentile(X_train_np, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    X_train_clipped = np.clip(X_train_np, lower, upper)
    X_val_clipped = np.clip(X_val_np, lower, upper)
    # STEP 3: Imputazione mediana
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clipped)
    X_val_imputed = imputer.transform(X_val_clipped)
    # STEP 4: Rimozione feature quasi-costanti (usando solo il train)
    X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
    X_val_reduced = X_val_imputed[:, keep_mask]
    print(f"[Client {client_id}] Feature dopo rimozione quasi-costanti: {X_train_reduced.shape[1]} (da {X_train.shape[1] if 'X_train' in locals() else X_train_imputed.shape[1]})")
    # STEP 5: Scaling standard
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_val_scaled = scaler.transform(X_val_reduced)
    print(f"[Client {client_id}] Preprocessing completato (clipping, imputazione, costanti, scaling)")
    # STEP 6: PCA FISSA (compatibilità automatica)
    X_train_final = apply_pca(X_train_scaled, client_id=client_id)
    X_val_final = apply_pca(X_val_scaled, client_id=client_id)
    expected_shape = (len(X_train_final), PCA_COMPONENTS)
    if X_train_final.shape[1] != PCA_COMPONENTS:
        raise RuntimeError(f"Client {client_id}: PCA output shape inconsistente: {X_train_final.shape} vs {expected_shape}")
    class_weights = compute_class_weights(y_train)
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
        'pca_features': X_train_final.shape[1],
        'pca_components_fixed': PCA_COMPONENTS,
        'class_weights': class_weights,
        'preprocessing_method': 'iqr_clipping_impute_remove_constants_scaling',
        'compatibility_guaranteed': True
    }
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO (PCA FISSA ROBUSTA) ===")
    print(f"[Client {client_id}]   - Training: {X_train_final.shape}")
    print(f"[Client {client_id}]   - Validation: {X_val_final.shape}")
    print(f"[Client {client_id}]   - Compatibilità: GARANTITA (PCA fissa)")
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_dnn_model():
    """
    Crea il modello DNN SmartGrid con architettura FISSA per compatibilità garantita.
    SEMPLIFICATO: Architettura sempre identica = nessun controllo compatibilità necessario.
    
    Returns:
        Modello Keras compilato con architettura fissa
    """
    print(f"[Client] === CREAZIONE DNN ARCHITETTURA FISSA ===")
    print(f"[Client] Input features: {PCA_COMPONENTS} (FISSO - compatibilità garantita)")
    print(f"[Client] Architettura: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1 (FISSA)")
    print(f"[Client] Attivazione: {ACTIVATION_FUNCTION}")
    print(f"[Client] Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"[Client] Dropout esteso: {EXTENDED_DROPOUT}")
    
    # PARAMETRI OTTIMIZZABILI CON OPTUNA
    dropout_rate = 0.2         # Ottimizzabile
    dropout_final = 0.15         # Ottimizzabile
    l2_reg = 0.0002726058480553248             # Ottimizzabile
    
    # ARCHITETTURA FISSA: garantisce compatibilità automatica
    # PCA_COMPONENTS → ... → 1 (sempre uguale)
    
    # Selezione funzione di attivazione (ottimizzabile)
    if ACTIVATION_FUNCTION == 'leaky_relu':
        activation_layer = lambda: layers.LeakyReLU(alpha=0.01)
        initializer = 'he_normal'
    elif ACTIVATION_FUNCTION == 'selu':
        activation_layer = lambda: layers.Activation('selu')
        initializer = 'lecun_normal'
    else:  # relu default
        activation_layer = lambda: layers.Activation('relu')
        initializer = 'he_normal'
    
    print(f"[Client] Funzione attivazione: {ACTIVATION_FUNCTION}, Initializer: {initializer}")
    
    # MODELLO CON ARCHITETTURA FISSA (sempre identica)
    model = keras.Sequential([
        # Input layer esplicito con dimensione FISSA
        layers.Input(shape=(PCA_COMPONENTS,), name='input_layer'),

        # Layer 1: 112 neuroni (FISSO - ottimizzabile con Optuna)
        layers.Dense(112, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_1'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),

        # Layer 2: 64 neuroni (FISSO - ottimizzabile con Optuna)
        layers.Dense(64, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_2'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate if EXTENDED_DROPOUT else 0.0, name='dropout_2'),

        # Layer 3: 12 neuroni (FISSO - ottimizzabile con Optuna)
        layers.Dense(12, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_3'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate, name='dropout_3'),

        # Layer 4: 10 neuroni (FISSO - ottimizzabile con Optuna)
        layers.Dense(10, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_4'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(dropout_final, name='dropout_4'),
        
        # Output layer: 1 neurone (FISSO)
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # OTTIMIZZATORE CONFIGURABILE (ottimizzabile con Optuna)
    if USE_ADAMW:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.006025741928842929,  # Ottimizzabile
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Client] Ottimizzatore: AdamW")
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.006025741928842929,  # Ottimizzabile
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Client] Ottimizzatore: Adam")
    
    # Compila il modello
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
    
    # Statistiche modello
    total_params = model.count_params()
    
    return model

def create_training_callbacks():
    """
    Crea i callback di training ottimizzati.
    """
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_loss',  # <-- cambia da 'loss' a 'val_loss'
            patience=3,
            restore_best_weights=True,
            verbose=0,
            mode='min',
            min_delta=0.001
        ),
        
        # Reduce Learning Rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,                   # Ottimizzabile
            patience=2,                   # Ottimizzabile
            min_lr=1e-6,
            verbose=0,
            mode='min'
        )
    ]
    
    return callbacks

# Variabili globali per il client
client_id = None
model = None
X_train = None
y_train = None
X_val = None
y_val = None
dataset_info = None

class SmartGridClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con architettura FISSA.
    SEMPLIFICATO: Rimossi tutti i controlli di compatibilità ridondanti.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello.
        SEMPLIFICATO: Nessun controllo necessario (architettura fissa).
        """
        return model.get_weights()

    def fit(self, parameters, config):
        """
        Addestra il modello con architettura fissa.
        SEMPLIFICATO: Compatibilità garantita automaticamente.
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] Round di addestramento con architettura FISSA...")
        
        # IMPOSTAZIONE PESI SEMPLIFICATA
        # Nessun controllo di compatibilità necessario (architettura fissa)
        try:
            model.set_weights(parameters)
            print(f"[Client {client_id}] ✅ Pesi impostati (compatibilità garantita)")
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore impostazione pesi: {e}")
            return model.get_weights(), 0, {'error': f'weight_setting_failed: {str(e)}'}
        
        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training!")
            return model.get_weights(), 0, {}
        
        # Configurazione addestramento (ottimizzabile con Optuna)
        local_epochs = 15            # Ottimizzabile
        batch_size = 32              # Ottimizzabile

        # Class weights
        class_weights = dataset_info['class_weights']
        
        # Callback
        callbacks = create_training_callbacks()
        
        try:
            print(f"[Client {client_id}] Training con architettura fissa:")
            print(f"[Client {client_id}]   - Epoche: {local_epochs}")
            print(f"[Client {client_id}]   - Batch size: {batch_size}")
            print(f"[Client {client_id}]   - Architettura: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1")
            
            history = model.fit(
                X_train, y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0,
                shuffle=True
            )
            
            # Estrai metriche
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            train_precision = history.history.get('precision', [0])[-1]
            train_recall = history.history.get('recall', [0])[-1]
            train_auc = history.history.get('auc', [0])[-1]
            
            # F1-score
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            # Balanced Accuracy
            y_pred_prob = model.predict(X_train, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_binary)
            
            # Early stopping info
            actual_epochs = len(history.history['loss'])
            early_stopped = actual_epochs < local_epochs
            
            print(f"[Client {client_id}] Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"[Client {client_id}] F1: {train_f1:.4f}, Balanced Acc: {train_balanced_acc:.4f}")
            print(f"[Client {client_id}] Epoche effettive: {actual_epochs}/{local_epochs}")
            if early_stopped:
                print(f"[Client {client_id}] ✅ EarlyStopping attivato")
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante addestramento: {e}")
            return model.get_weights(), 0, {'error': f'training_failed: {str(e)}'}
        
        # Metriche da inviare al server
        metrics = {
            # Metriche base
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_auc': float(train_auc),
            'train_f1_score': float(train_f1),
            'train_balanced_accuracy': float(train_balanced_acc),
            
            # Configurazione
            'local_epochs_planned': int(local_epochs),
            'local_epochs_actual': int(actual_epochs),
            'early_stopped': bool(early_stopped),
            'batch_size': int(batch_size),
            'architecture_fixed': True,
            'compatibility_guaranteed': True,
            'compatibility_checks_removed': True,
            
            # Dataset info
            'client_id': int(dataset_info['client_id']),
            'train_samples': int(dataset_info['train_samples']),
            'pca_features': int(dataset_info['pca_features']),
            'pca_components_fixed': int(dataset_info['pca_components_fixed']),
            
            # Metodologia semplificata
            'preprocessing_method': dataset_info['preprocessing_method'],
            'model_type': 'dnn_fixed_architecture_simplified'
        }
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello con architettura fissa.
        SEMPLIFICATO: Nessun controllo di compatibilità necessario.
        """
        global model, X_val, y_val
        
        # Impostazione pesi semplificata (compatibilità garantita)
        try:
            model.set_weights(parameters)
        except Exception as e:
            print(f"[Client {client_id}] Errore impostazione pesi valutazione: {e}")
            return 1.0, 0, {"accuracy": 0.0, "error": f"weight_setting_eval_failed: {str(e)}"}
        
        if len(X_val) == 0:
            return 0.0, 0, {"accuracy": 0.0}
        
        try:
            # Valutazione
            results = model.evaluate(X_val, y_val, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # F1-score
            f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Balanced Accuracy
            y_pred_prob = model.predict(X_val, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_val, y_pred_binary)
            
            print(f"[Client {client_id}] Val Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
            print(f"[Client {client_id}] Val F1: {f1_score_val:.4f}, Val Balanced Acc: {balanced_acc:.4f}")
            
            # Metriche
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "f1_score": f1_score_val,
                "balanced_accuracy": balanced_acc,
                "val_samples": len(X_val),
                "architecture_fixed": True,
                "compatibility_guaranteed": True
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante valutazione: {e}")
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

def main():
    """
    Funzione principale per avviare il client SmartGrid con architettura fissa semplificata.
    """
    global client_id, model, X_train, y_train, X_val, y_val, dataset_info
    
    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        print("Esempio: python client.py 1")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("Client ID deve essere tra 1 e 13")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"=== AVVIO CLIENT SMARTGRID DNN ARCHITETTURA FISSA {client_id} ===")
    print("CONFIGURAZIONE SEMPLIFICATA:")
    print(f"  ✅ PCA FISSA: {PCA_COMPONENTS} componenti (compatibilità garantita)")
    print("  ✅ Architettura FISSA: 35 → 64 → 32 → 16 → 8 → 1")
    print("  ✅ Controlli compatibilità: RIMOSSI (non necessari)")
    print("  ✅ Distribuzione naturale mantenuta (NO SMOTE)")
    print("  ✅ Parametri ottimizzabili con Optuna")
    
    try:
        # Carica i dati con PCA fissa
        print(f"[Client {client_id}] Caricamento dati con PCA fissa...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
        
        # Crea il modello con architettura fissa
        model = create_dnn_model()

        print(f"[Client {client_id}] === RIASSUNTO CLIENT ARCHITETTURA FISSA ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['pca_features']}")
        print(f"[Client {client_id}] PCA: {dataset_info['pca_components_fixed']} componenti FISSI")
        print(f"[Client {client_id}] Modello: {model.count_params():,} parametri")
        print(f"[Client {client_id}] Architettura: FISSA (compatibilità garantita)")
        print(f"[Client {client_id}] Compatibilità: {dataset_info['compatibility_guaranteed']}")
        print(f"[Client {client_id}] Connessione al server su localhost:8080...")
        
        # Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()