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
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# CONFIGURAZIONE PCA STATICA 
PCA_COMPONENTS = 74  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_STATE = 42

# CONFIGURAZIONE MODELLO DNN 
ACTIVATION_FUNCTION = 'relu'  # Ottimizzabile: 'leaky_relu', 'selu', 'relu'
USE_ADAMW = False  # Ottimizzabile: True per AdamW, False per Adam
EXTENDED_DROPOUT = True  # Ottimizzabile: True per dropout esteso

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
    """
    print(f"[Client {client_id}] === APPLICAZIONE PCA ===")

    original_features = X_preprocessed.shape[1]
    n_samples = len(X_preprocessed)
    n_components = min(PCA_COMPONENTS, original_features, n_samples)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
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

def compute_class_weights(y_train):
    """
    Calcola i pesi delle classi per compensare lo sbilanciamento.
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
    print(f"[Client {client_id}] === CARICAMENTO E PREPROCESSING DATI ===")
    print(f"[Client {client_id}] Dataset caricato: {len(df)} campioni")

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
    lower, upper = fit_clip_outliers_iqr(X_train_np, k=5.0)
    X_train_clipped = transform_clip_outliers_iqr(X_train_np, lower, upper)
    X_val_clipped = transform_clip_outliers_iqr(X_val_np, lower, upper)
    
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
   
   # STEP 6: PCA
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
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO ===")
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_dnn_model():
    """
    Crea il modello DNN SmartGrid.
    
    Returns:
        Modello Keras compilato con architettura fissa
    """
    print(f"[Client] === CREAZIONE DNN ===")
    print(f"[Client] Input features: {PCA_COMPONENTS}")
    print(f"[Client] Architettura: {PCA_COMPONENTS} → 112 → 64 → 12 → 10 → 1")
    print(f"[Client] Attivazione: {ACTIVATION_FUNCTION}")
    print(f"[Client] Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"[Client] Dropout esteso: {EXTENDED_DROPOUT}")
    
    # PARAMETRI OTTIMIZZABILI
    dropout_rate = 0.2         
    dropout_final = 0.15         
    l2_reg = 0.0002726058480553248             
    
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
    
    print(f"[Client] Funzione attivazione: {ACTIVATION_FUNCTION}, Initializer: {initializer}")
    
    # MODELLO 
    model = keras.Sequential([
        # Input layer esplicito 
        layers.Input(shape=(PCA_COMPONENTS,), name='input_layer'),

        # Layer 1: 112 neuroni
        layers.Dense(112, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_1'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),

        # Layer 2: 64 neuroni
        layers.Dense(64, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_2'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate if EXTENDED_DROPOUT else 0.0, name='dropout_2'),

        # Layer 3: 12 neuroni
        layers.Dense(12, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_3'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate, name='dropout_3'),

        # Layer 4: 10 neuroni
        layers.Dense(10, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_4'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(dropout_final, name='dropout_4'),
        
        # Output layer: 1 neurone
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # OTTIMIZZATORE CONFIGURABILE
    if USE_ADAMW:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.006025741928842929,  
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Client] Ottimizzatore: AdamW")
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.006025741928842929, 
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
            monitor='val_loss',  # monitora val_loss invece che loss
            patience=3,
            restore_best_weights=True,
            verbose=0,
            mode='min',
            min_delta=0.001
        ),
        
        # Reduce Learning Rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=2,
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
    Client Flower per SmartGrid.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello.
        """
        return model.get_weights()

    def fit(self, parameters, config):
        """
        Addestra il modello.
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] Round di addestramento ...")
        
        # IMPOSTAZIONE PESI SEMPLIFICATA
        try:
            model.set_weights(parameters)
            print(f"[Client {client_id}] ✅ Pesi impostati")
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore impostazione pesi: {e}")
            return model.get_weights(), 0, {'error': f'weight_setting_failed: {str(e)}'}
        
        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training!")
            return model.get_weights(), 0, {}
        
        # Configurazione addestramento
        local_epochs = 15
        batch_size = 32

        # Class weights
        class_weights = dataset_info['class_weights']
        
        # Callback
        callbacks = create_training_callbacks()
        
        try:
            
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
            
            # Dataset info
            'client_id': int(dataset_info['client_id']),
            'train_samples': int(dataset_info['train_samples']),
        }
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello con architettura fissa.
        """
        global model, X_val, y_val
        
        # Impostazione pesi semplificata
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
            
            # Metriche per classe
            report = classification_report(y_val, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_val, y_pred_binary)

            print(f"[Client {client_id}] Val Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
            print(f"[Client {client_id}] Val F1: {f1_score_val:.4f}, Val Balanced Acc: {balanced_acc:.4f}")
            print(f"[Client {client_id}] Classification report (per classe):")
            print(classification_report(y_val, y_pred_binary, target_names=["natural", "attack"], zero_division=0))
            print(f"[Client {client_id}] Confusion matrix:")
            print(f"tn: {conf_matrix[0, 0]}, fp: {conf_matrix[0, 1]}, fn: {conf_matrix[1, 0]}, tp: {conf_matrix[1, 1]}")
            
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
                # Nuovo: confusion matrix flatten
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
    Funzione principale per avviare il client SmartGrid.
    """
    global client_id, model, X_train, y_train, X_val, y_val, dataset_info
    
    if len(sys.argv) != 2:
        print("Usa: python client.py <client_id>")
        print("Esempio: python client.py 1")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("Client ID deve essere tra 1 e 13")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"=== AVVIO CLIENT {client_id} ===")
    
    try:
        # Carica i dati con PCA 
        print(f"[Client {client_id}] Caricamento dati con PCA ...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
        
        # Crea il modello con architettura fissa
        model = create_dnn_model()

        print(f"[Client {client_id}] === RIASSUNTO CLIENT ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['pca_features']}")
        print(f"[Client {client_id}] PCA: {dataset_info['pca_components_fixed']} componenti FISSI")
        print(f"[Client {client_id}] Modello: {model.count_params():,} parametri")
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