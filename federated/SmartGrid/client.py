import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings

def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico.
    Ogni client ha accesso solo al proprio file CSV.
    
    Args:
        client_id: ID del client (1-15)
    
    Returns:
        Tuple con (X_train_final, y_train_final, X_val_final, y_val, dataset_info)
    """
    print(f"=== CARICAMENTO DATI CLIENT {client_id} SMARTGRID ===")
    
    # Directory contenente questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    # Carica il dataset del client
    df = pd.read_csv(file_path)
    
    print(f"Dataset del client {client_id}:")
    print(f"  - Totale campioni: {len(df)}")
    print(f"  - Feature: {df.shape[1] - 1}")
    
    # Separa feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    # Statistiche del dataset locale
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    print(f"  - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"  - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    
    # STEP 1: Suddivisione train/validation (70%/30% dei dati locali) PRIMA del preprocessing
    print(f"\n=== STEP 1: SUDDIVISIONE LOCALE TRAIN/VALIDATION ===")
    
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y,
        test_size=0.3,  # 30% per validation locale
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"  - Training set locale: {len(X_train_raw)} campioni ({len(X_train_raw)/len(X)*100:.1f}%)")
    print(f"  - Validation set locale: {len(X_val_raw)} campioni ({len(X_val_raw)/len(X)*100:.1f}%)")
    
    # STEP 2-3: Pipeline di preprocessing (Imputazione + Normalizzazione)
    print(f"\n=== STEP 2-3: PIPELINE PREPROCESSING CLIENT ===")
    
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # CORREZIONE PROBLEMA 1: Pulizia robusta dei dati prima del preprocessing
    print(f"  - Applicazione pulizia robusta dei dati...")
    X_train_cleaned = clean_data_for_pca(X_train_raw)
    X_val_cleaned = clean_data_for_pca(X_val_raw)
    
    # Fit della pipeline SOLO sui dati di training del client
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_cleaned)
    X_val_preprocessed = preprocessing_pipeline.transform(X_val_cleaned)
    
    # CORREZIONE PROBLEMA 1: Controllo e pulizia post-preprocessing
    X_train_preprocessed = ensure_numerical_stability(X_train_preprocessed, "training")
    X_val_preprocessed = ensure_numerical_stability(X_val_preprocessed, "validation")
    
    print(f"  - Fit pipeline sui dati training del client")
    print(f"  - Transform applicato su training e validation")
    print(f"  - Training preprocessed shape: {X_train_preprocessed.shape}")
    print(f"  - Validation preprocessed shape: {X_val_preprocessed.shape}")
    
    # STEP 4: SMOTE solo sul training set
    print(f"\n=== STEP 4: BILANCIAMENTO CLASSI CON SMOTE ===")
    
    train_attack_ratio = y_train.mean()
    minority_class_ratio = min(train_attack_ratio, 1 - train_attack_ratio)
    unique_classes = len(np.unique(y_train))
    
    print(f"  - Distribuzione training PRIMA del bilanciamento:")
    print(f"    - Classe 0 (Natural): {(y_train == 0).sum()} campioni ({(1-train_attack_ratio)*100:.2f}%)")
    print(f"    - Classe 1 (Attack): {(y_train == 1).sum()} campioni ({train_attack_ratio*100:.2f}%)")
    
    # Applica SMOTE se necessario
    if minority_class_ratio < 0.4 and unique_classes > 1:
        print(f"  - Applicazione SMOTE...")
        
        try:
            min_samples_per_class = min((y_train == 0).sum(), (y_train == 1).sum())
            k_neighbors = min(5, min_samples_per_class - 1) if min_samples_per_class > 1 else 1
            
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)
            
            # CORREZIONE PROBLEMA 1: Controllo stabilità post-SMOTE
            X_train_balanced = ensure_numerical_stability(X_train_balanced, "post-SMOTE")
            
            print(f"  - SMOTE applicato: {len(X_train_balanced)} campioni finali")
            print(f"  - Campioni sintetici: {len(X_train_balanced) - len(X_train_preprocessed)}")
            
        except Exception as e:
            print(f"  - Errore SMOTE: {e}, uso dati originali")
            X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
    else:
        print(f"  - SMOTE non necessario o non applicabile")
        X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
    
    # STEP 5: PCA (fit sui dati training bilanciati)
    print(f"\n=== STEP 5: RIDUZIONE DIMENSIONALE CON PCA ===")
    
    original_features = X_train_balanced.shape[1]
    
    # CORREZIONE PROBLEMA 1: PCA numericamente stabile
    X_train_final, X_val_final, n_components = apply_stable_pca(
        X_train_balanced, X_val_preprocessed, variance_threshold=0.95
    )
    
    print(f"  - Training final shape: {X_train_final.shape}")
    print(f"  - Validation final shape: {X_val_final.shape}")
    
    # Informazioni del dataset per reporting
    dataset_info = {
        'client_id': client_id,
        'total_samples': len(df),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio,
        'original_features': original_features,
        'pca_features': n_components,
        'pca_reduction': (1 - n_components / original_features) * 100
    }
    
    print("=" * 60)
    
    return X_train_final, y_train_balanced, X_val_final, y_val, dataset_info

def clean_data_for_pca(X):
    """
    CORREZIONE PROBLEMA 1: Pulizia robusta dei dati per prevenire problemi numerici in PCA.
    
    Args:
        X: DataFrame o array dei dati
    
    Returns:
        Array pulito numericamente stabile
    """
    # Converti a numpy se necessario
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    
    # Rimuovi valori estremi che potrebbero causare overflow
    # Sostituisci valori molto grandi con valori più gestibili
    threshold = 1e10
    X_array = np.where(np.abs(X_array) > threshold, np.nan, X_array)
    
    return X_array

def ensure_numerical_stability(X, stage_name):
    """
    CORREZIONE PROBLEMA 1: Assicura stabilità numerica rimuovendo inf, nan e valori estremi.
    
    Args:
        X: Array dei dati
        stage_name: Nome dello stage per logging
    
    Returns:
        Array numericamente stabile
    """
    print(f"  - Controllo stabilità numerica ({stage_name})...")
    
    # Conta problemi numerici
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"    - Trovati {nan_count} NaN e {inf_count} inf")
        
        # Sostituisci NaN e inf con valori finiti
        X_clean = np.where(np.isnan(X) | np.isinf(X), 0, X)
        
        # Clip valori estremi per prevenire overflow in operazioni matriciali
        X_clean = np.clip(X_clean, -1e6, 1e6)
        
        print(f"    - Valori problematici sostituiti e clippati")
        return X_clean
    else:
        # Clip comunque per sicurezza
        X_clipped = np.clip(X, -1e6, 1e6)
        print(f"    - Dati numericamente stabili, applicato clipping preventivo")
        return X_clipped

def apply_stable_pca(X_train, X_val, variance_threshold=0.95):
    """
    CORREZIONE PROBLEMA 1: Applica PCA con controlli di stabilità numerica.
    
    Args:
        X_train: Dati di training (dopo SMOTE)
        X_val: Dati di validation
        variance_threshold: Soglia di varianza cumulativa
    
    Returns:
        Tuple (X_train_pca, X_val_pca, n_components_selected)
    """
    original_features = X_train.shape[1]
    print(f"  - Feature originali: {original_features}")
    print(f"  - Soglia varianza cumulativa: {variance_threshold*100:.1f}%")
    
    # Assicura stabilità numerica pre-PCA
    X_train_stable = ensure_numerical_stability(X_train, "pre-PCA training")
    X_val_stable = ensure_numerical_stability(X_val, "pre-PCA validation")
    
    try:
        # Sopprimi warning numerici temporaneamente per gestirli noi
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Prima esecuzione: PCA completa per analizzare la varianza
            pca_full = PCA()
            pca_full.fit(X_train_stable)
            
            # Controlla se explained_variance_ratio_ contiene valori validi
            if np.any(np.isnan(pca_full.explained_variance_ratio_)) or np.any(np.isinf(pca_full.explained_variance_ratio_)):
                raise ValueError("PCA ha prodotto explained_variance_ratio_ non validi")
            
            # Calcola varianza cumulativa
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Trova il numero di componenti necessarie per raggiungere la soglia
            n_components_selected = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # Assicurati che il numero di componenti sia valido e ragionevole
            n_components_selected = min(n_components_selected, original_features, len(X_train_stable))
            n_components_selected = max(n_components_selected, min(10, original_features))
            
            print(f"  - Componenti selezionate: {n_components_selected}")
            print(f"  - Varianza spiegata: {cumulative_variance[n_components_selected-1]*100:.2f}%")
            print(f"  - Riduzione dimensionalità: {original_features} → {n_components_selected}")
            
            # Seconda esecuzione: PCA con numero ottimale di componenti
            pca_optimal = PCA(n_components=n_components_selected)
            
            # Fit e transform con controllo degli output
            X_train_pca = pca_optimal.fit_transform(X_train_stable)
            X_val_pca = pca_optimal.transform(X_val_stable)
            
            # Controllo finale degli output PCA
            if np.any(np.isnan(X_train_pca)) or np.any(np.isinf(X_train_pca)):
                raise ValueError("PCA ha prodotto output con NaN o inf")
            
            if np.any(np.isnan(X_val_pca)) or np.any(np.isinf(X_val_pca)):
                raise ValueError("PCA ha prodotto output validation con NaN o inf")
            
            print(f"  - PCA applicato con successo")
            return X_train_pca, X_val_pca, n_components_selected
            
    except Exception as e:
        print(f"  - Errore PCA: {e}")
        print(f"  - Fallback: riduzione semplice alle prime {min(50, original_features)} feature")
        
        # Fallback: usa solo le prime N feature più stabili
        n_components_fallback = min(50, original_features)
        
        # Seleziona feature con varianza più alta (più stabili numericamente)
        feature_vars = np.var(X_train_stable, axis=0)
        # Sostituisci eventuali NaN nelle varianze con 0
        feature_vars = np.where(np.isnan(feature_vars), 0, feature_vars)
        
        top_features = np.argsort(feature_vars)[-n_components_fallback:]
        
        X_train_fallback = X_train_stable[:, top_features]
        X_val_fallback = X_val_stable[:, top_features]
        
        print(f"  - Fallback applicato: {n_components_fallback} feature selezionate")
        return X_train_fallback, X_val_fallback, n_components_fallback

def create_smartgrid_client_dnn_model(input_shape):
    """
    CORREZIONE PROBLEMA 2: Crea il modello DNN SmartGrid per il client.
    Architettura FISSA e IDENTICA al server per garantire compatibilità federata.
    
    Args:
        input_shape: Numero di feature in input (dopo PCA)
    
    Returns:
        Modello Keras compilato
    """
    print(f"=== CREAZIONE MODELLO DNN CLIENT ===")
    
    # Configurazione FISSA per compatibilità federata
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    # CORREZIONE PROBLEMA 2: Architettura IDENTICA al server (no Input layer esplicito)
    model = keras.Sequential([
        # Primo blocco - usa input_shape nel primo Dense invece di Input layer separato
        layers.Dense(128, 
                    activation='relu',
                    input_shape=(input_shape,),  # CORREZIONE: usa input_shape invece di Input layer
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Secondo blocco
        layers.Dense(64, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Terzo blocco
        layers.Dense(32, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        # Layer finale
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # Ottimizzatore IDENTICO al server
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
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
    
    # CORREZIONE PROBLEMA 2: Logging per verificare compatibilità
    print(f"  - Modello DNN client creato")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Architettura: Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    print(f"  - Numero di pesi: {len(model.get_weights())}")
    print(f"  - Parametri totali: {model.count_params():,}")
    
    # Debug: stampa info sui layer per verifica compatibilità
    print(f"  - Layer del modello:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            print(f"    - {i}: {layer.name} - {layer.units} unità")
        else:
            print(f"    - {i}: {layer.name}")
    
    print("=" * 60)
    
    return model

# Variabili globali per il client
client_id = None
model = None
X_train = None
y_train = None
X_val = None
y_val = None
dataset_info = None

class SmartGridDNNClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid che implementa l'addestramento federato
    per la rilevazione di intrusioni in smart grid con DNN e pipeline corretta.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello DNN locale.
        """
        weights = model.get_weights()
        # CORREZIONE PROBLEMA 2: Log per debugging compatibilità
        print(f"[Client {client_id}] Invio {len(weights)} pesi al server")
        return weights

    def fit(self, parameters, config):
        """
        Addestra il modello DNN sui dati locali del client con configurazione ottimizzata.
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO DNN ===")
        print(f"[Client {client_id}] Ricevuti {len(parameters)} pesi dal server")
        sys.stdout.flush()
        
        # CORREZIONE PROBLEMA 2: Verifica compatibilità pesi prima di impostarli
        current_weights = model.get_weights()
        if len(parameters) != len(current_weights):
            print(f"[Client {client_id}] ERRORE: Incompatibilità numero pesi!")
            print(f"[Client {client_id}] Ricevuti: {len(parameters)}, Attesi: {len(current_weights)}")
            
            # Debug dettagliato delle forme
            print(f"[Client {client_id}] Forme pesi ricevuti:")
            for i, w in enumerate(parameters):
                print(f"  {i}: {w.shape}")
            print(f"[Client {client_id}] Forme pesi modello:")
            for i, w in enumerate(current_weights):
                print(f"  {i}: {w.shape}")
                
            return model.get_weights(), 0, {'error': 'weight_shape_mismatch'}
        
        # Imposta i pesi ricevuti dal server
        try:
            model.set_weights(parameters)
            print(f"[Client {client_id}] Pesi impostati con successo")
        except Exception as e:
            print(f"[Client {client_id}] Errore nell'impostazione pesi: {e}")
            return model.get_weights(), 0, {'error': f'set_weights_failed: {str(e)}'}
        
        # Verifica che ci siano dati di training
        if len(X_train) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Configurazione addestramento locale
        local_epochs = 3
        batch_size = 16
        
        print(f"[Client {client_id}] Addestramento DNN su {len(X_train)} campioni per {local_epochs} epoche...")
        print(f"[Client {client_id}] Feature utilizzate: {X_train.shape[1]}")
        print(f"[Client {client_id}] Batch size: {batch_size}")
        
        try:
            history = model.fit(
                X_train, y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )
            
            # Estrai le metriche
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            train_precision = history.history.get('precision', [0])[-1]
            train_recall = history.history.get('recall', [0])[-1]
            train_auc = history.history.get('auc', [0])[-1]
            
            # Calcola F1-score
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            print(f"[Client {client_id}] Addestramento DNN completato:")
            print(f"[Client {client_id}]   - Loss: {train_loss:.4f}")
            print(f"[Client {client_id}]   - Accuracy: {train_accuracy:.4f}")
            print(f"[Client {client_id}]   - Precision: {train_precision:.4f}")
            print(f"[Client {client_id}]   - Recall: {train_recall:.4f}")
            print(f"[Client {client_id}]   - F1-Score: {train_f1:.4f}")
            print(f"[Client {client_id}]   - AUC: {train_auc:.4f}")
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante addestramento: {e}")
            return model.get_weights(), 0, {'error': f'training_failed: {str(e)}'}
        
        # Metriche da inviare al server
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            'train_auc': float(train_auc),
            'local_epochs': int(local_epochs),
            'batch_size': int(batch_size),
            'client_id': int(dataset_info['client_id']),
            'total_samples': int(dataset_info['total_samples']),
            'train_samples': int(dataset_info['train_samples']),
            'val_samples': int(dataset_info['val_samples']),
            'attack_samples': int(dataset_info['attack_samples']),
            'natural_samples': int(dataset_info['natural_samples']),
            'attack_ratio': float(dataset_info['attack_ratio']),
            'original_features': int(dataset_info['original_features']),
            'pca_features': int(dataset_info['pca_features']),
            'pca_reduction': float(dataset_info['pca_reduction']),
            'pipeline_applied': 'split_impute_scale_smote_stable_pca',
            'model_type': 'DNN_Stable_Compatible',
            'weights_count': len(model.get_weights())  # CORREZIONE PROBLEMA 2: traccia numero pesi
        }
        
        print(f"[Client {client_id}] Invio {len(model.get_weights())} pesi aggiornati al server...")
        sys.stdout.flush()
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello DNN sui dati di validation locali del client.
        """
        global model, X_val, y_val
        
        print(f"[Client {client_id}] === VALUTAZIONE LOCALE DNN ===")
        
        # CORREZIONE PROBLEMA 2: Verifica compatibilità pesi in valutazione
        current_weights = model.get_weights()
        if len(parameters) != len(current_weights):
            print(f"[Client {client_id}] ERRORE: Incompatibilità pesi in valutazione!")
            return 1.0, 0, {"accuracy": 0.0, "error": "weight_mismatch_eval"}
        
        # Imposta i pesi da valutare
        try:
            model.set_weights(parameters)
        except Exception as e:
            print(f"[Client {client_id}] Errore nell'impostazione pesi per valutazione: {e}")
            return 1.0, 0, {"accuracy": 0.0, "error": f"set_weights_eval_failed: {str(e)}"}
        
        # Verifica che ci siano dati di validation
        if len(X_val) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di validation disponibile!")
            return 0.0, 0, {"accuracy": 0.0}
        
        try:
            # Valuta sui dati di validation locali
            results = model.evaluate(X_val, y_val, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"[Client {client_id}] Valutazione DNN locale completata:")
            print(f"[Client {client_id}]   - Loss: {loss:.4f}")
            print(f"[Client {client_id}]   - Accuracy: {accuracy:.4f}")
            print(f"[Client {client_id}]   - Precision: {precision:.4f}")
            print(f"[Client {client_id}]   - Recall: {recall:.4f}")
            print(f"[Client {client_id}]   - F1-Score: {f1_score:.4f}")
            print(f"[Client {client_id}]   - AUC: {auc:.4f}")
            print(f"[Client {client_id}]   - Campioni validation: {len(X_val)}")
            print(f"[Client {client_id}]   - Feature utilizzate: {X_val.shape[1]}")
            
            # Metriche da restituire
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auc": auc,
                "val_samples": len(X_val)
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante valutazione: {e}")
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

def main():
    """
    Funzione principale per avviare il client SmartGrid DNN con correzioni per stabilità numerica e compatibilità.
    """
    global client_id, model, X_train, y_train, X_val, y_val, dataset_info
    
    # Verifica argomenti della riga di comando
    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        print("Esempio: python client.py 1")
        print("Client ID validi: 1-13 (per training federato)")
        print("Client 14-15 sono riservati per test globale del server")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("Client ID deve essere tra 1 e 13 per training federato")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"AVVIO CLIENT SMARTGRID DNN CON CORREZIONI {client_id}")
    print("Correzioni applicate:")
    print("  - Problema 1: Stabilità numerica PCA (pulizia dati, clipping, fallback)")
    print("  - Problema 2: Compatibilità architettura (modello identico server-client)")
    print("Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA Stabile")
    print("Modello: DNN compatibile (architettura fissa) con controlli pesi")
    print("=" * 80)
    
    try:
        # 1. Carica i dati locali del client con pipeline corretta e correzioni
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
        
        # 2. Crea il modello DNN compatibile
        print(f"[Client {client_id}] Creazione modello DNN compatibile...")
        model = create_smartgrid_client_dnn_model(X_train.shape[1])
        print(f"[Client {client_id}] Modello DNN creato con {X_train.shape[1]} feature di input")
        
        # 3. Stampa riassunto del client
        print(f"[Client {client_id}] === RIASSUNTO CLIENT DNN CON CORREZIONI ===")
        print(f"[Client {client_id}] Dataset info:")
        for key, value in dataset_info.items():
            if key == 'pca_reduction':
                print(f"[Client {client_id}]   - {key}: {value:.1f}%")
            else:
                print(f"[Client {client_id}]   - {key}: {value}")
        
        print(f"[Client {client_id}] Modello: DNN con architettura fissa per compatibilità")
        print(f"[Client {client_id}] Parametri totali: {model.count_params():,}")
        print(f"[Client {client_id}] Numero pesi: {len(model.get_weights())}")
        print(f"[Client {client_id}] Correzioni: Stabilità numerica + Compatibilità architettura")
        print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
        sys.stdout.flush()
        
        # 4. Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridDNNClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()