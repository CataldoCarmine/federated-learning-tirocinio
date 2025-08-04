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

def load_client_smartgrid_data(client_id, fixed_pca_components=50):
    """
    Carica i dati SmartGrid per un client specifico.
    CORREZIONE: Usa un numero FISSO di componenti PCA per tutti i client.
    
    Args:
        client_id: ID del client (1-15)
        fixed_pca_components: Numero fisso di componenti PCA per tutti i client
    
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
    
    # Gestione preliminare valori infiniti
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
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
    
    # Fit della pipeline SOLO sui dati di training del client
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_raw)
    X_val_preprocessed = preprocessing_pipeline.transform(X_val_raw)
    
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
            
            print(f"  - SMOTE applicato: {len(X_train_balanced)} campioni finali")
            print(f"  - Campioni sintetici: {len(X_train_balanced) - len(X_train_preprocessed)}")
            
        except Exception as e:
            print(f"  - Errore SMOTE: {e}, uso dati originali")
            X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
    else:
        print(f"  - SMOTE non necessario o non applicabile")
        X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
    
    # STEP 5: PCA con numero FISSO di componenti
    print(f"\n=== STEP 5: RIDUZIONE DIMENSIONALE CON PCA FISSO ===")
    
    original_features = X_train_balanced.shape[1]
    
    # CORREZIONE: Usa sempre lo stesso numero di componenti PCA per tutti i client
    n_components = min(fixed_pca_components, original_features, len(X_train_balanced))
    
    print(f"  - Feature originali: {original_features}")
    print(f"  - Componenti PCA FISSE: {n_components}")
    print(f"  - Riduzione: {(1-n_components/original_features)*100:.1f}%")
    
    # Applica PCA con numero fisso di componenti
    pca_fixed = PCA(n_components=n_components)
    X_train_final = pca_fixed.fit_transform(X_train_balanced)
    X_val_final = pca_fixed.transform(X_val_preprocessed)
    
    # Calcola varianza spiegata per informazione
    variance_explained = np.sum(pca_fixed.explained_variance_ratio_)
    print(f"  - Varianza spiegata con {n_components} componenti: {variance_explained*100:.2f}%")
    print(f"  - Training final shape: {X_train_final.shape}")
    print(f"  - Validation final shape: {X_val_final.shape}")
    
    # Verifica che tutte le forme siano corrette
    assert X_train_final.shape[1] == n_components, f"Errore: forma training {X_train_final.shape[1]} != {n_components}"
    assert X_val_final.shape[1] == n_components, f"Errore: forma validation {X_val_final.shape[1]} != {n_components}"
    
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
        'pca_reduction': (1 - n_components / original_features) * 100,
        'variance_explained': variance_explained
    }
    
    print("=" * 60)
    
    return X_train_final, y_train_balanced, X_val_final, y_val, dataset_info

def create_smartgrid_client_dnn_model(input_shape):
    """
    CORREZIONE ARCHITETTURA: Crea il modello DNN SmartGrid STANDARDIZZATO per il client.
    IMPORTANTE: Deve essere IDENTICO al modello del server per compatibilità federata.
    
    Args:
        input_shape: Numero di feature in input (FISSO per tutti i client)
    
    Returns:
        Modello Keras compilato
    """
    print(f"=== CREAZIONE MODELLO DNN CLIENT STANDARDIZZATO ===")
    
    # CORREZIONE: Configurazione IDENTICA server-client
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    # CORREZIONE: Verifica che input_shape sia corretto
    expected_input_shape = 50  # Numero fisso di componenti PCA
    if input_shape != expected_input_shape:
        print(f"ATTENZIONE: Input shape {input_shape} diverso da atteso {expected_input_shape}")
    
    # CORREZIONE: Architettura STANDARDIZZATA e IDENTICA con input_shape fisso
    model = keras.Sequential([
        # Layer di input esplicito con shape fisso
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        # CORREZIONE: Architettura semplificata e standardizzata
        # Primo blocco: 128 neuroni
        layers.Dense(128, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Secondo blocco: 64 neuroni
        layers.Dense(64, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Terzo blocco: 32 neuroni
        layers.Dense(32, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        # Layer finale: classificazione binaria
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # CORREZIONE: Ottimizzatore IDENTICO al server
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
    
    print(f"  - Modello DNN client standardizzato creato")
    print(f"  - Input shape FISSO: {input_shape}")
    print(f"  - Architettura FISSA: Input({input_shape}) → Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    print(f"  - Numero di pesi: {len(model.get_weights())}")
    print(f"  - Parametri totali: {model.count_params():,}")
    
    # CORREZIONE: Debug architettura per verifica compatibilità
    print(f"  - Layer del modello:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            print(f"    - {i}: {layer.name} ({layer.__class__.__name__}) - {layer.units} unità")
        else:
            print(f"    - {i}: {layer.name} ({layer.__class__.__name__})")
    
    # CORREZIONE: Debug forme pesi per verifica compatibilità
    print(f"  - Forme pesi del modello:")
    for i, weight in enumerate(model.get_weights()):
        print(f"    - Peso {i}: {weight.shape}")
    
    print("=" * 60)
    
    return model

def safe_extract_parameters(parameters):
    """
    CORREZIONE GESTIONE PARAMETERS: Estrae i pesi dal tipo Parameters di Flower in modo sicuro.
    Gestisce diversi tipi di tensori e conversioni per evitare errori di tipo.
    
    Args:
        parameters: Oggetto Parameters, lista di array numpy, o lista di tensori
    
    Returns:
        Lista di array numpy
    """
    try:
        # CASO 1: Se è già una lista di array numpy
        if isinstance(parameters, list):
            weights_list = []
            for tensor in parameters:
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    # Se è un tensor TensorFlow
                    weights_list.append(tensor.numpy())
                else:
                    # Prova conversione diretta
                    weights_list.append(np.array(tensor, dtype=np.float32))
            return weights_list
        
        # CASO 2: Se è un oggetto Parameters di Flower
        elif hasattr(parameters, 'tensors'):
            weights_list = []
            for tensor in parameters.tensors:
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    # Se è un tensor TensorFlow
                    weights_list.append(tensor.numpy())
                else:
                    # Prova conversione diretta
                    weights_list.append(np.array(tensor, dtype=np.float32))
            return weights_list
        
        # CASO 3: Fallback - prova conversione diretta
        else:
            if hasattr(parameters, 'numpy'):
                return [parameters.numpy()]
            else:
                return [np.array(parameters, dtype=np.float32)]
            
    except Exception as e:
        print(f"Errore nell'estrazione parametri: {e}")
        # Ultimo fallback: ritorna i parametri come sono
        return parameters

def check_parameters_compatibility(received_params, model_weights, client_id):
    """
    CORREZIONE GESTIONE PARAMETERS: Verifica la compatibilità tra parametri ricevuti e modello.
    Include controlli di tipo, numero e forme dei pesi.
    
    Args:
        received_params: Parametri ricevuti (possono essere di diversi tipi)
        model_weights: Pesi del modello locale
        client_id: ID del client per logging
    
    Returns:
        Tuple (is_compatible, extracted_weights, error_message)
    """
    try:
        # STEP 1: Estrai i pesi in modo sicuro
        extracted_weights = safe_extract_parameters(received_params)
        
        # STEP 2: Verifica che sia una lista
        if not isinstance(extracted_weights, list):
            return False, None, f"Parametri estratti non sono una lista: {type(extracted_weights)}"
        
        # STEP 3: Verifica numero di pesi
        if len(extracted_weights) != len(model_weights):
            error_msg = f"Numero pesi incompatibile: ricevuti {len(extracted_weights)}, attesi {len(model_weights)}"
            return False, None, error_msg
        
        # STEP 4: Verifica forme dei pesi
        for i, (received_weight, model_weight) in enumerate(zip(extracted_weights, model_weights)):
            # Converti a numpy se necessario
            if not isinstance(received_weight, np.ndarray):
                try:
                    received_weight = np.array(received_weight, dtype=np.float32)
                    extracted_weights[i] = received_weight
                except Exception as e:
                    return False, None, f"Impossibile convertire peso {i} a numpy array: {e}"
            
            # Verifica forma
            if received_weight.shape != model_weight.shape:
                error_msg = f"Forma peso {i} incompatibile: ricevuta {received_weight.shape}, attesa {model_weight.shape}"
                return False, None, error_msg
        
        # STEP 5: Tutto OK
        print(f"[Client {client_id}] ✅ Compatibilità parametri verificata:")
        print(f"[Client {client_id}]   - Numero pesi: {len(extracted_weights)}")
        print(f"[Client {client_id}]   - Forme verificate: {len(extracted_weights)} pesi")
        
        return True, extracted_weights, None
        
    except Exception as e:
        error_msg = f"Errore durante verifica compatibilità: {str(e)}"
        return False, None, error_msg

def safe_set_model_weights(model, parameters, client_id):
    """
    CORREZIONE GESTIONE PARAMETERS: Imposta i pesi del modello in modo sicuro.
    Gestisce diversi tipi di parametri e include controlli di validazione.
    
    Args:
        model: Modello Keras
        parameters: Parametri da impostare (diversi tipi possibili)
        client_id: ID del client per logging
    
    Returns:
        Tuple (success, error_message)
    """
    try:
        # STEP 1: Ottieni pesi attuali del modello
        current_weights = model.get_weights()
        
        # STEP 2: Verifica compatibilità e estrai pesi
        is_compatible, extracted_weights, error_msg = check_parameters_compatibility(
            parameters, current_weights, client_id
        )
        
        if not is_compatible:
            return False, error_msg
        
        # STEP 3: Imposta i pesi estratti
        model.set_weights(extracted_weights)
        
        print(f"[Client {client_id}] ✅ Pesi impostati con successo")
        return True, None
        
    except Exception as e:
        error_msg = f"Errore durante impostazione pesi: {str(e)}"
        print(f"[Client {client_id}] ❌ {error_msg}")
        return False, error_msg

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
    per la rilevazione di intrusioni in smart grid con DNN e gestione robusta dei Parameters.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello DNN locale.
        CORREZIONE: Gestione robusta del tipo di ritorno.
        """
        weights = model.get_weights()
        
        # CORREZIONE GESTIONE PARAMETERS: Log dettagliato per debug
        print(f"[Client {client_id}] === INVIO PARAMETRI AL SERVER ===")
        print(f"[Client {client_id}] Numero pesi da inviare: {len(weights)}")
        
        # Debug forme pesi per verifica
        print(f"[Client {client_id}] Forme pesi da inviare:")
        for i, weight in enumerate(weights):
            print(f"[Client {client_id}]   - Peso {i}: {weight.shape} (tipo: {type(weight)})")
        
        # CORREZIONE: Assicura che tutti i pesi siano numpy arrays
        processed_weights = []
        for i, weight in enumerate(weights):
            if isinstance(weight, np.ndarray):
                processed_weights.append(weight)
            elif hasattr(weight, 'numpy'):
                processed_weights.append(weight.numpy())
            else:
                processed_weights.append(np.array(weight, dtype=np.float32))
        
        print(f"[Client {client_id}] ✅ Parametri processati e pronti per l'invio")
        return processed_weights

    def fit(self, parameters, config):
        """
        Addestra il modello DNN sui dati locali del client con gestione robusta dei Parameters.
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO DNN CON GESTIONE ROBUSTA PARAMETERS ===")
        
        # CORREZIONE GESTIONE PARAMETERS: Log dettagliato tipo parametri ricevuti
        print(f"[Client {client_id}] Tipo parametri ricevuti: {type(parameters)}")
        if hasattr(parameters, '__len__'):
            print(f"[Client {client_id}] Numero elementi: {len(parameters)}")
        if hasattr(parameters, 'tensors'):
            print(f"[Client {client_id}] Parametri Flower con {len(parameters.tensors)} tensori")
        
        sys.stdout.flush()
        
        # CORREZIONE GESTIONE PARAMETERS: Usa funzione sicura per impostare pesi
        success, error_msg = safe_set_model_weights(model, parameters, client_id)
        
        if not success:
            print(f"[Client {client_id}] ❌ Errore nell'impostazione parametri: {error_msg}")
            return model.get_weights(), 0, {'error': f'parameter_handling_failed: {error_msg}'}
        
        # Verifica che ci siano dati di training
        if len(X_train) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Configurazione addestramento locale
        local_epochs = 3
        batch_size = 16
        
        print(f"[Client {client_id}] === CONFIGURAZIONE ADDESTRAMENTO ===")
        print(f"[Client {client_id}] Campioni training: {len(X_train)}")
        print(f"[Client {client_id}] Feature utilizzate (FISSE): {X_train.shape[1]}")
        print(f"[Client {client_id}] Epoche locali: {local_epochs}")
        print(f"[Client {client_id}] Batch size: {batch_size}")
        
        try:
            print(f"[Client {client_id}] Avvio addestramento DNN...")
            history = model.fit(
                X_train, y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )
            
            # Estrai le metriche dell'addestramento (ultima epoca)
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            train_precision = history.history.get('precision', [0])[-1]
            train_recall = history.history.get('recall', [0])[-1]
            train_auc = history.history.get('auc', [0])[-1]
            
            # Calcola F1-score
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            print(f"[Client {client_id}] === RISULTATI ADDESTRAMENTO DNN ===")
            print(f"[Client {client_id}]   - Loss: {train_loss:.4f}")
            print(f"[Client {client_id}]   - Accuracy: {train_accuracy:.4f}")
            print(f"[Client {client_id}]   - Precision: {train_precision:.4f}")
            print(f"[Client {client_id}]   - Recall: {train_recall:.4f}")
            print(f"[Client {client_id}]   - F1-Score: {train_f1:.4f}")
            print(f"[Client {client_id}]   - AUC: {train_auc:.4f}")
            
            # Controlli di qualità
            if train_loss > 5.0:
                print(f"[Client {client_id}] ⚠️  ATTENZIONE: Loss molto alta!")
            elif train_loss < 1.0:
                print(f"[Client {client_id}] ✅  Loss accettabile")
            
            if train_accuracy < 0.5:
                print(f"[Client {client_id}] ⚠️  ATTENZIONE: Accuracy sotto random!")
            elif train_accuracy > 0.7:
                print(f"[Client {client_id}] ✅  Accuracy buona")
            
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore durante addestramento: {e}")
            return model.get_weights(), 0, {'error': f'training_failed: {str(e)}'}
        
        # CORREZIONE GESTIONE PARAMETERS: Metriche estese da inviare al server
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
            'variance_explained': float(dataset_info['variance_explained']),
            'pipeline_applied': 'split_impute_scale_smote_pca_fixed',
            'model_type': 'DNN_Standardized_Fixed_PCA_Parameters_Robust',
            'weights_count': len(model.get_weights()),
            'parameter_handling': 'robust_safe_extraction'  # NUOVO: indica gestione robusta
        }
        
        print(f"[Client {client_id}] === INVIO RISULTATI AL SERVER ===")
        print(f"[Client {client_id}] Pesi aggiornati: {len(model.get_weights())}")
        print(f"[Client {client_id}] Metriche: {len(metrics)} elementi")
        sys.stdout.flush()
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello DNN sui dati di validation locali del client con gestione robusta dei Parameters.
        """
        global model, X_val, y_val
        
        print(f"[Client {client_id}] === VALUTAZIONE LOCALE DNN CON GESTIONE ROBUSTA PARAMETERS ===")
        
        # CORREZIONE GESTIONE PARAMETERS: Usa funzione sicura per impostare pesi
        success, error_msg = safe_set_model_weights(model, parameters, client_id)
        
        if not success:
            print(f"[Client {client_id}] ❌ Errore nell'impostazione parametri per valutazione: {error_msg}")
            return 1.0, 0, {"accuracy": 0.0, "error": f"parameter_handling_eval_failed: {error_msg}"}
        
        # Verifica che ci siano dati di validation
        if len(X_val) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di validation disponibile!")
            return 0.0, 0, {"accuracy": 0.0}
        
        try:
            print(f"[Client {client_id}] Avvio valutazione su {len(X_val)} campioni...")
            
            # Valuta sui dati di validation locali
            results = model.evaluate(X_val, y_val, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"[Client {client_id}] === RISULTATI VALUTAZIONE DNN LOCALE ===")
            print(f"[Client {client_id}]   - Loss: {loss:.4f}")
            print(f"[Client {client_id}]   - Accuracy: {accuracy:.4f}")
            print(f"[Client {client_id}]   - Precision: {precision:.4f}")
            print(f"[Client {client_id}]   - Recall: {recall:.4f}")
            print(f"[Client {client_id}]   - F1-Score: {f1_score:.4f}")
            print(f"[Client {client_id}]   - AUC: {auc:.4f}")
            print(f"[Client {client_id}]   - Campioni validation: {len(X_val)}")
            print(f"[Client {client_id}]   - Feature utilizzate (FISSE): {X_val.shape[1]}")
            
            # Controlli di qualità
            if loss > 2.0:
                print(f"[Client {client_id}] ⚠️  ATTENZIONE: Validation loss alta!")
            elif loss < 1.0:
                print(f"[Client {client_id}] ✅  Validation loss accettabile")
            
            if accuracy < 0.6:
                print(f"[Client {client_id}] ⚠️  ATTENZIONE: Validation accuracy bassa!")
            elif accuracy > 0.7:
                print(f"[Client {client_id}] ✅  Validation accuracy buona")
            
            # Metriche da restituire
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auc": auc,
                "val_samples": len(X_val),
                "parameter_handling": "robust_safe_extraction"  # NUOVO: indica gestione robusta
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore durante valutazione: {e}")
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

def main():
    """
    Funzione principale per avviare il client SmartGrid DNN con gestione robusta dei Parameters.
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
    
    print(f"AVVIO CLIENT SMARTGRID DNN CON GESTIONE ROBUSTA PARAMETERS {client_id}")
    print("CORREZIONI APPLICATE:")
    print("  - PCA FISSO: 50 componenti per tutti i client (risolve errore broadcast)")
    print("  - Architettura standardizzata: Input(50) → Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    print("  - Compatibilità server-client: modelli identici per evitare errore pesi")
    print("  - Input layer esplicito: evita warning Keras")
    print("  - GESTIONE ROBUSTA PARAMETERS: extraction sicura, controlli tipo, fallback")
    print("  - Debug pesi: logging dettagliato per troubleshooting")
    print("  - Verifiche forme: controlli compatibilità forme pesi")
    print("  - Controlli qualità: monitoraggio automatico performance")
    print("Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA(50) FISSO")
    print("Configurazione: train (70%) + validation locale (30%)")
    print("=" * 90)
    
    try:
        # 1. Carica i dati locali del client con PCA fisso
        fixed_pca_components = 50  # CORREZIONE: Numero fisso di componenti PCA
        print(f"[Client {client_id}] Caricamento dati con PCA fisso a {fixed_pca_components} componenti...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id, fixed_pca_components)
        
        # 2. CORREZIONE: Verifica che la forma sia quella attesa
        expected_shape = fixed_pca_components
        if X_train.shape[1] != expected_shape:
            raise ValueError(f"Errore: forma training {X_train.shape[1]} != {expected_shape}")
        
        # 3. Crea il modello DNN standardizzato con input fisso
        print(f"[Client {client_id}] Creazione modello DNN standardizzato con gestione robusta Parameters...")
        model = create_smartgrid_client_dnn_model(X_train.shape[1])
        print(f"[Client {client_id}] ✅ Modello DNN standardizzato creato con {X_train.shape[1]} feature di input FISSE")
        
        # 4. Stampa riassunto del client
        print(f"[Client {client_id}] === RIASSUNTO CLIENT DNN CON GESTIONE ROBUSTA PARAMETERS ===")
        print(f"[Client {client_id}] Dataset info:")
        for key, value in dataset_info.items():
            if key in ['pca_reduction', 'variance_explained']:
                print(f"[Client {client_id}]   - {key}: {value:.1f}%")
            else:
                print(f"[Client {client_id}]   - {key}: {value}")
        
        print(f"[Client {client_id}] Modello: DNN standardizzata per compatibilità federata")
        print(f"[Client {client_id}] Parametri totali: {model.count_params():,}")
        print(f"[Client {client_id}] Numero pesi: {len(model.get_weights())}")
        print(f"[Client {client_id}] Input shape FISSO: {X_train.shape[1]} (garantisce compatibilità)")
        print(f"[Client {client_id}] Architettura: compatibile con server federato")
        print(f"[Client {client_id}] Gestione Parameters: robusta con extraction sicura e controlli tipo")
        print(f"[Client {client_id}] Funzionalità aggiunte:")
        print(f"[Client {client_id}]   - safe_extract_parameters(): gestione sicura diversi tipi Parameters")
        print(f"[Client {client_id}]   - check_parameters_compatibility(): verifica compatibilità completa")
        print(f"[Client {client_id}]   - safe_set_model_weights(): impostazione sicura pesi del modello")
        print(f"[Client {client_id}]   - Controlli qualità automatici in fit() e evaluate()")
        print(f"[Client {client_id}]   - Logging esteso per debugging avanzato")
        print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
        sys.stdout.flush()
        
        # 5. Avvia il client Flower con gestione robusta
        print(f"[Client {client_id}] Avvio client Flower con gestione robusta Parameters...")
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridDNNClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()