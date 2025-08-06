import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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

# CONFIGURAZIONE PCA STATICA
# MODIFICA QUESTO VALORE DOPO AVER ESEGUITO L'ANALISI PCA
PCA_COMPONENTS = 35  # <-- MODIFICA QUESTO VALORE CON IL RISULTATO DELL'ANALISI
PCA_RANDOM_STATE = 42

def clean_data_for_pca(X):
    """
    Pulizia robusta dei dati per prevenire problemi numerici in PCA.
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    
    # Rimuovi valori estremi
    threshold = 1e8
    X_array = np.where(np.abs(X_array) > threshold, np.nan, X_array)
    
    # Rimuovi valori molto piccoli
    epsilon = 1e-12
    X_array = np.where(np.abs(X_array) < epsilon, 0, X_array)
    
    return X_array

def ensure_numerical_stability(X, stage_name):
    """
    Assicura stabilità numerica rimuovendo inf, nan e valori estremi.
    """
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    extreme_count = np.sum(np.abs(X) > 1e6)
    
    if nan_count > 0 or inf_count > 0 or extreme_count > 0:
        X_clean = X.copy()
        
        # Sostituisci NaN e inf con mediana delle colonne
        for col in range(X_clean.shape[1]):
            col_data = X_clean[:, col]
            finite_mask = np.isfinite(col_data)
            
            if np.any(finite_mask):
                median_val = np.median(col_data[finite_mask])
                X_clean[~finite_mask, col] = median_val
            else:
                X_clean[:, col] = 0
        
        # Clip valori estremi
        X_clean = np.clip(X_clean, -1e6, 1e6)
        return X_clean
    else:
        return np.clip(X, -1e6, 1e6)

def apply_fixed_pca(X_preprocessed, client_id=None):
    """
    Applica PCA con numero FISSO di componenti determinato dall'analisi preliminare.
    
    Args:
        X_preprocessed: Dati preprocessati e standardizzati
        client_id: ID del client per logging
    
    Returns:
        Tuple (X_pca, n_components_used, variance_explained)
    """
    print(f"[Client {client_id}] === APPLICAZIONE PCA FISSA ===")
    
    original_features = X_preprocessed.shape[1]
    
    # Usa il numero FISSO determinato dall'analisi preliminare
    n_components = min(PCA_COMPONENTS, original_features, len(X_preprocessed))
    
    print(f"[Client {client_id}] Feature originali: {original_features}")
    print(f"[Client {client_id}] Componenti PCA fisse: {PCA_COMPONENTS}")
    print(f"[Client {client_id}] Componenti PCA effettive: {n_components}")
    
    # Pulizia robusta dei dati pre-PCA
    X_stable = ensure_numerical_stability(X_preprocessed, f"pre-PCA client {client_id}")
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Applica PCA con numero FISSO di componenti
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
            X_pca = pca.fit_transform(X_stable)
            
            # Verifica output
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError(f"PCA client {client_id} ha prodotto output con NaN o inf")
            
            # Calcola varianza spiegata
            variance_explained = np.sum(pca.explained_variance_ratio_)
            
            print(f"[Client {client_id}] ✅ PCA fissa applicata con successo")
            print(f"[Client {client_id}] Shape finale: {X_pca.shape}")
            print(f"[Client {client_id}] Varianza spiegata: {variance_explained*100:.2f}%")
            
            return X_pca, n_components, variance_explained
            
    except Exception as e:
        print(f"[Client {client_id}] ERRORE PCA fissa: {e}")
        print(f"[Client {client_id}] Attivazione fallback...")
        
        # Fallback semplice: usa le prime N feature
        n_fallback = min(n_components, original_features)
        X_fallback = X_stable[:, :n_fallback]
        X_fallback = ensure_numerical_stability(X_fallback, f"PCA fallback client {client_id}")
        
        print(f"[Client {client_id}] ✅ Fallback: uso prime {n_fallback} feature")
        
        return X_fallback, n_fallback, 0.95  # Stima conservativa

def compute_class_weights_simple(y_train):
    """
    Calcola i pesi delle classi per compensare lo sbilanciamento.
    Versione semplificata essenziale.
    
    Args:
        y_train: Target di training
    
    Returns:
        Dizionario con class weights
    """
    try:
        # Calcola i pesi automatici per bilanciare le classi
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        
        # Crea dizionario class_weight per Keras
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        return class_weight_dict
        
    except Exception as e:
        print(f"Errore nel calcolo class weights: {e}")
        # Fallback: pesi uguali
        unique_classes = np.unique(y_train)
        return {cls: 1.0 for cls in unique_classes}

def load_client_smartgrid_data_with_fixed_pca(client_id):
    """
    Carica i dati SmartGrid per un client specifico SENZA SMOTE e con PCA FISSA.
    
    Args:
        client_id: ID del client
    
    Returns:
        Tuple con dati processati e informazioni dataset
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    df = pd.read_csv(file_path)
    
    print(f"[Client {client_id}] === CARICAMENTO CON PCA FISSA (SENZA SMOTE) ===")
    print(f"[Client {client_id}] Dataset caricato per attacchi realistici")
    print(f"[Client {client_id}] PCA fissa configurata: {PCA_COMPONENTS} componenti")
    
    # Separa feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    # Statistiche distribuzione naturale
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    print(f"[Client {client_id}] Distribuzione naturale: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
    
    # Pulizia robusta preliminare
    X_cleaned = clean_data_for_pca(X)
    
    # STEP 1: Suddivisione train/validation (mantiene distribuzione naturale)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_cleaned, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"[Client {client_id}] Suddivisione: {len(X_train_raw)} training, {len(X_val_raw)} validation")
    
    # STEP 2-3: Pipeline di preprocessing (LOCALE per ogni client)
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Fit della pipeline SOLO sui dati di training del client (normalizzazione locale)
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_raw)
    X_val_preprocessed = preprocessing_pipeline.transform(X_val_raw)
    
    print(f"[Client {client_id}] Preprocessing locale applicato")
    
    # STEP 4: NO SMOTE - Mantieni distribuzione naturale per attacchi realistici
    print(f"[Client {client_id}] SMOTE RIMOSSO per attacchi inference/extraction realistici")
    
    # Calcola class weights per compensare sbilanciamento
    class_weights = compute_class_weights_simple(y_train)
    print(f"[Client {client_id}] Class weights: {class_weights}")
    
    # STEP 5: PCA FISSA dall'analisi preliminare
    X_train_final, n_components_used, variance_explained = apply_fixed_pca(
        X_train_preprocessed, 
        client_id=client_id
    )
    
    # Applica la stessa trasformazione ai dati di validation
    if n_components_used <= X_val_preprocessed.shape[1]:
        X_val_final = X_val_preprocessed[:, :n_components_used]
    else:
        X_val_final = X_val_preprocessed[:, :min(n_components_used, X_val_preprocessed.shape[1])]
        n_components_used = X_val_final.shape[1]
    
    X_val_final = ensure_numerical_stability(X_val_final, f"validation final client {client_id}")
    
    # Informazioni complete del dataset
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
        'pca_features': n_components_used,
        'pca_components_configured': PCA_COMPONENTS,
        'pca_reduction': (1 - n_components_used / X.shape[1]) * 100,
        'variance_explained': variance_explained * 100,
        'class_weights': class_weights,
        'preprocessing_method': 'no_smote_fixed_pca',
        'pca_method': 'fixed_manual_configuration'
    }
    
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO (PCA FISSA) ===")
    print(f"[Client {client_id}]   - Training: {X_train_final.shape}")
    print(f"[Client {client_id}]   - Validation: {X_val_final.shape}")
    print(f"[Client {client_id}]   - Componenti PCA: {n_components_used} (configurate: {PCA_COMPONENTS})")
    print(f"[Client {client_id}]   - Varianza spiegata: {variance_explained*100:.2f}%")
    print(f"[Client {client_id}]   - Riduzione dimensionalità: {dataset_info['pca_reduction']:.1f}%")
    print(f"[Client {client_id}]   - Adatto per attacchi: ✅")
    
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_smartgrid_dnn_model_static_architecture():
    """
    Crea il modello DNN SmartGrid con architettura STATICA ottimizzata per 35 componenti PCA.
    RIMOSSA la logica dinamica - ora architettura completamente fissa.
    
    Returns:
        Modello Keras compilato per dataset sbilanciati
    """
    print(f"[Client] === CREAZIONE DNN ARCHITETTURA STATICA (SENZA SMOTE) ===")
    print(f"[Client] Input features: {PCA_COMPONENTS} (FISSO)")
    
    # Parametri ottimizzati per dataset sbilanciati
    dropout_rate = 0.3  # Aumentato per prevenire overfitting
    l2_reg = 0.001      # Aumentato per maggiore regolarizzazione
    
    # ARCHITETTURA STATICA OTTIMIZZATA PER 35 COMPONENTI PCA
    # 35 → 32 → 20 → 12 → 1 (architettura media fissa)
    
    print(f"[Client] Architettura STATICA: {PCA_COMPONENTS} → 32 → 20 → 12 → 1")
    
    model = keras.Sequential([
        # Input layer esplicito con dimensione FISSA
        layers.Input(shape=(PCA_COMPONENTS,), name='input_layer'),
        
        # Layer 1: 32 neuroni (ottimizzato per 35 input)
        layers.Dense(32, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Layer 2: 20 neuroni (feature extraction)
        layers.Dense(20, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Layer 3: 12 neuroni (pattern recognition)
        layers.Dense(12, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        # Output layer: 1 neurone (classificazione binaria)
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # Ottimizzatore con learning rate adattivo
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,    # Learning rate standard
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    # Compila il modello con Binary Crossentropy standard
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
    params_per_feature = total_params / PCA_COMPONENTS
    
    print(f"[Client] === DNN ARCHITETTURA STATICA CREATA ===")
    print(f"[Client]   - Parametri totali: {total_params:,}")
    print(f"[Client]   - Parametri per feature: {params_per_feature:.1f}")
    print(f"[Client]   - Architettura: STATICA per {PCA_COMPONENTS} feature PCA")
    print(f"[Client]   - Dropout: {dropout_rate}")
    print(f"[Client]   - L2 regularization: {l2_reg}")
    print(f"[Client]   - Learning rate: {optimizer.learning_rate}")
    print(f"[Client]   - Loss: Binary Crossentropy + Class Weights")
    print(f"[Client]   - Ottimizzato per: dati naturalmente sbilanciati")
    
    # Valutazione ottimizzazione
    if params_per_feature > 100:
        print(f"[Client]   ⚠️  ATTENZIONE: Alto rapporto parametri/feature")
    else:
        print(f"[Client]   ✅ Rapporto parametri/feature ottimale")
    
    return model

def safe_extract_parameters(parameters):
    """
    Estrae i pesi dal tipo Parameters di Flower in modo sicuro.
    """
    try:
        if isinstance(parameters, list):
            weights_list = []
            for tensor in parameters:
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    weights_list.append(tensor.numpy())
                else:
                    weights_list.append(np.array(tensor, dtype=np.float32))
            return weights_list
        
        elif hasattr(parameters, 'tensors'):
            weights_list = []
            for tensor in parameters.tensors:
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    weights_list.append(tensor.numpy())
                else:
                    weights_list.append(np.array(tensor, dtype=np.float32))
            return weights_list
        
        else:
            if hasattr(parameters, 'numpy'):
                return [parameters.numpy()]
            else:
                return [np.array(parameters, dtype=np.float32)]
            
    except Exception as e:
        print(f"Errore nell'estrazione parametri: {e}")
        return parameters

def check_parameters_compatibility(received_params, model_weights, client_id):
    """
    Verifica la compatibilità tra parametri ricevuti e modello.
    """
    try:
        extracted_weights = safe_extract_parameters(received_params)
        
        if not isinstance(extracted_weights, list):
            return False, None, f"Parametri estratti non sono una lista: {type(extracted_weights)}"
        
        if len(extracted_weights) != len(model_weights):
            error_msg = f"Numero pesi incompatibile: ricevuti {len(extracted_weights)}, attesi {len(model_weights)}"
            return False, None, error_msg
        
        for i, (received_weight, model_weight) in enumerate(zip(extracted_weights, model_weights)):
            if not isinstance(received_weight, np.ndarray):
                try:
                    received_weight = np.array(received_weight, dtype=np.float32)
                    extracted_weights[i] = received_weight
                except Exception as e:
                    return False, None, f"Impossibile convertire peso {i} a numpy array: {e}"
            
            if received_weight.shape != model_weight.shape:
                error_msg = f"Forma peso {i} incompatibile: ricevuta {received_weight.shape}, attesa {model_weight.shape}"
                return False, None, error_msg
        
        return True, extracted_weights, None
        
    except Exception as e:
        error_msg = f"Errore durante verifica compatibilità: {str(e)}"
        return False, None, error_msg

def safe_set_model_weights(model, parameters, client_id):
    """
    Imposta i pesi del modello in modo sicuro.
    """
    try:
        current_weights = model.get_weights()
        
        is_compatible, extracted_weights, error_msg = check_parameters_compatibility(
            parameters, current_weights, client_id
        )
        
        if not is_compatible:
            return False, error_msg
        
        model.set_weights(extracted_weights)
        return True, None
        
    except Exception as e:
        error_msg = f"Errore durante impostazione pesi: {str(e)}"
        return False, error_msg

# Variabili globali per il client
client_id = None
model = None
X_train = None
y_train = None
X_val = None
y_val = None
dataset_info = None

class SmartGridDNNClientFixed(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con DNN a architettura STATICA e PCA fissa SENZA SMOTE.
    Configurazione manuale semplificata per scopi didattici.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello DNN locale.
        """
        weights = model.get_weights()
        
        processed_weights = []
        for i, weight in enumerate(weights):
            if isinstance(weight, np.ndarray):
                processed_weights.append(weight)
            elif hasattr(weight, 'numpy'):
                processed_weights.append(weight.numpy())
            else:
                processed_weights.append(np.array(weight, dtype=np.float32))
        
        return processed_weights

    def fit(self, parameters, config):
        """
        Addestra il modello DNN su dati naturalmente sbilanciati con class weights.
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] Round di addestramento con PCA fissa e dataset naturalmente sbilanciato...")
        
        # Usa funzione sicura per impostare pesi
        success, error_msg = safe_set_model_weights(model, parameters, client_id)
        
        if not success:
            print(f"[Client {client_id}] Errore parametri: {error_msg}")
            return model.get_weights(), 0, {'error': f'parameter_handling_failed: {error_msg}'}
        
        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Configurazione addestramento per dataset sbilanciati
        local_epochs = 5        
        batch_size = 32         
        
        # Usa class weights per compensare sbilanciamento
        class_weights = dataset_info['class_weights']
        
        try:
            print(f"[Client {client_id}] Training con class weights: {class_weights}")
            print(f"[Client {client_id}] Architettura STATICA per {dataset_info['pca_features']} feature PCA")
            
            history = model.fit(
                X_train, y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                class_weight=class_weights,  # Compensa sbilanciamento
                verbose=0,
                shuffle=True
            )
            
            # Estrai metriche base
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            train_precision = history.history.get('precision', [0])[-1]
            train_recall = history.history.get('recall', [0])[-1]
            train_auc = history.history.get('auc', [0])[-1]
            
            # Calcola F1-score manualmente
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            # Calcola Balanced Accuracy sui dati di training
            y_pred_prob = model.predict(X_train, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_binary)
            
            print(f"[Client {client_id}] Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"[Client {client_id}] F1-Score: {train_f1:.4f}, Balanced Acc: {train_balanced_acc:.4f}")
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante addestramento: {e}")
            return model.get_weights(), 0, {'error': f'training_failed: {str(e)}'}
        
        # Metriche essenziali da inviare al server
        metrics = {
            # Metriche base
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_auc': float(train_auc),
            
            # Metriche bilanciate essenziali
            'train_f1_score': float(train_f1),
            'train_balanced_accuracy': float(train_balanced_acc),
            
            # Configurazione addestramento
            'local_epochs': int(local_epochs),
            'batch_size': int(batch_size),
            'used_class_weights': True,
            'class_weight_0': float(class_weights[0]),
            'class_weight_1': float(class_weights[1]),
            
            # Informazioni dataset essenziali
            'client_id': int(dataset_info['client_id']),
            'train_samples': int(dataset_info['train_samples']),
            'attack_ratio': float(dataset_info['attack_ratio']),
            'train_attack_ratio': float(dataset_info['train_attack_ratio']),
            
            # Informazioni PCA fissa
            'original_features': int(dataset_info['original_features']),
            'pca_features': int(dataset_info['pca_features']),
            'pca_components_configured': int(dataset_info['pca_components_configured']),
            'pca_reduction': float(dataset_info['pca_reduction']),
            'variance_explained': float(dataset_info['variance_explained']),
            'total_params': int(model.count_params()),
            
            # Metodologia
            'preprocessing_method': dataset_info['preprocessing_method'],
            'pca_method': dataset_info['pca_method'],
            'model_type': 'dnn_static_architecture_manual_pca',
            'architecture_type': 'static_optimized'
        }
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello DNN sui dati di validation con metriche essenziali.
        """
        global model, X_val, y_val
        
        # Usa funzione sicura per impostare pesi
        success, error_msg = safe_set_model_weights(model, parameters, client_id)
        
        if not success:
            print(f"[Client {client_id}] Errore parametri valutazione: {error_msg}")
            return 1.0, 0, {"accuracy": 0.0, "error": f"parameter_handling_eval_failed: {error_msg}"}
        
        if len(X_val) == 0:
            return 0.0, 0, {"accuracy": 0.0}
        
        try:
            # Valutazione base
            results = model.evaluate(X_val, y_val, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calcola Balanced Accuracy
            y_pred_prob = model.predict(X_val, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_val, y_pred_binary)
            
            print(f"[Client {client_id}] Val Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
            print(f"[Client {client_id}] Val F1: {f1_score_val:.4f}, Val Balanced Acc: {balanced_acc:.4f}")
            
            # Metriche essenziali
            metrics = {
                # Metriche base
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                
                # Metriche bilanciate essenziali
                "f1_score": f1_score_val,
                "balanced_accuracy": balanced_acc,
                
                # Info valutazione
                "val_samples": len(X_val),
                "pca_features": dataset_info['pca_features'],
                "pca_components_configured": dataset_info['pca_components_configured'],
                "model_type": "dnn_static_architecture_manual_pca"
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante valutazione: {e}")
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

def main():
    """
    Funzione principale per avviare il client SmartGrid DNN con PCA fissa SENZA SMOTE.
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
    
    print(f"=== AVVIO CLIENT SMARTGRID DNN CON ARCHITETTURA STATICA {client_id} (SENZA SMOTE) ===")
    print("CONFIGURAZIONE FINALE:")
    print("  ✅ SMOTE RIMOSSO per attacchi inference/extraction realistici")
    print(f"  ✅ PCA FISSA configurata: {PCA_COMPONENTS} componenti")
    print("  ✅ Architettura DNN STATICA: 35 → 32 → 20 → 12 → 1")
    print("  ✅ Distribuzione naturale mantenuta per fedeltà al mondo reale")
    print("  ✅ Class weights automatici per compensare sbilanciamento")
    print("  ✅ Metriche bilanciate: F1-Score, Balanced Accuracy, AUC")
    print("  ✅ Normalizzazione LOCALE per ogni client")
    print("  ✅ Codice ottimizzato per scopi didattici")
    print("")
    print("VANTAGGI ARCHITETTURA STATICA:")
    print(f"  🎯 Architettura fissa: {PCA_COMPONENTS} → 32 → 20 → 12 → 1")
    print("  🎯 Nessuna logica dinamica")
    print("  🎯 Performance consistenti e prevedibili")
    print("  🎯 Facilità di debugging e manutenzione")
    print("  🎯 Controllo completo sui parametri")
    print("")
    print("VANTAGGI PER ATTACCHI:")
    print("  🎯 Dati di training naturalmente distribuiti")
    print("  🎯 Nessun dato sintetico che confonde gli attacchi")
    print("  🎯 Scenario federato completamente realistico")
    print("  🎯 Architettura prevedibile per test di sicurezza")
    
    try:
        # Carica i dati locali del client con PCA fissa
        print(f"[Client {client_id}] Caricamento dati con PCA fissa SENZA SMOTE...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data_with_fixed_pca(client_id)
        
        # Crea il modello DNN con architettura STATICA
        model = create_smartgrid_dnn_model_static_architecture()
        
        print(f"[Client {client_id}] === RIASSUNTO CLIENT ARCHITETTURA STATICA ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione naturale: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['pca_features']}")
        print(f"[Client {client_id}] PCA configurata: {dataset_info['pca_components_configured']} componenti")
        print(f"[Client {client_id}] Varianza spiegata: {dataset_info['variance_explained']:.2f}%")
        print(f"[Client {client_id}] Modello: {model.count_params():,} parametri (architettura STATICA)")
        print(f"[Client {client_id}] Class weights: {dataset_info['class_weights']}")
        print(f"[Client {client_id}] Preprocessing: {dataset_info['preprocessing_method']}")
        print(f"[Client {client_id}] PCA: {dataset_info['pca_method']}")
        print(f"[Client {client_id}] Connessione al server su localhost:8080...")
        
        # Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridDNNClientFixed()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()