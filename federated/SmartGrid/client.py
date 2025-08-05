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
from imblearn.over_sampling import SMOTE

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

def apply_robust_pca(X_preprocessed, variance_threshold=0.95, client_id=None):
    """
    Applica PCA con controlli di stabilità numerica e fallback automatico.
    """
    original_features = X_preprocessed.shape[1]
    
    # Pulizia robusta dei dati pre-PCA
    X_stable = ensure_numerical_stability(X_preprocessed, f"pre-PCA client {client_id}")
    
    # Tentativo PCA principale
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            
            pca_full = PCA()
            pca_full.fit(X_stable)
            
            if np.any(np.isnan(pca_full.explained_variance_ratio_)) or np.any(np.isinf(pca_full.explained_variance_ratio_)):
                raise ValueError(f"PCA client {client_id} ha prodotto explained_variance_ratio_ non validi")
            
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components_selected = np.argmax(cumulative_variance >= variance_threshold) + 1
            n_components_selected = min(n_components_selected, original_features, len(X_stable))
            n_components_selected = max(n_components_selected, min(10, original_features))
            
            pca_optimal = PCA(n_components=n_components_selected)
            X_pca = pca_optimal.fit_transform(X_stable)
            
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError(f"PCA client {client_id} ha prodotto output con NaN o inf")
            
            return X_pca, n_components_selected
            
    except Exception as e:
        print(f"[Client {client_id}] PCA normale fallito: {e}, attivazione fallback...")
        
        # FALLBACK 1: PCA con regolarizzazione
        try:
            regularization = 1e-6
            X_regularized = X_stable + np.random.normal(0, regularization, X_stable.shape)
            X_regularized = ensure_numerical_stability(X_regularized, f"PCA regularized client {client_id}")
            
            pca_reg = PCA(n_components=min(30, original_features, len(X_stable)))
            X_pca_reg = pca_reg.fit_transform(X_regularized)
            
            if not (np.any(np.isnan(X_pca_reg)) or np.any(np.isinf(X_pca_reg))):
                return X_pca_reg, X_pca_reg.shape[1]
            else:
                raise ValueError("PCA regolarizzata ha prodotto valori non validi")
                
        except Exception as e2:
            # FALLBACK 2: Selezione feature per varianza
            try:
                feature_vars = np.var(X_stable, axis=0)
                feature_vars = np.where(np.isnan(feature_vars), 0, feature_vars)
                
                n_components_fallback = min(20, original_features)
                top_features = np.argsort(feature_vars)[-n_components_fallback:]
                
                X_fallback = X_stable[:, top_features]
                X_fallback = ensure_numerical_stability(X_fallback, f"feature selection client {client_id}")
                
                return X_fallback, n_components_fallback
                
            except Exception as e3:
                # FALLBACK 3: Riduzione semplice
                n_components_final = min(10, original_features)
                X_final = X_stable[:, :n_components_final]
                X_final = ensure_numerical_stability(X_final, f"simple reduction client {client_id}")
                
                return X_final, n_components_final

def load_client_smartgrid_data(client_id, fixed_pca_components=50):
    """
    Carica i dati SmartGrid per un client specifico con stabilità numerica PCA.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    df = pd.read_csv(file_path)
    
    # Separa feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    # Statistiche base
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    # Pulizia robusta preliminare
    X_cleaned = clean_data_for_pca(X)
    
    # Suddivisione train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_cleaned, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Pipeline di preprocessing
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_raw)
    X_val_preprocessed = preprocessing_pipeline.transform(X_val_raw)
    
    # SMOTE per bilanciamento classi
    train_attack_ratio = y_train.mean()
    minority_class_ratio = min(train_attack_ratio, 1 - train_attack_ratio)
    unique_classes = len(np.unique(y_train))
    
    if minority_class_ratio < 0.4 and unique_classes > 1:
        try:
            min_samples_per_class = min((y_train == 0).sum(), (y_train == 1).sum())
            k_neighbors = min(5, min_samples_per_class - 1) if min_samples_per_class > 1 else 1
            
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)
        except Exception as e:
            X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
    
    # PCA robusto con fallback
    X_train_final, n_components = apply_robust_pca(
        X_train_balanced, 
        variance_threshold=0.95, 
        client_id=client_id
    )
    
    # Applica la stessa trasformazione ai dati di validation
    if n_components <= X_val_preprocessed.shape[1]:
        X_val_final = X_val_preprocessed[:, :n_components]
    else:
        X_val_final = X_val_preprocessed[:, :min(n_components, X_val_preprocessed.shape[1])]
        n_components = X_val_final.shape[1]
    
    X_val_final = ensure_numerical_stability(X_val_final, f"validation final client {client_id}")
    
    # Calcola varianza spiegata
    try:
        total_variance = np.var(X_train_preprocessed, axis=0).sum()
        final_variance = np.var(X_train_final, axis=0).sum()
        variance_explained = min(final_variance / total_variance, 1.0) if total_variance > 0 else 0.95
    except:
        variance_explained = 0.95
    
    # Informazioni del dataset
    dataset_info = {
        'client_id': client_id,
        'total_samples': len(df),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio,
        'original_features': X.shape[1],
        'pca_features': n_components,
        'pca_reduction': (1 - n_components / X.shape[1]) * 100,
        'variance_explained': variance_explained * 100,
        'pca_method': 'robust_with_fallback'
    }
    
    return X_train_final, y_train_balanced, X_val_final, y_val, dataset_info

def create_smartgrid_client_dnn_model(input_shape):
    """
    Crea il modello DNN SmartGrid standardizzato per il client.
    """
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    model = keras.Sequential([
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        layers.Dense(128, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        layers.Dense(64, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        layers.Dense(32, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
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

class SmartGridDNNClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con DNN e stabilità numerica PCA.
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
        Addestra il modello DNN sui dati locali del client.
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] Round di addestramento...")
        
        # Usa funzione sicura per impostare pesi
        success, error_msg = safe_set_model_weights(model, parameters, client_id)
        
        if not success:
            print(f"[Client {client_id}] Errore parametri: {error_msg}")
            return model.get_weights(), 0, {'error': f'parameter_handling_failed: {error_msg}'}
        
        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Configurazione addestramento
        local_epochs = 3
        batch_size = 16
        
        try:
            history = model.fit(
                X_train, y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )
            
            # Estrai metriche
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            train_precision = history.history.get('precision', [0])[-1]
            train_recall = history.history.get('recall', [0])[-1]
            train_auc = history.history.get('auc', [0])[-1]
            
            # Calcola F1-score
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            print(f"[Client {client_id}] Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            
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
            'variance_explained': float(dataset_info['variance_explained']),
            'weights_count': len(model.get_weights()),
            'pca_method': dataset_info['pca_method']
        }
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello DNN sui dati di validation locali del client.
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
            results = model.evaluate(X_val, y_val, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"[Client {client_id}] Val Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
            
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
    Funzione principale per avviare il client SmartGrid DNN.
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
    
    print(f"Avvio Client SmartGrid DNN {client_id}")
    print("Funzionalità implementate:")
    print("  - Stabilità numerica PCA con fallback automatico")
    print("  - Gestione robusta Parameters Flower")
    print("  - Input layer esplicito per compatibilità")
    
    try:
        # Carica i dati locali del client
        fixed_pca_components = 50
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id, fixed_pca_components)
        
        # Crea il modello DNN
        model = create_smartgrid_client_dnn_model(X_train.shape[1])
        
        print(f"[Client {client_id}] Dataset caricato: {dataset_info['train_samples']} training, {dataset_info['val_samples']} validation")
        print(f"[Client {client_id}] Modello creato: {model.count_params():,} parametri, {X_train.shape[1]} feature input")
        print(f"[Client {client_id}] Metodo PCA: {dataset_info['pca_method']}")
        
        # Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridDNNClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()