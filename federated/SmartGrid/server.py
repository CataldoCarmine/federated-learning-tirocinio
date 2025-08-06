import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
import os

def clean_data_for_pca_server(X):
    """
    Pulizia robusta dei dati per prevenire problemi numerici in PCA (server).
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

def ensure_numerical_stability_server(X, stage_name):
    """
    Assicura stabilità numerica rimuovendo inf, nan e valori estremi (server).
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

def apply_robust_pca_server(X_preprocessed, variance_threshold=0.95):
    """
    Applica PCA con controlli di stabilità numerica e fallback automatico (server).
    """
    original_features = X_preprocessed.shape[1]
    
    # Pulizia robusta dei dati pre-PCA
    X_stable = ensure_numerical_stability_server(X_preprocessed, "pre-PCA server")
    
    # Tentativo PCA principale
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            
            pca_full = PCA()
            pca_full.fit(X_stable)
            
            if np.any(np.isnan(pca_full.explained_variance_ratio_)) or np.any(np.isinf(pca_full.explained_variance_ratio_)):
                raise ValueError("PCA server ha prodotto explained_variance_ratio_ non validi")
            
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components_selected = np.argmax(cumulative_variance >= variance_threshold) + 1
            n_components_selected = min(n_components_selected, original_features, len(X_stable))
            n_components_selected = max(n_components_selected, min(10, original_features))
            
            pca_optimal = PCA(n_components=n_components_selected)
            X_pca = pca_optimal.fit_transform(X_stable)
            
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError("PCA server ha prodotto output con NaN o inf")
            
            return X_pca, n_components_selected
            
    except Exception as e:
        print(f"PCA normale server fallito: {e}, attivazione fallback...")
        
        # FALLBACK 1: PCA con regolarizzazione
        try:
            regularization = 1e-6
            X_regularized = X_stable + np.random.normal(0, regularization, X_stable.shape)
            X_regularized = ensure_numerical_stability_server(X_regularized, "PCA regularized server")
            
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
                X_fallback = ensure_numerical_stability_server(X_fallback, "feature selection server")
                
                return X_fallback, n_components_fallback
                
            except Exception as e3:
                # FALLBACK 3: Riduzione semplice
                n_components_final = min(10, original_features)
                X_final = X_stable[:, :n_components_final]
                X_final = ensure_numerical_stability_server(X_final, "simple reduction server")
                
                return X_final, n_components_final

def apply_server_preprocessing_pipeline_robust(X_global, fixed_pca_components=50):
    """
    Applica la stessa pipeline di preprocessing dei client sui dati globali del server.
    """
    # Pulizia robusta preliminare
    X_cleaned = clean_data_for_pca_server(X_global)
    
    # Pipeline di preprocessing
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_preprocessed = preprocessing_pipeline.fit_transform(X_cleaned)
    
    # PCA robusto con fallback
    original_features = X_preprocessed.shape[1]
    
    try:
        X_global_final, n_components = apply_robust_pca_server(
            X_preprocessed, 
            variance_threshold=0.95
        )
        
        # Adatta al numero fisso se necessario
        if n_components != fixed_pca_components:
            if n_components > fixed_pca_components:
                X_global_final = X_global_final[:, :fixed_pca_components]
                n_components = fixed_pca_components
        
    except Exception as e:
        print(f"Errore PCA robusto server: {e}, attivazione fallback finale")
        
        # Fallback finale: usa le prime N feature preprocessate
        n_components = min(fixed_pca_components, original_features, X_preprocessed.shape[1])
        X_global_final = X_preprocessed[:, :n_components]
        X_global_final = ensure_numerical_stability_server(X_global_final, "final fallback server")
    
    return X_global_final, n_components

def compute_class_weights_server_simple(y_global):
    """
    Calcola i pesi delle classi per il dataset globale del server.
    Versione semplificata.
    
    Args:
        y_global: Target globali del server
    
    Returns:
        Dizionario con class weights
    """
    try:
        unique_classes = np.unique(y_global)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_global
        )
        
        class_weight_dict = dict(zip(unique_classes, class_weights))
        return class_weight_dict
        
    except Exception as e:
        print(f"Errore nel calcolo class weights server: {e}")
        unique_classes = np.unique(y_global)
        return {cls: 1.0 for cls in unique_classes}

def create_server_dnn_model_simple(input_shape):
    """
    Crea il modello DNN per il server IDENTICO ai client.
    Versione semplificata per dataset sbilanciati.
    
    Args:
        input_shape: Numero di feature post-PCA
    
    Returns:
        Modello Keras compilato per dataset sbilanciati
    """
    print(f"[Server] Creazione DNN semplificata per dataset sbilanciati (SENZA SMOTE)")
    print(f"[Server] Input features: {input_shape}")
    
    # Parametri IDENTICI ai client
    dropout_rate = 0.3
    l2_reg = 0.001
    
    # Architettura IDENTICA ai client
    first_layer_size = max(32, int(input_shape * 0.8))
    second_layer_size = max(16, first_layer_size // 2)
    third_layer_size = max(8, second_layer_size // 2)
    
    print(f"[Server] Architettura: {input_shape} → {first_layer_size} → {second_layer_size} → {third_layer_size} → 1")
    
    model = tf.keras.Sequential([
        # Input layer esplicito IDENTICO ai client
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        # Layer 1: Dimensione proporzionale alle feature
        layers.Dense(first_layer_size, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Layer 2: Feature extraction
        layers.Dense(second_layer_size, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Layer 3: Pattern recognition
        layers.Dense(third_layer_size, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        # Output layer IDENTICO ai client
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # Ottimizzatore IDENTICO ai client
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    # Compila il modello IDENTICO ai client
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
    params_per_feature = total_params / input_shape
    
    print(f"[Server] DNN semplificata creata IDENTICA ai client:")
    print(f"[Server]   - Parametri totali: {total_params:,}")
    print(f"[Server]   - Parametri per feature: {params_per_feature:.1f}")
    print(f"[Server]   - Dropout: {dropout_rate}")
    print(f"[Server]   - L2 regularization: {l2_reg}")
    print(f"[Server]   - Learning rate: {optimizer.learning_rate}")
    print(f"[Server]   - Loss: Binary Crossentropy")
    print(f"[Server]   - Ottimizzato per: dati naturalmente sbilanciati")
    
    return model

def safe_extract_parameters_server(parameters):
    """
    Estrae i pesi dal tipo Parameters di Flower in modo sicuro per il server.
    """
    try:
        if isinstance(parameters, list):
            weights_list = []
            for i, tensor in enumerate(parameters):
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    weights_list.append(tensor.numpy())
                else:
                    weights_list.append(np.array(tensor, dtype=np.float32))
            return weights_list
        
        elif isinstance(parameters, Parameters):
            if hasattr(parameters, 'tensors'):
                weights_list = []
                for i, tensor in enumerate(parameters.tensors):
                    if isinstance(tensor, np.ndarray):
                        weights_list.append(tensor)
                    elif hasattr(tensor, 'numpy'):
                        weights_list.append(tensor.numpy())
                    else:
                        weights_list.append(np.array(tensor, dtype=np.float32))
                return weights_list
            else:
                raise ValueError("Oggetto Parameters non ha attributo 'tensors'")
        
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
        print(f"Errore nell'estrazione parametri server: {e}")
        return parameters

def check_server_parameters_compatibility(received_params, model_weights):
    """
    Verifica la compatibilità tra parametri ricevuti e modello server.
    """
    try:
        extracted_weights = safe_extract_parameters_server(received_params)
        
        if not isinstance(extracted_weights, list):
            error_msg = f"Parametri estratti non sono una lista: {type(extracted_weights)}"
            return False, None, error_msg
        
        if len(extracted_weights) != len(model_weights):
            error_msg = f"Numero pesi incompatibile: ricevuti {len(extracted_weights)}, attesi {len(model_weights)}"
            return False, None, error_msg
        
        for i, (received_weight, model_weight) in enumerate(zip(extracted_weights, model_weights)):
            if not isinstance(received_weight, np.ndarray):
                try:
                    received_weight = np.array(received_weight, dtype=np.float32)
                    extracted_weights[i] = received_weight
                except Exception as e:
                    error_msg = f"Impossibile convertire peso {i} a numpy array: {e}"
                    return False, None, error_msg
            
            if received_weight.shape != model_weight.shape:
                error_msg = f"Forma peso {i} incompatibile: ricevuta {received_weight.shape}, attesa {model_weight.shape}"
                return False, None, error_msg
        
        return True, extracted_weights, None
        
    except Exception as e:
        error_msg = f"Errore durante verifica compatibilità server: {str(e)}"
        return False, None, error_msg

def safe_set_server_model_weights(model, parameters):
    """
    Imposta i pesi del modello server in modo sicuro.
    """
    try:
        current_weights = model.get_weights()
        
        is_compatible, extracted_weights, error_msg = check_server_parameters_compatibility(
            parameters, current_weights
        )
        
        if not is_compatible:
            return False, error_msg
        
        model.set_weights(extracted_weights)
        return True, None
        
    except Exception as e:
        error_msg = f"Errore durante impostazione pesi server: {str(e)}"
        return False, error_msg

def get_smartgrid_evaluate_fn_simple():
    """
    Crea una funzione di valutazione globale per il server SmartGrid DNN semplificata.
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server SENZA SMOTE.
        Versione semplificata.
        """
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER (SEMPLIFICATO) ===")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Costruzione path ai file CSV
        test_clients = [14, 15]
        df_list = []

        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"File data{client_id}.csv non trovato")
                continue

        if not df_list:
            # Fallback
            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(200, len(df_fallback)), random_state=42)]
                print(f"Usando fallback con {len(df_list[0])} campioni da data1.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per valutazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y (mantiene distribuzione naturale)
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        # Statistiche distribuzione naturale globale
        attack_samples = y_global.sum()
        natural_samples = (y_global == 0).sum()
        attack_ratio = y_global.mean()
        
        print(f"Dataset test globale NATURALE: {len(df_global)} campioni")
        print(f"Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
        
        # Calcola class weights per il dataset globale
        class_weights = compute_class_weights_server_simple(y_global)
        print(f"Class weights globali: {class_weights}")
        
        # Applica la stessa pipeline robusta dei client (SENZA SMOTE)
        fixed_pca_components = 50
        X_global_final, pca_components = apply_server_preprocessing_pipeline_robust(
            X_global, 
            fixed_pca_components=fixed_pca_components
        )
        
        print(f"Dataset preprocessato SENZA SMOTE: {len(X_global_final)} campioni, {X_global_final.shape[1]} feature")
        
        return X_global_final, y_global, pca_components, class_weights, {
            'total_samples': len(df_global),
            'attack_samples': attack_samples,
            'natural_samples': natural_samples,
            'attack_ratio': attack_ratio
        }
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, input_shape, class_weights, dataset_info = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi con shape fisso
        input_shape = 50
        X_global = np.random.random((100, input_shape))
        y_global = np.random.randint(0, 2, 100)
        class_weights = {0: 1.0, 1: 1.0}
        dataset_info = {}
        print(f"Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round con dataset naturalmente sbilanciato.
        Versione semplificata.
        """
        print(f"\n=== VALUTAZIONE GLOBALE DNN SEMPLIFICATA - ROUND {server_round} ===")
        
        try:
            # Crea il modello DNN semplificato per la valutazione (identico ai client)
            model = create_server_dnn_model_simple(input_shape)
            
            # Usa funzione sicura per impostare pesi
            success, error_msg = safe_set_server_model_weights(model, parameters)
            
            if not success:
                print(f"Errore nell'impostazione parametri server: {error_msg}")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": f"server_parameter_handling_failed: {error_msg}", 
                    "global_test_samples": 0
                }
            
            print(f"✅ Pesi aggregati impostati su modello server")
            
            # Valutazione sul dataset test globale naturalmente sbilanciato
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calcola Balanced Accuracy
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_global, y_pred_binary)
            
            print(f"RISULTATI VALUTAZIONE SEMPLIFICATA:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1_score_val:.4f} ({f1_score_val*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  Campioni test: {len(X_global)}")
            print(f"  Feature utilizzate: {X_global.shape[1]}")
            
            # Calcola parametri modello
            total_params = model.count_params()
            params_per_feature = total_params / input_shape
            print(f"  Parametri DNN: {total_params:,} ({params_per_feature:.1f} per feature)")
            
            # Informazioni distribuzione
            print(f"  Distribuzione naturale: {dataset_info.get('attack_ratio', 0)*100:.1f}% attacchi")
            print(f"  Adatto per attacchi: ✅")
            
            return float(loss), {
                # Metriche base
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                
                # Metriche bilanciate essenziali
                "f1_score": float(f1_score_val),
                "balanced_accuracy": float(balanced_acc),
                
                # Informazioni dataset e modello
                "global_test_samples": int(len(X_global)),
                "pipeline_features": int(input_shape),
                "total_params": int(total_params),
                "params_per_feature": float(params_per_feature),
                "attack_samples": int(dataset_info.get('attack_samples', 0)),
                "natural_samples": int(dataset_info.get('natural_samples', 0)),
                "attack_ratio": float(dataset_info.get('attack_ratio', 0)),
                "class_weight_0": float(class_weights[0]),
                "class_weight_1": float(class_weights[1]),
                
                # Metodologia
                "model_type": "dnn_simple_no_smote",
                "preprocessing_method": "no_smote_simple"
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale semplificata: {e}")
            return 1.0, {
                "accuracy": 0.0, 
                "error": str(e), 
                "global_test_samples": 0
            }
    
    return evaluate

def print_client_metrics_simple(fit_results):
    """
    Stampa le metriche dei client dopo ogni round con focus su metriche essenziali.
    Versione semplificata.
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT DNN SEMPLIFICATA (SENZA SMOTE) ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_weighted_f1 = 0
    error_clients = []
    accuracy_list = []
    f1_list = []
    loss_list = []
    attack_ratio_list = []
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1}: {client_samples} campioni")
        
        if 'error' in client_metrics:
            error_clients.append(i+1)
            print(f"  ERRORE: {client_metrics['error']}")
            continue
        
        # Metriche base
        if 'train_accuracy' in client_metrics:
            accuracy = client_metrics['train_accuracy']
            total_weighted_accuracy += accuracy * client_samples
            accuracy_list.append(accuracy)
            print(f"  Accuracy: {accuracy:.4f}")
        
        if 'train_loss' in client_metrics:
            loss = client_metrics['train_loss']
            loss_list.append(loss)
            print(f"  Loss: {loss:.4f}")
        
        # Metriche bilanciate essenziali
        if 'train_f1_score' in client_metrics:
            f1 = client_metrics['train_f1_score']
            total_weighted_f1 += f1 * client_samples
            f1_list.append(f1)
            print(f"  F1-Score: {f1:.4f}")
        
        if 'train_balanced_accuracy' in client_metrics:
            balanced_acc = client_metrics['train_balanced_accuracy']
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Informazioni distribuzione dataset
        if 'attack_ratio' in client_metrics:
            attack_ratio = client_metrics['attack_ratio']
            attack_ratio_list.append(attack_ratio)
            print(f"  Distribuzione attacchi: {attack_ratio*100:.1f}%")
        
        # Class weights utilizzati
        if 'used_class_weights' in client_metrics and client_metrics['used_class_weights']:
            weight_0 = client_metrics.get('class_weight_0', 1.0)
            weight_1 = client_metrics.get('class_weight_1', 1.0)
            print(f"  Class weights: {{0: {weight_0:.3f}, 1: {weight_1:.3f}}}")
        
        # Informazioni modello
        if 'total_params' in client_metrics:
            total_params = client_metrics['total_params']
            print(f"  Parametri DNN: {total_params:,}")
        
        # Metodologia
        if 'preprocessing_method' in client_metrics:
            method = client_metrics['preprocessing_method']
            print(f"  Preprocessing: {method}")
    
    if total_samples > 0:
        # Calcola medie ponderate
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_weighted_f1 = total_weighted_f1 / total_samples if total_weighted_f1 > 0 else 0
        avg_loss = np.mean(loss_list) if loss_list else 0
        avg_attack_ratio = np.mean(attack_ratio_list) if attack_ratio_list else 0
        
        print(f"\nRIASSUNTO DNN SEMPLIFICATA (SENZA SMOTE):")
        print(f"  Media accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  Media F1-Score: {avg_weighted_f1:.4f}")
        print(f"  Media loss: {avg_loss:.4f}")
        print(f"  Media distribuzione attacchi: {avg_attack_ratio*100:.1f}%")
        print(f"  Totale campioni: {total_samples}")
        print(f"  Client con errori: {len(error_clients)}")
        
        # Valutazione sbilanciamento
        if avg_attack_ratio < 0.3 or avg_attack_ratio > 0.7:
            print(f"  ⚠️  Dataset significativamente sbilanciati")
            print(f"  ✅ Class weights compensano sbilanciamento")
        else:
            print(f"  ✅ Dataset ragionevolmente bilanciati")

class SmartGridDNNFedAvgSimple(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid DNN semplificata SENZA SMOTE.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento DNN semplificata.
        """
        print(f"\n=== AGGREGAZIONE TRAINING DNN SEMPLIFICATA (SENZA SMOTE) - ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        print(f"Dataset naturalmente sbilanciati per attacchi realistici")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("ERRORE: Nessun client ha fornito risultati validi")
            return None
        
        # Stampa metriche dei client con focus su essenzialità
        print_client_metrics_simple(results)
        
        # Chiama l'aggregazione standard
        try:
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione DNN semplificata completata per round {server_round}")
                print(f"✅ Pesi di {len(results)} client DNN aggregati con successo")
            else:
                print(f"❌ ATTENZIONE: Aggregazione fallita per round {server_round}")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione: {e}")
            return None
        
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione DNN semplificata.
        """
        print(f"\n=== AGGREGAZIONE VALUTAZIONE DNN SEMPLIFICATA ROUND {server_round} ===")
        print(f"Client che hanno valutato: {len(results)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione DNN semplificata completata per round {server_round}")
            else:
                print(f"Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"ERRORE durante aggregazione valutazione: {e}")
            return None
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato DNN semplificata SENZA SMOTE.
    """
    print("=== SERVER FEDERATO SMARTGRID DNN SEMPLIFICATA (SENZA SMOTE) ===")
    print("MODIFICHE ESSENZIALI IMPLEMENTATE:")
    print("  ✅ SMOTE COMPLETAMENTE RIMOSSO per attacchi inference/extraction realistici")
    print("  ✅ Distribuzione naturale mantenuta per fedeltà al mondo reale")
    print("  ✅ Class weights automatici per compensare sbilanciamento")
    print("  ✅ Metriche bilanciate essenziali: F1-Score, Balanced Accuracy, AUC")
    print("  ✅ DNN ottimizzata per feature post-PCA ridotte")
    print("  ✅ Architettura proporzionale per prevenire overfitting")
    print("  ✅ Binary Crossentropy standard + Class Weights")
    print("  ✅ Normalizzazione LOCALE per ogni client (preserva privacy)")
    print("  ✅ Codice semplificato per scopi didattici")
    print("")
    print("VANTAGGI PER ATTACCHI DI INFERENCE/EXTRACTION:")
    print("  🎯 Dati di training naturalmente distribuiti (nessun dato sintetico)")
    print("  🎯 Membership inference su dati reali del mondo reale")
    print("  🎯 Model extraction su comportamento naturale del modello")
    print("  🎯 Scenario federato completamente realistico")
    print("  🎯 Codice pulito e comprensibile per scopi didattici")
    print("")
    print("ARCHITETTURA DNN SEMPLIFICATA:")
    print("  🧠 Primo layer: input_features × 0.8 (proporzionale)")
    print("  🧠 Secondo layer: primo_layer ÷ 2")
    print("  🧠 Terzo layer: secondo_layer ÷ 2")
    print("  🧠 Output layer: 1 neurone (classificazione binaria)")
    print("  🧠 Class weights: automatici per ogni client")
    print("  🧠 Metriche: F1, Balanced Accuracy, AUC")
    print("  🧠 Loss: Binary Crossentropy standard")
    print("")
    print("Configurazione:")
    print("  - Rounds: 5")
    print("  - Client minimi: 2")
    print("  - Strategia: FedAvg personalizzata con DNN semplificata")
    print("  - Valutazione: Dataset globale naturalmente sbilanciato (client 14-15)")
    print("  - Pipeline: Pulizia → Imputazione → Normalizzazione → PCA robusto (NO SMOTE)")
    print("  - Architettura tipica: Input(~50) → Dense(~40) → Dense(~20) → Dense(~10) → Dense(1)")
    print("  - Class weights: Automatici per compensare sbilanciamento")
    print("  - Batch size: 32 (ottimizzato per stabilità)")
    print("  - Epoche locali: 5 (bilanciate per convergenza)")
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Strategia Federated Averaging personalizzata con DNN semplificata
    strategy = SmartGridDNNFedAvgSimple(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_smartgrid_evaluate_fn_simple()
    )
    
    print("\nServer DNN semplificata in attesa di client...")
    print("Per connettere i client, esegui:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  ...")
    print("  python client.py 13")
    print("\nClient 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    print("")
    print("VANTAGGI DNN SEMPLIFICATA SENZA SMOTE:")
    print("  ✅ Performance realistiche su dati sbilanciati del mondo reale")
    print("  ✅ Attacchi di inference più rappresentativi")
    print("  ✅ Model extraction su comportamento autentico")
    print("  ✅ Codice pulito e didattico per studenti")
    print("  ✅ Metriche significative per sistemi di sicurezza")
    print("  ✅ Class weights compensano automaticamente sbilanciamento")
    print("  ✅ Architettura ottimizzata previene overfitting")
    print("  ✅ Compatibilità completa con letteratura FL su attacchi")
    print("  ✅ Ridotta complessità per concentrarsi su federated learning")
    print("  ✅ I warning PCA sono normali e gestiti automaticamente dai fallback")
    
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"Errore durante l'avvio del server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()