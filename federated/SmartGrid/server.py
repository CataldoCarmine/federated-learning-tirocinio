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

def create_server_dnn_model(input_shape):
    """
    Crea il modello DNN per il server identico ai client.
    """
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    model = tf.keras.Sequential([
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
    
    optimizer = tf.keras.optimizers.Adam(
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

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid DNN.
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Costruzione path ai file CSV
        test_clients = [14, 15]
        df_list = []

        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
            except FileNotFoundError:
                continue

        if not df_list:
            # Fallback
            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(200, len(df_fallback)), random_state=42)]
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per valutazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        # Applica la stessa pipeline robusta dei client
        fixed_pca_components = 50
        X_global_final, pca_components = apply_server_preprocessing_pipeline_robust(
            X_global, 
            fixed_pca_components=fixed_pca_components
        )
        
        return X_global_final, y_global, pca_components
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, input_shape = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi
        input_shape = 50
        X_global = np.random.random((100, input_shape))
        y_global = np.random.randint(0, 2, 100)
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round.
        """
        print(f"\n=== VALUTAZIONE GLOBALE - ROUND {server_round} ===")
        
        try:
            # Crea il modello DNN per la valutazione
            model = create_server_dnn_model(input_shape)
            
            # Usa funzione sicura per impostare pesi
            success, error_msg = safe_set_server_model_weights(model, parameters)
            
            if not success:
                print(f"Errore nell'impostazione parametri server: {error_msg}")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": f"server_parameter_handling_failed: {error_msg}", 
                    "global_test_samples": 0
                }
            
            # Valutazione sul dataset test globale
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"F1-Score: {f1_score:.4f}, AUC: {auc:.4f}")
            print(f"Campioni test: {len(X_global)}, Feature: {X_global.shape[1]}")
            
            # Predizioni per analisi dettagliata
            predictions_prob = model.predict(X_global, verbose=0)
            predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
            
            # Matrice di confusione
            tn = np.sum((y_global == 0) & (predictions_binary == 0))
            fp = np.sum((y_global == 0) & (predictions_binary == 1))
            fn = np.sum((y_global == 1) & (predictions_binary == 0))
            tp = np.sum((y_global == 1) & (predictions_binary == 1))
            
            # Metriche di sicurezza
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f"Matrice confusione: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print(f"Specificity: {specificity:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")
            
            return float(loss), {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "auc": float(auc),
                "specificity": float(specificity),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "global_test_samples": int(len(X_global)),
                "pipeline_features": int(input_shape),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            return 1.0, {
                "accuracy": 0.0, 
                "error": str(e), 
                "global_test_samples": 0
            }
    
    return evaluate

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round di addestramento.
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    error_clients = []
    accuracy_list = []
    loss_list = []
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1}: {client_samples} campioni")
        
        if 'error' in client_metrics:
            error_clients.append(i+1)
            print(f"  ERRORE: {client_metrics['error']}")
            continue
        
        if 'train_accuracy' in client_metrics:
            accuracy = client_metrics['train_accuracy']
            total_weighted_accuracy += accuracy * client_samples
            accuracy_list.append(accuracy)
            print(f"  Accuracy: {accuracy:.4f}")
        
        if 'train_loss' in client_metrics:
            loss = client_metrics['train_loss']
            loss_list.append(loss)
            print(f"  Loss: {loss:.4f}")
        
        if 'pca_method' in client_metrics:
            pca_method = client_metrics['pca_method']
            print(f"  Metodo PCA: {pca_method}")
    
    if total_samples > 0:
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_loss = np.mean(loss_list) if loss_list else 0
        
        print(f"\nRiassunto:")
        print(f"  Media accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  Media loss: {avg_loss:.4f}")
        print(f"  Totale campioni: {total_samples}")
        print(f"  Client con errori: {len(error_clients)}")

class SmartGridDNNFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid DNN.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento DNN.
        """
        print(f"\n=== AGGREGAZIONE TRAINING - ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("ERRORE: Nessun client ha fornito risultati validi")
            return None
        
        # Stampa metriche dei client
        print_client_metrics(results)
        
        # Chiama l'aggregazione standard
        try:
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"Aggregazione completata per round {server_round}")
            else:
                print(f"ATTENZIONE: Aggregazione fallita per round {server_round}")
                
        except Exception as e:
            print(f"ERRORE durante aggregazione: {e}")
            return None
        
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione DNN.
        """
        print(f"\n=== AGGREGAZIONE VALUTAZIONE ROUND {server_round} ===")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"Aggregazione valutazione completata per round {server_round}")
            else:
                print(f"Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"ERRORE durante aggregazione valutazione: {e}")
            return None
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato DNN.
    """
    print("=== SERVER FEDERATO SMARTGRID DNN ===")
    print("Funzionalità implementate:")
    print("  - Stabilità numerica PCA con fallback automatico")
    print("  - Gestione robusta Parameters Flower")
    print("  - Input layer esplicito per compatibilità")
    print("  - Architettura DNN standardizzata")
    print("")
    print("Configurazione:")
    print("  - Rounds: 5")
    print("  - Client minimi: 2")
    print("  - Strategia: FedAvg personalizzata")
    print("  - Valutazione: Dataset globale (client 14-15)")
    print("  - Pipeline: Pulizia → Imputazione → Normalizzazione → SMOTE → PCA robusto")
    print("  - Architettura: Input → Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Strategia Federated Averaging personalizzata
    strategy = SmartGridDNNFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_smartgrid_evaluate_fn()
    )
    
    print("\nServer in attesa di client...")
    print("Per connettere i client, esegui:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  ...")
    print("  python client.py 13")
    print("\nClient 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    
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