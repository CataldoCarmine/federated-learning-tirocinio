import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
import os
warnings.filterwarnings('ignore')

# CONFIGURAZIONE PCA STATICA 
PCA_COMPONENTS = 21  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_STATE = 42

# CONFIGURAZIONE MODELLO DNN
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
    Pulizia robusta dei dati per prevenire problemi numerici in PCA (server).
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X_preprocessed):
    """
    Applica PCA con numero FISSO di componenti (server, identico ai client).
    """
    print(f"[Server] === APPLICAZIONE PCA ===")

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
                raise ValueError("PCA server ha prodotto output con NaN o inf")
            if X_pca.shape[1] != n_components:
                raise ValueError(f"PCA server output shape inconsistente: {X_pca.shape[1]} vs {n_components}")
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            print(f"[Server] ✅ PCA fissa server applicata: {X_pca.shape}")
            print(f"[Server] Varianza spiegata: {variance_explained*100:.2f}%")
            return X_pca
        
    except Exception as e:
        print(f"[Server] ERRORE PCA fissa server: {e}")
        print(f"[Server] Attivazione fallback...")
        n_fallback = min(n_components, original_features)
        X_fallback = X_preprocessed[:, :n_fallback]
        print(f"[Server] ✅ Fallback server: {X_fallback.shape}")
        return X_fallback

def apply_preprocessing_pipeline(X_global):
    """
    Applica la stessa pipeline di preprocessing dei client sui dati globali del server.
    Pipeline:
      - Pulizia inf/NaN
      - Clipping outlier per quantili (feature-wise)
      - Imputazione mediana
      - Rimozione feature quasi-costanti
      - Scaling standard
      - PCA fissa
    """
    print(f"[Server] === PIPELINE PREPROCESSING SERVER ===")

    # Pulizia preliminare (inf/NaN)
    X_cleaned = clean_data_for_pca(X_global)
    # Clipping outlier feature-wise usando limiti calcolati sui dati di test globali
    X_clipped = clip_outliers_iqr(np.array(X_cleaned, dtype=float))
    # Imputazione mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_clipped)
    # Rimozione feature quasi-costanti
    X_reduced, keep_mask = remove_near_constant_features(X_imputed, threshold_var=1e-12, threshold_ratio=0.999)
    print(f"[Server] Feature dopo rimozione quasi-costanti: {X_reduced.shape[1]} (da {X_imputed.shape[1]})")
    # Scaling standard
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    print(f"[Server] Preprocessing completato (clipping, imputazione, costanti, scaling)")
    # PCA fissa identica ai client (garantisce compatibilità)
    X_global_final = apply_pca(X_scaled)

    # VERIFICA FINALE: Dimensioni corrette garantite
    if X_global_final.shape[1] != PCA_COMPONENTS:
        raise RuntimeError(f"Server PCA output shape inconsistente: {X_global_final.shape[1]} vs {PCA_COMPONENTS}")
    print(f"[Server] ✅ Pipeline preprocessing con PCA fissa completata")
    print(f"[Server] Risultato finale: {X_global_final.shape}")
    return X_global_final

def compute_class_weights(y_global):
    """
    Calcola i pesi delle classi per il dataset globale del server.
    Versione semplificata identica ai client.
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

def create_dnn_model():
    """
    Crea il modello DNN per il server IDENTICO ai client con architettura FISSA.
    
    Returns:
        Modello Keras compilato IDENTICO ai client
    """
    print(f"[Server] === CREAZIONE DNN ARCHITETTURA FISSA SERVER ===")
    print(f"[Server] Architettura: {PCA_COMPONENTS} → 112 → 64 → 12 → 10 → 1")
    print(f"[Server] Attivazione: {ACTIVATION_FUNCTION}")
    print(f"[Server] Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"[Server] Dropout esteso: {EXTENDED_DROPOUT}")

    # Parametri IDENTICI ai client
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
    
    # MODELLO
    model = tf.keras.Sequential([
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
        
        # Output layer 
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # OTTIMIZZATORE  
    if USE_ADAMW:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.006025741928842929,  
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Server] Ottimizzatore: AdamW")
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.006025741928842929,
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
    
    # Statistiche modello
    total_params = model.count_params()
    
    return model

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid.
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Usa PCA fissa identica ai client.
        """
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER ===")
        
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
        class_weights = compute_class_weights(y_global)
        print(f"Class weights globali: {class_weights}")
        
        # Applica pipeline con PCA 
        X_global_final = apply_preprocessing_pipeline(X_global)
        
        print(f"Dataset preprocessato con PCA FISSA: {len(X_global_final)} campioni, {X_global_final.shape[1]} feature")
        print(f"Componenti PCA fisse: {PCA_COMPONENTS}")
        
        return X_global_final, y_global, class_weights, {
            'total_samples': len(df_global),
            'attack_samples': attack_samples,
            'natural_samples': natural_samples,
            'attack_ratio': attack_ratio
        }
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, class_weights, dataset_info = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi con shape fisso
        X_global = np.random.random((100, PCA_COMPONENTS))
        y_global = np.random.randint(0, 2, 100)
        class_weights = {0: 1.0, 1: 1.0}
        dataset_info = {}
        print(f"Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round con architettura FISSA.
        SEMPLIFICATO: Compatibilità garantita automaticamente.
        """
        print(f"\n=== VALUTAZIONE GLOBALE - ROUND {server_round + 1} ===")
        
        try:
            # Crea il modello DNN per la valutazione
            model = create_dnn_model()
            
            # IMPOSTAZIONE PESI SEMPLIFICATA
            try:
                model.set_weights(parameters)
                print(f"✅ Pesi aggregati impostati su modello server")
            except Exception as e:
                print(f"Errore nell'impostazione parametri server: {e}")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": f"server_weight_setting_failed: {str(e)}", 
                    "global_test_samples": 0
                }
            
            # Valutazione sul dataset test globale naturalmente sbilanciato
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # F1-score
            f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Balanced Accuracy
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_global, y_pred_binary)

            # Metriche per classe 
            report = classification_report(y_global, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_global, y_pred_binary)

            
            print(f"RISULTATI VALUTAZIONE:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1_score_val:.4f} ({f1_score_val*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  Campioni test: {len(X_global)}")
            
            # Calcola parametri modello
            total_params = model.count_params()
            print(f"  Distribuzione naturale: {dataset_info.get('attack_ratio', 0)*100:.1f}% attacchi")
            
            print(f"Classification report (per classe):")
            print(classification_report(y_global, y_pred_binary, target_names=["natural", "attack"], zero_division=0))
            print(f"Confusion matrix:")
            print(conf_matrix)


            return float(loss), {
                # Metriche base
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "balanced_accuracy": float(balanced_acc),

                # Nuove metriche per classe
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
                "tp": int(conf_matrix[1, 1]),
                
                # Informazioni dataset e modello
                "global_test_samples": int(len(X_global)),
                "total_params": int(total_params),
                "attack_samples": int(dataset_info.get('attack_samples', 0)),
                "natural_samples": int(dataset_info.get('natural_samples', 0)),
                "attack_ratio": float(dataset_info.get('attack_ratio', 0)),
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {
                "accuracy": 0.0, 
                "error": str(e), 
                "global_test_samples": 0,
                "architecture_fixed": False,
                "compatibility_guaranteed": False
            }
    
    return evaluate

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round con architettura fissa.
    SEMPLIFICATO: Focus sui risultati, non sui controlli di compatibilità.
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_weighted_f1 = 0
    error_clients = []
    accuracy_list = []
    f1_list = []
    loss_list = []
    early_stopped_count = 0
    compatibility_guaranteed_count = 0
    
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
        
        # Metriche bilanciate
        if 'train_f1_score' in client_metrics:
            f1 = client_metrics['train_f1_score']
            total_weighted_f1 += f1 * client_samples
            f1_list.append(f1)
            print(f"  F1-Score: {f1:.4f}")
        
        if 'train_balanced_accuracy' in client_metrics:
            balanced_acc = client_metrics['train_balanced_accuracy']
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Early stopping
        if 'early_stopped' in client_metrics:
            early_stopped = client_metrics['early_stopped']
            if early_stopped:
                early_stopped_count += 1
                print(f"  ✅ EarlyStopping attivato")
        
        # Informazioni training
        if 'local_epochs_actual' in client_metrics and 'local_epochs_planned' in client_metrics:
            actual = client_metrics['local_epochs_actual']
            planned = client_metrics['local_epochs_planned']
            print(f"  Epoche: {actual}/{planned}")
        
        if 'batch_size' in client_metrics:
            batch_size = client_metrics['batch_size']
            print(f"  Batch size: {batch_size}")
    
    if total_samples > 0:
        # Calcola medie ponderate
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_weighted_f1 = total_weighted_f1 / total_samples if total_weighted_f1 > 0 else 0
        avg_loss = np.mean(loss_list) if loss_list else 0
        
        print(f"\nRIASSUNTO METRICHE:")
        print(f"  Media accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  Media F1-Score: {avg_weighted_f1:.4f}")
        print(f"  Media loss: {avg_loss:.4f}")
        print(f"  Totale campioni: {total_samples}")
        print(f"  Client con errori: {len(error_clients)}")
        print(f"  Client con EarlyStopping: {early_stopped_count}/{len(fit_results)}")

class SmartGridDNNFedAvgFixed(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid DNN con architettura FISSA.
    SEMPLIFICATO: Compatibilità garantita automaticamente.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento DNN con architettura FISSA.
        SEMPLIFICATO: Nessun controllo di compatibilità necessario.
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
        
        # Stampa metriche dei client con architettura fissa
        print_client_metrics(results)
        
        # Chiama l'aggregazione standard (semplificata)
        try:
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione completata per round {server_round}")
                print(f"✅ Pesi di {len(results)} client aggregati con successo")
            else:
                print(f"❌ ATTENZIONE: Aggregazione fallita per round {server_round}")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione DNN con architettura FISSA.
        SEMPLIFICATO: Compatibilità garantita automaticamente.
        """
        print(f"\n=== AGGREGAZIONE VALUTAZIONE ROUND {server_round} ===")
        print(f"Client che hanno valutato: {len(results)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione completata per round {server_round}")
            else:
                print(f"Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"ERRORE durante aggregazione valutazione: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato DNN con architettura FISSA.
    SEMPLIFICATO: Controlli di compatibilità rimossi (non necessari).
    """
    print("=== SERVER FEDERATO SMARTGRID ===")
    print("Configurazione:")
    print(f"  - PCA Components: {PCA_COMPONENTS}")
    print(f"  - Architettura: {PCA_COMPONENTS} → 112 → 64 → 12 → 10 → 1")
    print(f"  - Attivazione: {ACTIVATION_FUNCTION}")
    print(f"  - Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"  - Learning Rate: 0.0008")
    print("  - Rounds: 100")
    print("  - Client minimi: 2")
    print("  - Strategia: FedAvg personalizzata con architettura fissa")
    print("  - Valutazione: Dataset globale con PCA fissa (client 14-15)")
    print("  - Pipeline: Pulizia → Imputazione → Normalizzazione → PCA fissa")
    print("  - Class weights: Automatici per compensare sbilanciamento")
    print("  - Regolarizzazione: Completa ma semplificata")
    print("  - Callback: EarlyStopping + ReduceLROnPlateau sui client")
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=100)
    
    # Strategia Federated Averaging personalizzata con architettura fissa
    strategy = SmartGridDNNFedAvgFixed(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_smartgrid_evaluate_fn()
    )
    
    print(f"\nServer DNN in attesa di client su localhost:8080...")
    print("Per connettere i client, esegui:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  ...")
    print("  python client.py 13")
    print("\nClient 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    print("")
    
    try:
        # Avvia il server Flower
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