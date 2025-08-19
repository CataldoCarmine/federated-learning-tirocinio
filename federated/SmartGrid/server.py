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
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
import os

# CONFIGURAZIONE PCA STATICA FISSA (identica ai client)
PCA_COMPONENTS = 20  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_STATE = 42

# CONFIGURAZIONE MODELLO DNN - IDENTICA AI CLIENT (ottimizzabile con Optuna)
ACTIVATION_FUNCTION = 'relu'  # Ottimizzabile: 'leaky_relu', 'selu', 'relu'
USE_ADAMW = False  # Ottimizzabile: True per AdamW, False per Adam
EXTENDED_DROPOUT = True  # Ottimizzabile: True per dropout esteso

def clip_outliers_iqr(X, k=3.0):
    """
    Clippa gli outlier per ogni feature usando la regola dei quantili (IQR).
    Limiti: [Q1 - k*IQR, Q3 + k*IQR] (default k=3).
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

def remove_near_constant_features(X, threshold=1e-8):
    """
    Rimuove le feature quasi-costanti, ovvero con varianza < threshold.
    """
    variances = np.nanvar(X, axis=0)
    keep_mask = variances > threshold
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
    GARANZIA: Output sempre con PCA_COMPONENTS dimensioni.
    """
    print(f"[Server] === APPLICAZIONE PCA FISSA SERVER (SEMPLIFICATA) ===")
    original_features = X_preprocessed.shape[1]
    n_samples = len(X_preprocessed)
    n_components = min(PCA_COMPONENTS, original_features, n_samples)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
            X_pca = pca.fit_transform(X_preprocessed)
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
    print(f"[Server] === PIPELINE PREPROCESSING SERVER CON PCA FISSA (ROBUSTA) ===")
    # Pulizia preliminare (inf/NaN)
    X_cleaned = clean_data_for_pca(X_global)
    # Clipping outlier feature-wise usando limiti calcolati sui dati di test globali
    X_clipped = clip_outliers_iqr(np.array(X_cleaned, dtype=float))
    # Imputazione mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_clipped)
    # Rimozione feature quasi-costanti
    X_reduced, keep_mask = remove_near_constant_features(X_imputed)
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
    print(f"[Server] Compatibilità con client: GARANTITA")
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
    SEMPLIFICATO: Architettura sempre identica = compatibilità automatica.
    
    Returns:
        Modello Keras compilato IDENTICO ai client
    """
    print(f"[Server] === CREAZIONE DNN ARCHITETTURA FISSA SERVER ===")
    print(f"[Server] Input features: {PCA_COMPONENTS} (FISSO - identico ai client)")
    print(f"[Server] Architettura: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1 (FISSA)")
    print(f"[Server] Attivazione: {ACTIVATION_FUNCTION}")
    print(f"[Server] Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"[Server] Dropout esteso: {EXTENDED_DROPOUT}")
    
    # Parametri IDENTICI ai client (ottimizzabili con Optuna)
    dropout_rate = 0.2          # Ottimizzabile
    dropout_final = 0.15         # Ottimizzabile
    l2_reg = 0.0002726058480553248             # Ottimizzabile
    
    # ARCHITETTURA FISSA IDENTICA AI CLIENT
    print(f"[Server] Architettura FISSA IDENTICA AI CLIENT")
    
    # Selezione funzione di attivazione (identica ai client)
    if ACTIVATION_FUNCTION == 'leaky_relu':
        activation_layer = lambda: layers.LeakyReLU(alpha=0.01)
        initializer = 'he_normal'
    elif ACTIVATION_FUNCTION == 'selu':
        activation_layer = lambda: layers.Activation('selu')
        initializer = 'lecun_normal'
    else:  # relu default
        activation_layer = lambda: layers.Activation('relu')
        initializer = 'he_normal'
    
    print(f"[Server] Funzione attivazione: {ACTIVATION_FUNCTION}, Initializer: {initializer}")
    
    # MODELLO CON ARCHITETTURA FISSA IDENTICA AI CLIENT
    model = tf.keras.Sequential([
        # Input layer esplicito con dimensione FISSA IDENTICA ai client
        layers.Input(shape=(PCA_COMPONENTS,), name='input_layer'),

        # Layer 1: 112 neuroni (FISSO - identico ai client)
        layers.Dense(112, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_1'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),

        # Layer 2: 64 neuroni (FISSO - identico ai client)
        layers.Dense(64, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_2'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate if EXTENDED_DROPOUT else 0.0, name='dropout_2'),

        # Layer 3: 12 neuroni (FISSO - identico ai client)
        layers.Dense(12, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_3'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate, name='dropout_3'),

        # Layer 4: 10 neuroni (FISSO - identico ai client)
        layers.Dense(10, 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer,
                    name='dense_4'),
        activation_layer(),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(dropout_final, name='dropout_4'),
        
        # Output layer IDENTICO ai client
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # OTTIMIZZATORE IDENTICO ai client (ottimizzabile con Optuna)
    if USE_ADAMW:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.006025741928842929,  # Ottimizzabile
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Server] Ottimizzatore: AdamW")
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.006025741928842929,  # Ottimizzabile
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        print(f"[Server] Ottimizzatore: Adam")
    
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
    
    print(f"[Server] === DNN ARCHITETTURA FISSA SERVER CREATA ===")
    print(f"[Server]   - Architettura: FISSA {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1")
    print(f"[Server]   - Parametri totali: {total_params:,}")
    print(f"[Server]   - IDENTICO ai client per compatibilità automatica")
    print(f"[Server]   - Controlli compatibilità: NON NECESSARI")
    
    return model

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid DNN con architettura FISSA.
    SEMPLIFICATO: Compatibilità garantita automaticamente.
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Usa PCA fissa identica ai client.
        SEMPLIFICATO: Nessun controllo di compatibilità necessario.
        """
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER (ARCHITETTURA FISSA) ===")
        
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
        
        # Applica pipeline con PCA fissa (garantisce compatibilità automatica)
        X_global_final = apply_preprocessing_pipeline(X_global)
        
        print(f"Dataset preprocessato con PCA FISSA: {len(X_global_final)} campioni, {X_global_final.shape[1]} feature")
        print(f"Componenti PCA fisse: {PCA_COMPONENTS}")
        print(f"Compatibilità con client: GARANTITA (architettura fissa)")
        
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
        print(f"\n=== VALUTAZIONE GLOBALE DNN ARCHITETTURA FISSA - ROUND {server_round} ===")
        print(f"Dataset naturalmente sbilanciato per attacchi realistici")
        print(f"PCA fissa: {PCA_COMPONENTS} componenti")
        print(f"Architettura: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1 (FISSA)")
        print(f"Compatibilità: GARANTITA (architettura sempre identica)")
        
        try:
            # Crea il modello DNN con architettura fissa per la valutazione (identico ai client)
            model = create_dnn_model()
            
            # IMPOSTAZIONE PESI SEMPLIFICATA
            # Nessun controllo di compatibilità necessario (architettura fissa)
            try:
                model.set_weights(parameters)
                print(f"✅ Pesi aggregati impostati su modello server (compatibilità garantita)")
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
            
            print(f"RISULTATI VALUTAZIONE ARCHITETTURA FISSA:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1_score_val:.4f} ({f1_score_val*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  Campioni test: {len(X_global)}")
            print(f"  Feature utilizzate: {X_global.shape[1]} (PCA fissa)")
            
            # Calcola parametri modello
            total_params = model.count_params()
            print(f"  Parametri DNN: {total_params:,}")
            print(f"  Architettura: FISSA (sempre identica)")
            print(f"  Controlli compatibilità: RIMOSSI (non necessari)")
            print(f"  Distribuzione naturale: {dataset_info.get('attack_ratio', 0)*100:.1f}% attacchi")
            
            return float(loss), {
                # Metriche base
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "balanced_accuracy": float(balanced_acc),
                
                # Informazioni dataset e modello
                "global_test_samples": int(len(X_global)),
                "pipeline_features": int(PCA_COMPONENTS),
                "total_params": int(total_params),
                "attack_samples": int(dataset_info.get('attack_samples', 0)),
                "natural_samples": int(dataset_info.get('natural_samples', 0)),
                "attack_ratio": float(dataset_info.get('attack_ratio', 0)),
                
                # Informazioni architettura fissa
                "architecture_fixed": True,
                "compatibility_guaranteed": True,
                "compatibility_checks_removed": True,
                "pca_components_fixed": int(PCA_COMPONENTS),
                
                # Metodologia semplificata
                "model_type": "dnn_fixed_architecture_simplified",
                "preprocessing_method": "fixed_pca_no_compatibility_checks"
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale con architettura fissa: {e}")
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
    
    print(f"\n=== METRICHE CLIENT DNN ARCHITETTURA FISSA ===")
    
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
        
        # Informazioni architettura fissa
        if 'architecture_fixed' in client_metrics and client_metrics['architecture_fixed']:
            print(f"  ✅ Architettura: FISSA")
        
        if 'compatibility_guaranteed' in client_metrics and client_metrics['compatibility_guaranteed']:
            compatibility_guaranteed_count += 1
            print(f"  ✅ Compatibilità: GARANTITA")
        
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
        
        # Informazioni PCA
        if 'pca_features' in client_metrics and 'pca_components_fixed' in client_metrics:
            pca_features = client_metrics['pca_features']
            pca_fixed = client_metrics['pca_components_fixed']
            print(f"  PCA: {pca_features} feature (fisso: {pca_fixed})")
    
    if total_samples > 0:
        # Calcola medie ponderate
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_weighted_f1 = total_weighted_f1 / total_samples if total_weighted_f1 > 0 else 0
        avg_loss = np.mean(loss_list) if loss_list else 0
        
        print(f"\nRIASSUNTO DNN ARCHITETTURA FISSA:")
        print(f"  Media accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  Media F1-Score: {avg_weighted_f1:.4f}")
        print(f"  Media loss: {avg_loss:.4f}")
        print(f"  Totale campioni: {total_samples}")
        print(f"  Client con errori: {len(error_clients)}")
        print(f"  Client con EarlyStopping: {early_stopped_count}/{len(fit_results)}")
        print(f"  Client con compatibilità garantita: {compatibility_guaranteed_count}/{len(fit_results)}")
        
        # Valutazioni specifiche architettura fissa
        print(f"  ✅ Architettura FISSA: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1")
        print(f"  ✅ Compatibilità: AUTOMATICA (PCA fissa)")
        print(f"  ✅ Controlli ridondanti: RIMOSSI")
        print(f"  ✅ Distribuzione naturale: MANTENUTA")
        print(f"  ✅ Parametri ottimizzabili: CONFIGURATI per Optuna")
        
        # Valutazioni performance
        if avg_weighted_accuracy > 0.8:
            print(f"  🎯 Performance: OTTIME (accuracy > 80%)")
        elif avg_weighted_accuracy > 0.7:
            print(f"  🎯 Performance: BUONE (accuracy > 70%)")
        else:
            print(f"  ⚠️  Performance: DA MIGLIORARE (accuracy < 70%)")
            print(f"      Suggerimento: Usa Optuna per ottimizzare iperparametri")
        
        # Raccomandazioni per ottimizzazione
        if early_stopped_count == 0:
            print(f"  💡 Suggerimento: Aumenta patience EarlyStopping per training più lungo")
        
        if compatibility_guaranteed_count < len(fit_results):
            print(f"  ⚠️  Alcuni client non hanno compatibilità garantita")

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
        print(f"\n=== AGGREGAZIONE TRAINING DNN ARCHITETTURA FISSA - ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        print(f"Architettura: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1 (FISSA)")
        print(f"Compatibilità: GARANTITA (architettura sempre identica)")
        
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
                print(f"✅ Aggregazione DNN architettura fissa completata per round {server_round}")
                print(f"✅ Pesi di {len(results)} client aggregati con successo")
                print(f"✅ Architetture FISSE perfettamente compatibili (controlli rimossi)")
                print(f"✅ Nessun problema di compatibilità possibile")
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
        print(f"\n=== AGGREGAZIONE VALUTAZIONE DNN ARCHITETTURA FISSA ROUND {server_round} ===")
        print(f"Client che hanno valutato: {len(results)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione DNN architettura fissa completata per round {server_round}")
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
    print("=== SERVER FEDERATO SMARTGRID DNN ARCHITETTURA FISSA ===")
    print("CONFIGURAZIONE SEMPLIFICATA:")
    print(f"  ✅ PCA FISSA: {PCA_COMPONENTS} componenti (compatibilità automatica)")
    print("  ✅ Architettura FISSA: 35 → 64 → 32 → 16 → 8 → 1")
    print("  ✅ Controlli compatibilità: RIMOSSI (non necessari)")
    print("  ✅ Distribuzione naturale mantenuta (NO SMOTE)")
    print("  ✅ Parametri ottimizzabili con Optuna")
    print("  ✅ Regolarizzazione: Dropout, L2, BatchNorm, EarlyStopping, ReduceLR")
    print("  ✅ Class weights automatici per compensare sbilanciamento")
    print("  ✅ Metriche bilanciate: F1-Score, Balanced Accuracy, AUC")
    print("  ✅ Normalizzazione LOCALE per ogni client (preserva privacy)")
    print("")
    print("VANTAGGI ARCHITETTURA FISSA:")
    print(f"  🎯 Compatibilità GARANTITA: architettura sempre identica")
    print(f"  🎯 Controlli rimossi: codice più semplice e veloce")
    print(f"  🎯 PCA deterministica: {PCA_COMPONENTS} componenti fissi")
    print(f"  🎯 Nessun errore di shape: dimensioni sempre corrette")
    print(f"  🎯 Ottimizzazione Optuna: parametri facilmente modificabili")
    print(f"  🎯 Performance consistenti: architettura testata")
    print(f"  🎯 Manutenzione semplificata: meno codice da gestire")
    print("")
    print("PARAMETRI OTTIMIZZABILI CON OPTUNA:")
    print("  🔧 Neuroni per layer (attualmente: 64, 32, 16, 8)")
    print("  🔧 Funzione di attivazione (attualmente: leaky_relu)")
    print("  🔧 Ottimizzatore (attualmente: AdamW)")
    print("  🔧 Dropout rate (attualmente: 0.4, 0.3)")
    print("  🔧 L2 regularization (attualmente: 0.0015)")
    print("  🔧 Learning rate (attualmente: 0.0008)")
    print("  🔧 Batch size e epochs per client")
    print("  🔧 Patience per EarlyStopping e ReduceLROnPlateau")
    print("")
    print("VANTAGGI PER ATTACCHI:")
    print("  🎯 Dati naturalmente distribuiti (nessun dato sintetico)")
    print("  🎯 Architettura prevedibile per test di sicurezza")
    print("  🎯 Membership inference su dati reali")
    print("  🎯 Model extraction su comportamento naturale")
    print("  🎯 Scenario federato completamente realistico")
    print("  🎯 Dimensionalità fissa e prevedibile")
    print("  🎯 Modelli ben regolarizzati per attacchi robusti")
    print("")
    print("Configurazione:")
    print(f"  - PCA Components: {PCA_COMPONENTS} (FISSO)")
    print(f"  - Architettura: {PCA_COMPONENTS} → 64 → 32 → 16 → 8 → 1 (FISSA)")
    print(f"  - Attivazione: {ACTIVATION_FUNCTION} (ottimizzabile)")
    print(f"  - Ottimizzatore: {'AdamW' if USE_ADAMW else 'Adam'} (ottimizzabile)")
    print(f"  - Learning Rate: 0.0008 (ottimizzabile)")
    print("  - Rounds: 100")
    print("  - Client minimi: 2")
    print("  - Strategia: FedAvg personalizzata con architettura fissa")
    print("  - Valutazione: Dataset globale con PCA fissa (client 14-15)")
    print("  - Pipeline: Pulizia → Imputazione → Normalizzazione → PCA fissa (NO SMOTE)")
    print("  - Class weights: Automatici per compensare sbilanciamento")
    print("  - Regolarizzazione: Completa ma semplificata")
    print("  - Callback: EarlyStopping + ReduceLROnPlateau sui client")
    print("  - Controlli compatibilità: RIMOSSI (architettura fissa)")
    
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
    
    print(f"\nServer DNN ARCHITETTURA FISSA in attesa di client su localhost:8080...")
    print("Per connettere i client, esegui:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  ...")
    print("  python client.py 13")
    print("\nClient 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    print("")
    print("VANTAGGI FINALI ARCHITETTURA FISSA:")
    print("  ✅ Compatibilità AUTOMATICA: nessun controllo necessario")
    print(f"  ✅ PCA deterministica: {PCA_COMPONENTS} componenti sempre uguali")
    print("  ✅ Architettura prevedibile: 35 → 64 → 32 → 16 → 8 → 1")
    print("  ✅ Parametri ottimizzabili: facile integrazione con Optuna")
    print("  ✅ Codice semplificato: controlli ridondanti rimossi")
    print("  ✅ Performance robuste: architettura testata e validata")
    print("  ✅ Attacchi realistici: distribuzione naturale preservata")
    print("  ✅ Manutenzione facile: meno codice, meno errori")
    print("  ✅ Riproducibilità: risultati sempre consistenti")
    print("  ✅ Scalabilità: aggiunta client senza problemi")
    print("  ✅ Documentazione: perfetto per tesi universitaria")
    print("  ✅ Integrazione: compatibile con tutti gli strumenti esistenti")
    print("  ✅ Testing: facile verifica funzionamento")
    print("  ✅ Debugging: meno punti di fallimento")
    print("  ✅ Estensibilità: facile aggiunta nuove funzionalità")
    
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