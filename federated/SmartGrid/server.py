import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os
import warnings

def clean_data_for_pca_server(X):
    """
    CORREZIONE PROBLEMA 1: Pulizia robusta dei dati per prevenire problemi numerici in PCA.
    Identica alla funzione client per consistenza.
    
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

def ensure_numerical_stability_server(X, stage_name):
    """
    CORREZIONE PROBLEMA 1: Assicura stabilità numerica rimuovendo inf, nan e valori estremi.
    Identica alla funzione client per consistenza.
    
    Args:
        X: Array dei dati
        stage_name: Nome dello stage per logging
    
    Returns:
        Array numericamente stabile
    """
    print(f"  - Controllo stabilità numerica server ({stage_name})...")
    
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

def apply_stable_pca_server(X_preprocessed, variance_threshold=0.95):
    """
    CORREZIONE PROBLEMA 1: Applica PCA con controlli di stabilità numerica.
    Identica alla logica client per consistenza.
    
    Args:
        X_preprocessed: Dati preprocessati del server
        variance_threshold: Soglia di varianza cumulativa
    
    Returns:
        Tuple (X_pca, n_components_selected)
    """
    print(f"=== APPLICAZIONE PCA STABILE SERVER ===")
    
    original_features = X_preprocessed.shape[1]
    print(f"  - Feature originali: {original_features}")
    print(f"  - Soglia varianza cumulativa: {variance_threshold*100:.1f}%")
    
    # Assicura stabilità numerica pre-PCA
    X_stable = ensure_numerical_stability_server(X_preprocessed, "pre-PCA server")
    
    try:
        # Sopprimi warning numerici temporaneamente per gestirli noi
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Prima esecuzione: PCA completa per analizzare la varianza
            pca_full = PCA()
            pca_full.fit(X_stable)
            
            # Controlla se explained_variance_ratio_ contiene valori validi
            if np.any(np.isnan(pca_full.explained_variance_ratio_)) or np.any(np.isinf(pca_full.explained_variance_ratio_)):
                raise ValueError("PCA server ha prodotto explained_variance_ratio_ non validi")
            
            # Calcola varianza cumulativa
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Trova il numero di componenti necessarie per raggiungere la soglia
            n_components_selected = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # Assicurati che il numero di componenti sia valido e ragionevole
            n_components_selected = min(n_components_selected, original_features, len(X_stable))
            n_components_selected = max(n_components_selected, min(10, original_features))
            
            print(f"  - Componenti selezionate: {n_components_selected}")
            print(f"  - Varianza spiegata: {cumulative_variance[n_components_selected-1]*100:.2f}%")
            print(f"  - Riduzione dimensionalità: {original_features} → {n_components_selected}")
            
            # Seconda esecuzione: PCA con numero ottimale di componenti
            pca_optimal = PCA(n_components=n_components_selected)
            
            # Fit e transform con controllo degli output
            X_pca = pca_optimal.fit_transform(X_stable)
            
            # Controllo finale degli output PCA
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError("PCA server ha prodotto output con NaN o inf")
            
            print(f"  - PCA server applicato con successo")
            return X_pca, n_components_selected
            
    except Exception as e:
        print(f"  - Errore PCA server: {e}")
        print(f"  - Fallback server: riduzione semplice alle prime {min(50, original_features)} feature")
        
        # Fallback: usa solo le prime N feature più stabili
        n_components_fallback = min(50, original_features)
        
        # Seleziona feature con varianza più alta (più stabili numericamente)
        feature_vars = np.var(X_stable, axis=0)
        # Sostituisci eventuali NaN nelle varianze con 0
        feature_vars = np.where(np.isnan(feature_vars), 0, feature_vars)
        
        top_features = np.argsort(feature_vars)[-n_components_fallback:]
        
        X_fallback = X_stable[:, top_features]
        
        print(f"  - Fallback server applicato: {n_components_fallback} feature selezionate")
        return X_fallback, n_components_fallback

def apply_server_preprocessing_pipeline(X_global, variance_threshold=0.95):
    """
    Applica la stessa pipeline di preprocessing dei client sui dati globali del server.
    Include le stesse correzioni per stabilità numerica.
    
    Args:
        X_global: Dati globali del server
        variance_threshold: Soglia per PCA
    
    Returns:
        Tuple (X_global_final, pca_components)
    """
    print(f"=== APPLICAZIONE PIPELINE SERVER ===")
    
    # STEP 2-3: Pipeline di preprocessing (identica ai client)
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # CORREZIONE PROBLEMA 1: Stessa pulizia dei client
    print(f"  - Applicazione pulizia robusta dei dati...")
    X_global_cleaned = clean_data_for_pca_server(X_global)
    
    # Applica preprocessing
    X_preprocessed = preprocessing_pipeline.fit_transform(X_global_cleaned)
    print(f"  - Preprocessing applicato: Imputazione + Normalizzazione")
    print(f"  - Shape dopo preprocessing: {X_preprocessed.shape}")
    
    # CORREZIONE PROBLEMA 1: Controllo stabilità post-preprocessing
    X_preprocessed = ensure_numerical_stability_server(X_preprocessed, "server preprocessing")
    
    # STEP 5: PCA stabile (identico ai client, nessun SMOTE sui dati di test)
    X_global_final, n_components = apply_stable_pca_server(X_preprocessed, variance_threshold)
    
    print(f"  - Shape finale server: {X_global_final.shape}")
    print("=" * 60)
    
    return X_global_final, n_components

def create_server_dnn_model(input_shape):
    """
    CORREZIONE PROBLEMA 2: Crea il modello DNN per il server IDENTICO ai client.
    Deve avere esattamente la stessa architettura per garantire compatibilità.
    
    Args:
        input_shape: Numero di feature in input (dopo PCA)
    
    Returns:
        Modello Keras compilato
    """
    print(f"=== CREAZIONE MODELLO DNN SERVER IDENTICO AI CLIENT ===")
    
    # CORREZIONE PROBLEMA 2: Configurazione IDENTICA ai client
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    # CORREZIONE PROBLEMA 2: Architettura IDENTICA ai client (no Input layer esplicito)
    model = tf.keras.Sequential([
        # CORREZIONE: Stesso primo layer dei client con input_shape
        layers.Dense(128, 
                    activation='relu',
                    input_shape=(input_shape,),  # IDENTICO ai client
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Secondo blocco - IDENTICO ai client
        layers.Dense(64, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Terzo blocco - IDENTICO ai client
        layers.Dense(32, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        # Layer finale - IDENTICO ai client
        layers.Dense(1, 
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # CORREZIONE PROBLEMA 2: Ottimizzatore IDENTICO ai client
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
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
    
    # CORREZIONE PROBLEMA 2: Verifica che sia identico ai client
    print(f"  - Modello DNN server creato IDENTICO ai client")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Architettura: Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    print(f"  - Numero di pesi: {len(model.get_weights())}")
    print(f"  - Parametri totali: {model.count_params():,}")
    
    # Debug: verifica identità con client
    print(f"  - Layer del modello server:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            print(f"    - {i}: {layer.name} - {layer.units} unità")
        else:
            print(f"    - {i}: {layer.name}")
    
    print("=" * 60)
    
    return model

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid DNN.
    Include le stesse correzioni per stabilità numerica dei client.
    
    Returns:
        Funzione di valutazione che può essere usata dal server
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Applica la stessa pipeline dei client con correzioni per stabilità.
        """
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER ===")
        
        # Directory in cui si trova questo script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Costruzione robusta dei path ai file CSV
        test_clients = [14, 15]
        df_list = []

        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"  - Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"  - File data{client_id}.csv non trovato, saltato")
                continue

        if not df_list:
            print("  - ATTENZIONE: Nessun file di test globale trovato!")
            print("  - Usando il primo client disponibile come fallback...")

            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(200, len(df_fallback)), random_state=42)]
                print(f"  - Usato fallback con {len(df_list[0])} campioni da data1.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per valutazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        print(f"  - Dataset test globale: {len(df_global)} campioni, {X_global.shape[1]} feature originali")
        print(f"  - Distribuzione originale:")
        print(f"    - Natural: {(y_global == 0).sum()} ({(y_global == 0).mean()*100:.2f}%)")
        print(f"    - Attack: {(y_global == 1).sum()} ({(y_global == 1).mean()*100:.2f}%)")
        
        # Applica la stessa pipeline dei client con correzioni
        X_global_final, pca_components = apply_server_preprocessing_pipeline(
            X_global, 
            variance_threshold=0.95
        )
        
        print(f"  - Dataset test globale preprocessato:")
        print(f"    - Campioni finali: {len(X_global_final)}")
        print(f"    - Feature dopo pipeline: {X_global_final.shape[1]}")
        print(f"    - Distribuzione mantenuta sbilanciata per valutazione realistica")
        print("=" * 60)
        
        return X_global_final, y_global, pca_components
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, input_shape = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi per evitare crash
        X_global = np.random.random((100, 50))
        y_global = np.random.randint(0, 2, 100)
        input_shape = 50
        print("Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round.
        Include verifiche di compatibilità pesi.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato
            config: Configurazione
        
        Returns:
            Tuple con (loss, metriche)
        """
        print(f"\n=== VALUTAZIONE GLOBALE TEST ROUND {server_round} ===")
        
        try:
            # CORREZIONE PROBLEMA 2: Crea il modello IDENTICO ai client
            model = create_server_dnn_model(input_shape)
            
            # CORREZIONE PROBLEMA 2: Verifica compatibilità pesi
            current_weights = model.get_weights()
            print(f"Server: pesi ricevuti {len(parameters)}, pesi modello {len(current_weights)}")
            
            if len(parameters) != len(current_weights):
                print(f"ERRORE: Incompatibilità pesi server!")
                print(f"Ricevuti: {len(parameters)}, Attesi: {len(current_weights)}")
                
                # Debug dettagliato
                print(f"Forme pesi ricevuti:")
                for i, w in enumerate(parameters):
                    print(f"  {i}: {w.shape}")
                print(f"Forme pesi modello server:")
                for i, w in enumerate(current_weights):
                    print(f"  {i}: {w.shape}")
                
                return 1.0, {"accuracy": 0.0, "error": "server_weight_mismatch", "global_test_samples": 0}
            
            # Imposta i pesi aggregati
            model.set_weights(parameters)
            
            # Valutazione sul dataset test globale
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Risultati valutazione test globale (con correzioni):")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            print(f"  - AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  - Campioni test utilizzati: {len(X_global)}")
            print(f"  - Feature utilizzate: {X_global.shape[1]}")
            
            # Predizioni per analisi dettagliata
            predictions_prob = model.predict(X_global, verbose=0)
            predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
            
            # Matrice di confusione
            tn = np.sum((y_global == 0) & (predictions_binary == 0))
            fp = np.sum((y_global == 0) & (predictions_binary == 1))
            fn = np.sum((y_global == 1) & (predictions_binary == 0))
            tp = np.sum((y_global == 1) & (predictions_binary == 1))
            
            print(f"  - Matrice confusione: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            # Accuratezza per classe e metriche di sicurezza
            if tn + fp > 0:
                natural_accuracy = tn / (tn + fp)
                print(f"  - Accuracy classe Natural: {natural_accuracy:.4f}")
            
            if tp + fn > 0:
                attack_accuracy = tp / (tp + fn)
                print(f"  - Accuracy classe Attack: {attack_accuracy:.4f}")
            
            # Metriche di sicurezza aggiuntive
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f"  - Specificity: {specificity:.4f}")
            print(f"  - False Positive Rate: {fpr:.4f}")
            print(f"  - False Negative Rate: {fnr:.4f}")
            
            # Controlli qualità
            if loss > 10.0:
                print(f"  - ⚠️  ATTENZIONE: Loss molto alta ({loss:.4f})")
            if accuracy < 0.5:
                print(f"  - ⚠️  ATTENZIONE: Accuracy sotto random ({accuracy:.4f})")
            if accuracy > 0.7:
                print(f"  - ✅  Performance buone (accuracy {accuracy:.4f})")
            
            print("=" * 60)
            sys.stdout.flush()
            
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
                "true_positives": int(tp),
                "model_type": "DNN_Stable_Compatible"
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {"accuracy": 0.0, "error": str(e), "global_test_samples": 0}
    
    return evaluate

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round di addestramento.
    Include controlli per le correzioni applicate.
    
    Args:
        fit_results: Risultati dell'addestramento dai client
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ROUND (CON CORREZIONI) ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_original_features = 0
    total_pipeline_features = 0
    pipeline_info = None
    model_type = None
    total_local_epochs = 0
    total_batch_size = 0
    error_clients = []
    
    accuracy_list = []
    loss_list = []
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client DNN {i+1}:")
        print(f"  - Campioni training: {client_samples}")
        
        # CORREZIONE PROBLEMA 2: Controlla se ci sono stati errori di compatibilità
        if 'error' in client_metrics:
            error_clients.append(i+1)
            print(f"  - ❌ ERRORE: {client_metrics['error']}")
            continue
        
        # Metriche di training standard
        if 'train_accuracy' in client_metrics:
            accuracy = client_metrics['train_accuracy']
            total_weighted_accuracy += accuracy * client_samples
            accuracy_list.append(accuracy)
            print(f"  - Train Accuracy: {accuracy:.4f}")
        
        if 'train_loss' in client_metrics:
            loss = client_metrics['train_loss']
            loss_list.append(loss)
            print(f"  - Train Loss: {loss:.4f}")
            
            # Controllo qualità con correzioni
            if loss > 5.0:
                print(f"    ⚠️  ATTENZIONE: Loss alta per client {i+1}")
            elif loss < 1.0:
                print(f"    ✅  Loss accettabile per client {i+1}")
        
        if 'train_precision' in client_metrics:
            print(f"  - Train Precision: {client_metrics['train_precision']:.4f}")
        
        if 'train_recall' in client_metrics:
            print(f"  - Train Recall: {client_metrics['train_recall']:.4f}")
        
        # Metriche aggiuntive DNN
        if 'train_f1_score' in client_metrics:
            print(f"  - Train F1-Score: {client_metrics['train_f1_score']:.4f}")
        
        if 'train_auc' in client_metrics:
            print(f"  - Train AUC: {client_metrics['train_auc']:.4f}")
        
        if 'local_epochs' in client_metrics:
            epochs = client_metrics['local_epochs']
            total_local_epochs = epochs
            print(f"  - Epoche locali: {epochs}")
        
        if 'batch_size' in client_metrics:
            batch_size = client_metrics['batch_size']
            total_batch_size = batch_size
            print(f"  - Batch size: {batch_size}")
        
        # CORREZIONE PROBLEMA 2: Informazioni sulla compatibilità
        if 'weights_count' in client_metrics:
            weights_count = client_metrics['weights_count']
            print(f"  - Numero pesi: {weights_count}")
        
        # Informazioni sulla pipeline con correzioni
        if 'original_features' in client_metrics and 'pca_features' in client_metrics:
            orig_feat = client_metrics['original_features']
            pipeline_feat = client_metrics['pca_features']
            total_original_features = orig_feat
            total_pipeline_features = pipeline_feat
            print(f"  - Feature originali: {orig_feat}")
            print(f"  - Feature post-pipeline: {pipeline_feat}")
            
        if 'pca_reduction' in client_metrics:
            print(f"  - Riduzione pipeline: {client_metrics['pca_reduction']:.1f}%")
        
        if 'pipeline_applied' in client_metrics:
            pipeline_info = client_metrics['pipeline_applied']
        
        if 'model_type' in client_metrics:
            model_type = client_metrics['model_type']
        
        # Mostra anche informazioni sui dati di validation locali
        if 'val_samples' in client_metrics:
            print(f"  - Campioni validation: {client_metrics['val_samples']}")
    
    if total_samples > 0:
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_loss = np.mean(loss_list) if loss_list else 0
        std_accuracy = np.std(accuracy_list) if accuracy_list else 0
        
        print(f"\nRiassunto aggregato DNN con correzioni:")
        print(f"  - Client con errori: {len(error_clients)} / {len(fit_results)}")
        if error_clients:
            print(f"  - Client in errore: {error_clients}")
        
        print(f"  - Media pesata accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  - Media loss: {avg_loss:.4f}")
        print(f"  - Std accuracy tra client: {std_accuracy:.4f}")
        print(f"  - Totale campioni training: {total_samples}")
        print(f"  - Epoche locali per client: {total_local_epochs}")
        print(f"  - Batch size: {total_batch_size}")
        
        if total_original_features > 0 and total_pipeline_features > 0:
            print(f"  - Riduzione dimensionalità comune: {total_original_features} → {total_pipeline_features}")
            print(f"  - Percentuale riduzione: {(1 - total_pipeline_features/total_original_features)*100:.1f}%")
        
        if pipeline_info:
            print(f"  - Pipeline applicata: {pipeline_info}")
        
        if model_type:
            print(f"  - Tipo modello: {model_type}")
            
        # Controlli di qualità aggregati con correzioni
        if avg_loss > 2.0:
            print(f"  ⚠️  ATTENZIONE: Loss media alta ({avg_loss:.4f}) - possibili problemi numerici")
        elif avg_loss < 1.0:
            print(f"  ✅  Loss media accettabile ({avg_loss:.4f})")
        
        if avg_weighted_accuracy < 0.6:
            print(f"  ⚠️  ATTENZIONE: Accuracy media bassa ({avg_weighted_accuracy:.4f})")
        elif avg_weighted_accuracy > 0.7:
            print(f"  ✅  Accuracy media buona ({avg_weighted_accuracy:.4f})")
        
        if std_accuracy > 0.3:
            print(f"  ⚠️  ATTENZIONE: Alta variabilità tra client (std: {std_accuracy:.4f})")
        else:
            print(f"  ✅  Variabilità tra client accettabile (std: {std_accuracy:.4f})")
    
    print("=" * 60)

def print_client_evaluation_metrics(eval_results):
    """
    Stampa le metriche di valutazione dei client DNN con correzioni.
    Include controlli di errore e compatibilità.
    
    Args:
        eval_results: Risultati della valutazione dai client
    """
    if not eval_results:
        return
    
    print(f"\n=== METRICHE VALIDATION CLIENT DNN (CON CORREZIONI) ===")
    
    total_val_samples = 0
    total_weighted_val_accuracy = 0
    val_accuracy_list = []
    val_loss_list = []
    eval_error_clients = []
    avg_metrics = {'precision': 0, 'recall': 0, 'f1_score': 0, 'auc': 0}
    
    for i, (client_proxy, eval_res) in enumerate(eval_results):
        val_samples = eval_res.num_examples
        eval_metrics = eval_res.metrics
        val_loss = eval_res.loss
        
        total_val_samples += val_samples
        val_loss_list.append(val_loss)
        
        print(f"Client DNN {i+1} Validation:")
        print(f"  - Campioni validation: {val_samples}")
        print(f"  - Val Loss: {val_loss:.4f}")
        
        # CORREZIONE PROBLEMA 2: Controlla se ci sono stati errori di compatibilità
        if 'error' in eval_metrics:
            eval_error_clients.append(i+1)
            print(f"  - ❌ ERRORE VALIDATION: {eval_metrics['error']}")
            continue
        
        if 'accuracy' in eval_metrics:
            val_accuracy = eval_metrics['accuracy']
            total_weighted_val_accuracy += val_accuracy * val_samples
            val_accuracy_list.append(val_accuracy)
            print(f"  - Val Accuracy: {val_accuracy:.4f}")
        
        if 'precision' in eval_metrics:
            precision = eval_metrics['precision']
            avg_metrics['precision'] += precision
            print(f"  - Val Precision: {precision:.4f}")
        
        if 'recall' in eval_metrics:
            recall = eval_metrics['recall']
            avg_metrics['recall'] += recall
            print(f"  - Val Recall: {recall:.4f}")
        
        if 'f1_score' in eval_metrics:
            f1 = eval_metrics['f1_score']
            avg_metrics['f1_score'] += f1
            print(f"  - Val F1-Score: {f1:.4f}")
        
        if 'auc' in eval_metrics:
            auc = eval_metrics['auc']
            avg_metrics['auc'] += auc
            print(f"  - Val AUC: {auc:.4f}")
    
    num_clients = len(eval_results)
    if total_val_samples > 0 and num_clients > 0:
        avg_weighted_val_accuracy = total_weighted_val_accuracy / total_val_samples
        avg_val_loss = np.mean(val_loss_list)
        std_val_accuracy = np.std(val_accuracy_list) if val_accuracy_list else 0
        
        print(f"\nRiassunto validation aggregato DNN con correzioni:")
        print(f"  - Client con errori validation: {len(eval_error_clients)} / {num_clients}")
        if eval_error_clients:
            print(f"  - Client in errore validation: {eval_error_clients}")
        
        print(f"  - Media pesata validation accuracy: {avg_weighted_val_accuracy:.4f}")
        print(f"  - Media validation loss: {avg_val_loss:.4f}")
        print(f"  - Std validation accuracy: {std_val_accuracy:.4f}")
        print(f"  - Media validation precision: {avg_metrics['precision']/num_clients:.4f}")
        print(f"  - Media validation recall: {avg_metrics['recall']/num_clients:.4f}")
        print(f"  - Media validation F1-Score: {avg_metrics['f1_score']/num_clients:.4f}")
        print(f"  - Media validation AUC: {avg_metrics['auc']/num_clients:.4f}")
        print(f"  - Totale campioni validation: {total_val_samples}")
        
        # Controlli di qualità validation con correzioni
        if avg_val_loss > 2.0:
            print(f"  ⚠️  ATTENZIONE: Validation loss alta ({avg_val_loss:.4f})")
        elif avg_val_loss < 1.0:
            print(f"  ✅  Validation loss accettabile ({avg_val_loss:.4f})")
        
        if avg_weighted_val_accuracy < 0.6:
            print(f"  ⚠️  ATTENZIONE: Validation accuracy bassa ({avg_weighted_val_accuracy:.4f})")
        elif avg_weighted_val_accuracy > 0.7:
            print(f"  ✅  Validation accuracy buona ({avg_weighted_val_accuracy:.4f})")
    
    print("=" * 60)

class SmartGridDNNFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid DNN con logging migliorato e monitoraggio qualità.
    Include gestione errori avanzata e controlli di compatibilità per le correzioni applicate.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento DNN e stampa metriche dettagliate.
        Include gestione errori e controlli di compatibilità per le correzioni.
        """
        print(f"\n=== AGGREGAZIONE TRAINING DNN CON CORREZIONI - ROUND {server_round} ===")
        print(f"Client DNN partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Controlla se ci sono risultati validi
        if not results:
            print("❌ ERRORE: Nessun client ha fornito risultati validi per l'aggregazione")
            return None
        
        # CORREZIONE PROBLEMA 2: Controllo preventivo compatibilità pesi
        print(f"Controllo compatibilità pesi tra client...")
        weight_lengths = []
        for i, (client_proxy, fit_res) in enumerate(results):
            if 'weights_count' in fit_res.metrics:
                weight_count = fit_res.metrics['weights_count']
                weight_lengths.append(weight_count)
                print(f"  - Client {i+1}: {weight_count} pesi")
        
        # Verifica che tutti i client abbiano lo stesso numero di pesi
        if weight_lengths and len(set(weight_lengths)) > 1:
            print(f"⚠️  ATTENZIONE: Incompatibilità nel numero di pesi tra client: {set(weight_lengths)}")
        elif weight_lengths:
            print(f"✅  Compatibilità pesi verificata: {weight_lengths[0]} pesi per client")
        
        # Stampa metriche dei client DNN (include controlli qualità e correzioni)
        print_client_metrics(results)
        
        # Chiama l'aggregazione standard
        try:
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione DNN con correzioni completata per round {server_round}")
                print(f"Pesi di {len(results)} client DNN aggregati con successo")
                
                # CORREZIONE PROBLEMA 1: Verifica i pesi aggregati per stabilità numerica
                aggregated_weights, _ = aggregated_result
                print(f"Pesi aggregati: {len(aggregated_weights)} array")
                
                # Controllo stabilità pesi aggregati
                has_nan = any(np.any(np.isnan(w)) for w in aggregated_weights)
                has_inf = any(np.any(np.isinf(w)) for w in aggregated_weights)
                
                if has_nan or has_inf:
                    print(f"⚠️  ATTENZIONE: Pesi aggregati contengono valori non validi!")
                    print(f"    - NaN: {has_nan}, Inf: {has_inf}")
                    
                    # Correggi i pesi aggregati se necessario
                    corrected_weights = []
                    for w in aggregated_weights:
                        w_corrected = np.where(np.isnan(w) | np.isinf(w), 0, w)
                        w_corrected = np.clip(w_corrected, -1e6, 1e6)
                        corrected_weights.append(w_corrected)
                    print(f"    - Pesi aggregati corretti per stabilità numerica")
                    aggregated_result = (corrected_weights, aggregated_result[1])
                else:
                    print(f"✅  Pesi aggregati numericamente stabili")
                
            else:
                print(f"❌ ATTENZIONE: Aggregazione DNN fallita per round {server_round}")
            
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione DNN e stampa metriche dettagliate.
        Include gestione errori avanzata per le correzioni.
        """
        print(f"\n=== AGGREGAZIONE VALUTAZIONE DNN CON CORREZIONI ROUND {server_round} ===")
        print(f"Client DNN che hanno valutato: {len(results)}")
        print(f"Client falliti nella valutazione: {len(failures)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Stampa metriche di valutazione dei client DNN (include controlli errore)
        print_client_evaluation_metrics(results)
        
        # Chiama l'aggregazione standard
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione completata per round {server_round}")
            else:
                print(f"⚠️  Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione valutazione: {e}")
            return None
        
        print("=" * 60)
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato DNN con correzioni per stabilità numerica e compatibilità.
    """
    print("=== AVVIO SERVER FEDERATO SMARTGRID DNN CON CORREZIONI ===")
    print("Configurazione:")
    print("  - Numero di round: 5")
    print("  - Client minimi per training: 2")
    print("  - Client minimi per valutazione: 2")
    print("  - Client minimi disponibili: 2")
    print("  - Strategia: FedAvg personalizzata con correzioni per DNN")
    print("  - Valutazione: Dataset globale DNN con pipeline stabile (client 14-15)")
    print("  - Client training: Usano train (70%) + validation (30%) locale")
    print("  - Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA Stabile")
    print("  - Modello: DNN con architettura IDENTICA client-server per compatibilità")
    print("  - Architettura: 3 layer nascosti (128→64→32→1) - FISSA per evitare errori")
    print("  - Regolarizzazione: Dropout 0.2 + L2 0.0001 + BatchNormalization")
    print("  - Ottimizzazioni: Learning Rate 0.0001 + Gradient Clipping 1.0")
    print("  - Training locale: 3 epoche per round, batch size 16")
    print("  - CORREZIONE PROBLEMA 1: Stabilità numerica PCA (pulizia, clipping, fallback)")
    print("  - CORREZIONE PROBLEMA 2: Compatibilità architettura (pesi identici)")
    print("  - Gestione errori: Controlli numerici + Compatibilità pesi + Fallback robusti")
    print("  - Monitoraggio: Controlli qualità automatici + Debug architettura + Logging esteso")
    print("=" * 100)
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Strategia Federated Averaging personalizzata con correzioni per DNN
    strategy = SmartGridDNNFedAvg(
        fraction_fit=1.0,                    # Usa tutti i client disponibili per training
        fraction_evaluate=1.0,               # Usa tutti i client disponibili per valutazione
        min_fit_clients=2,                   # Numero minimo di client per iniziare training
        min_evaluate_clients=2,              # Numero minimo di client per valutazione
        min_available_clients=2,             # Numero minimo di client connessi
        evaluate_fn=get_smartgrid_evaluate_fn()  # Valutazione globale DNN con pipeline stabile
    )
    
    print("Server DNN con correzioni in attesa di client...")
    print("Per connettere i client DNN corretti, esegui in terminali separati:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  python client.py 3")
    print("  ...")
    print("  python client.py 13")
    print("\nNOTA: Usa client ID 1-13 per training federato DNN con correzioni")
    print("      Client 14-15 sono riservati per valutazione globale")
    print("      Ogni client applicherà le correzioni e addestrerà DNN compatibile:")
    print("      Split → Imputazione → Normalizzazione → SMOTE → PCA Stabile → DNN Training")
    print("      Ogni client DNN eseguirà 3 epoche locali per round federato con batch size 16")
    print("      Learning rate 0.0001 con gradient clipping per stabilità numerica")
    print("      Controlli automatici di compatibilità architettura e stabilità numerica")
    print("      Gestione errori robusta con fallback per problemi PCA e incompatibilità pesi")
    print("\nIl training federato DNN con correzioni inizierà quando almeno 2 client saranno connessi.")
    print("Il sistema include controlli automatici completi per:")
    print("  - Problema 1: Stabilità numerica (NaN/inf detection, clipping, fallback PCA)")
    print("  - Problema 2: Compatibilità architettura (verifica pesi, debug forms)")
    print("  - Monitoraggio performance e qualità con alert automatici")
    print("=" * 100)
    sys.stdout.flush()
    
    try:
        # Avvia il server
        print("Avvio del server Flower con correzioni...")
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Errore durante l'avvio del server DNN con correzioni: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()