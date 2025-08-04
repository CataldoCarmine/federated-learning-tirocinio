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

def apply_server_preprocessing_pipeline(X_global, variance_threshold=0.95):
    """
    Applica la stessa pipeline di preprocessing del client sui dati globali del server.
    Pipeline: Imputazione → Normalizzazione → PCA
    
    Args:
        X_global: Dati globali del server
        variance_threshold: Soglia per PCA
    
    Returns:
        Tuple (X_global_final, pca_components)
    """
    print(f"=== APPLICAZIONE PIPELINE SERVER ===")
    
    # STEP 2-3: Pipeline di preprocessing (Imputazione + Normalizzazione)
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Applica preprocessing
    X_preprocessed = preprocessing_pipeline.fit_transform(X_global)
    print(f"  - Preprocessing applicato: Imputazione + Normalizzazione")
    print(f"  - Shape dopo preprocessing: {X_preprocessed.shape}")
    
    # STEP 5: PCA (nessun SMOTE sui dati di test)
    original_features = X_preprocessed.shape[1]
    
    # Analisi PCA per selezione automatica componenti
    pca_full = PCA()
    pca_full.fit(X_preprocessed)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    n_components = min(n_components, original_features, len(X_preprocessed))
    
    print(f"  - Feature originali: {original_features}")
    print(f"  - Componenti PCA selezionate: {n_components}")
    print(f"  - Varianza spiegata: {cumulative_variance[n_components-1]*100:.2f}%")
    print(f"  - Riduzione dimensionalità: {(1-n_components/original_features)*100:.1f}%")
    
    # Applica PCA
    pca_optimal = PCA(n_components=n_components)
    X_global_final = pca_optimal.fit_transform(X_preprocessed)
    
    print(f"  - Shape finale: {X_global_final.shape}")
    print("=" * 60)
    
    return X_global_final, n_components

def create_server_dnn_model(input_shape):
    """
    Crea il modello DNN ottimizzato per la valutazione del server.
    Identico ai modelli dei client per garantire compatibilità perfetta.
    
    Args:
        input_shape: Numero di feature in input (dopo PCA)
    
    Returns:
        Modello Keras compilato
    """
    # Configurazione IDENTICA ai client ottimizzati
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    # Architettura DNN IDENTICA per compatibilità perfetta
    model = tf.keras.Sequential([
        # Layer di input
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        # Primo blocco: Estrazione feature di alto livello
        layers.Dense(128, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Secondo blocco: Raffinamento pattern
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Terzo blocco: Specializzazione per sicurezza
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer='he_normal',
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate / 2, name='dropout_3'),
        
        # Layer finale: Classificazione binaria
        layers.Dense(1, activation='sigmoid', 
                    kernel_initializer='glorot_uniform',
                    name='output_layer')
    ])
    
    # Ottimizzatore IDENTICO ai client
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    # Compila il modello con le stesse metriche
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

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale ottimizzata per il server SmartGrid DNN.
    Usa la stessa pipeline di preprocessing dei client.
    
    Returns:
        Funzione di valutazione che può essere usata dal server
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Applica la stessa pipeline di preprocessing dei client.
        """
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER DNN OTTIMIZZATO ===")
        
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
        
        # Gestione preliminare valori infiniti
        X_global.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Applica la stessa pipeline dei client
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
        X_global = np.random.random((100, 64))
        y_global = np.random.randint(0, 2, 100)
        input_shape = 64
        print("Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round.
        Valuta il modello DNN aggregato sui dati di test globali con pipeline.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato
            config: Configurazione
        
        Returns:
            Tuple con (loss, metriche)
        """
        print(f"\n=== VALUTAZIONE GLOBALE TEST DNN OTTIMIZZATO - ROUND {server_round} ===")
        
        try:
            # Crea il modello DNN ottimizzato per la valutazione (identico ai client)
            model = create_server_dnn_model(input_shape)
            
            # Imposta i pesi aggregati
            model.set_weights(parameters)
            
            # Valutazione sul dataset test globale (con pipeline)
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Risultati valutazione test globale DNN ottimizzato (post-pipeline):")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            print(f"  - AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  - Campioni test utilizzati: {len(X_global)}")
            print(f"  - Feature utilizzate (post-pipeline): {X_global.shape[1]}")
            
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
            
            # Controllo qualità delle metriche
            if loss > 10.0:
                print(f"  - ATTENZIONE: Loss molto alta ({loss:.4f}) - possibile problema di convergenza")
            if accuracy < 0.5:
                print(f"  - ATTENZIONE: Accuracy molto bassa ({accuracy:.4f}) - performance sotto random")
            
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
                "model_type": "DNN_Optimized"
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {"accuracy": 0.0, "error": str(e), "global_test_samples": 0}
    
    return evaluate

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round di addestramento DNN ottimizzato.
    Include informazioni sulla pipeline di preprocessing applicata e modello DNN.
    
    Args:
        fit_results: Risultati dell'addestramento dai client
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ROUND DNN OTTIMIZZATO (CON PIPELINE) ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_original_features = 0
    total_pipeline_features = 0
    pipeline_info = None
    model_type = None
    total_local_epochs = 0
    total_batch_size = 0
    
    accuracy_list = []
    loss_list = []
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client DNN {i+1}:")
        print(f"  - Campioni training: {client_samples}")
        
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
            
            # Controllo qualità
            if loss > 5.0:
                print(f"    ⚠️  ATTENZIONE: Loss alta per client {i+1}")
        
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
        
        # Informazioni sulla pipeline di preprocessing
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
        
        print(f"\nRiassunto aggregato DNN ottimizzato:")
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
            
        # Controlli di qualità aggregati
        if avg_loss > 2.0:
            print(f"  ⚠️  ATTENZIONE: Loss media alta ({avg_loss:.4f}) - possibili problemi di convergenza")
        if avg_weighted_accuracy < 0.6:
            print(f"  ⚠️  ATTENZIONE: Accuracy media bassa ({avg_weighted_accuracy:.4f})")
        if std_accuracy > 0.3:
            print(f"  ⚠️  ATTENZIONE: Alta variabilità tra client (std: {std_accuracy:.4f})")
    
    print("=" * 50)

def print_client_evaluation_metrics(eval_results):
    """
    Stampa le metriche di valutazione dei client DNN ottimizzati.
    
    Args:
        eval_results: Risultati della valutazione dai client
    """
    if not eval_results:
        return
    
    print(f"\n=== METRICHE VALIDATION CLIENT DNN OTTIMIZZATO (POST-PIPELINE) ===")
    
    total_val_samples = 0
    total_weighted_val_accuracy = 0
    avg_metrics = {'precision': 0, 'recall': 0, 'f1_score': 0, 'auc': 0}
    val_accuracy_list = []
    val_loss_list = []
    
    for i, (client_proxy, eval_res) in enumerate(eval_results):
        val_samples = eval_res.num_examples
        eval_metrics = eval_res.metrics
        val_loss = eval_res.loss
        
        total_val_samples += val_samples
        val_loss_list.append(val_loss)
        
        print(f"Client DNN {i+1} Validation:")
        print(f"  - Campioni validation: {val_samples}")
        print(f"  - Val Loss: {val_loss:.4f}")
        
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
        
        print(f"\nRiassunto validation aggregato DNN ottimizzato:")
        print(f"  - Media pesata validation accuracy: {avg_weighted_val_accuracy:.4f}")
        print(f"  - Media validation loss: {avg_val_loss:.4f}")
        print(f"  - Std validation accuracy: {std_val_accuracy:.4f}")
        print(f"  - Media validation precision: {avg_metrics['precision']/num_clients:.4f}")
        print(f"  - Media validation recall: {avg_metrics['recall']/num_clients:.4f}")
        print(f"  - Media validation F1-Score: {avg_metrics['f1_score']/num_clients:.4f}")
        print(f"  - Media validation AUC: {avg_metrics['auc']/num_clients:.4f}")
        print(f"  - Totale campioni validation: {total_val_samples}")
        
        # Controlli di qualità validation
        if avg_val_loss > 2.0:
            print(f"  ⚠️  ATTENZIONE: Validation loss alta ({avg_val_loss:.4f})")
        if avg_weighted_val_accuracy < 0.6:
            print(f"  ⚠️  ATTENZIONE: Validation accuracy bassa ({avg_weighted_val_accuracy:.4f})")
    
    print("=" * 50)

class SmartGridDNNFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata ottimizzata per SmartGrid DNN con logging migliorato e monitoraggio qualità.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento DNN ottimizzato e stampa metriche dettagliate.
        """
        print(f"\n=== AGGREGAZIONE TRAINING DNN OTTIMIZZATO - ROUND {server_round} ===")
        print(f"Client DNN partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Stampa metriche dei client DNN ottimizzati (include controlli qualità)
        print_client_metrics(results)
        
        # Chiama l'aggregazione standard
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            print(f"✅ Aggregazione DNN ottimizzata completata per round {server_round}")
            print(f"Pesi di {len(results)} client DNN aggregati con successo")
        else:
            print(f"❌ ATTENZIONE: Aggregazione DNN fallita per round {server_round}")
        
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione DNN ottimizzata e stampa metriche dettagliate.
        """
        print(f"\n=== AGGREGAZIONE VALUTAZIONE DNN OTTIMIZZATO ROUND {server_round} ===")
        print(f"Client DNN che hanno valutato: {len(results)}")
        print(f"Client falliti nella valutazione: {len(failures)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Stampa metriche di valutazione dei client DNN ottimizzati
        print_client_evaluation_metrics(results)
        
        # Chiama l'aggregazione standard
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        print("=" * 50)
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato DNN ottimizzato con pipeline corretta.
    """
    print("=== AVVIO SERVER FEDERATO SMARTGRID DNN OTTIMIZZATO CON PIPELINE CORRETTA ===")
    print("Configurazione:")
    print("  - Numero di round: 200")
    print("  - Client minimi per training: 2")
    print("  - Client minimi per valutazione: 2")
    print("  - Client minimi disponibili: 2")
    print("  - Strategia: FedAvg personalizzata ottimizzata per DNN")
    print("  - Valutazione: Dataset globale DNN ottimizzato con pipeline (client 14-15)")
    print("  - Client training: Usano train (70%) + validation (30%) locale")
    print("  - Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA")
    print("  - Modello: Deep Neural Network ottimizzata (3 layer nascosti: 128→64→32)")
    print("  - Regolarizzazione: Dropout 0.2 + L2 0.0001 + BatchNormalization")
    print("  - Ottimizzazioni: Learning Rate 0.0001 + Gradient Clipping 1.0")
    print("  - Training locale: 3 epoche per round, batch size 16")
    print("  - Riduzione dimensionalità: PCA con 95% varianza cumulativa")
    print("  - Monitoraggio: Controlli qualità automatici su loss e accuracy")
    print("=" * 90)
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=200)
    
    # Strategia Federated Averaging personalizzata ottimizzata per DNN
    strategy = SmartGridDNNFedAvg(
        fraction_fit=1.0,                    # Usa tutti i client disponibili per training
        fraction_evaluate=1.0,               # Usa tutti i client disponibili per valutazione
        min_fit_clients=2,                   # Numero minimo di client per iniziare training
        min_evaluate_clients=2,              # Numero minimo di client per valutazione
        min_available_clients=2,             # Numero minimo di client connessi
        evaluate_fn=get_smartgrid_evaluate_fn()  # Valutazione globale DNN ottimizzata con pipeline
    )
    
    print("Server DNN ottimizzato in attesa di client...")
    print("Per connettere i client DNN ottimizzati, esegui in terminali separati:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  python client.py 3")
    print("  ...")
    print("  python client.py 13")
    print("\nNOTA: Usa client ID 1-13 per training federato DNN ottimizzato")
    print("      Client 14-15 sono riservati per valutazione globale")
    print("      Ogni client applicherà la pipeline corretta e addestrerà DNN ottimizzata:")
    print("      Split → Imputazione → Normalizzazione → SMOTE → PCA → DNN Training Ottimizzato")
    print("      Ogni client DNN eseguirà 3 epoche locali per round federato con batch size 16")
    print("      Learning rate 0.0001 con gradient clipping per stabilità")
    print("\nIl training federato DNN ottimizzato inizierà quando almeno 2 client saranno connessi.")
    print("Il sistema include controlli automatici di qualità per monitorare le performance.")
    print("=" * 90)
    sys.stdout.flush()
    
    try:
        # Avvia il server
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"Errore durante l'avvio del server DNN ottimizzato: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()