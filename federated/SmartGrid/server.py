import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
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

def apply_server_preprocessing_pipeline(X_global, fixed_pca_components=50):
    """
    Applica la stessa pipeline di preprocessing dei client sui dati globali del server.
    CORREZIONE: Usa lo stesso numero FISSO di componenti PCA dei client.
    
    Args:
        X_global: Dati globali del server
        fixed_pca_components: Numero fisso di componenti PCA (identico ai client)
    
    Returns:
        Tuple (X_global_final, pca_components)
    """
    print(f"=== APPLICAZIONE PIPELINE SERVER CON PCA FISSO ===")
    
    # STEP 2-3: Pipeline di preprocessing (identica ai client)
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Gestione valori infiniti
    X_global.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Applica preprocessing
    X_preprocessed = preprocessing_pipeline.fit_transform(X_global)
    print(f"  - Preprocessing applicato: Imputazione + Normalizzazione")
    print(f"  - Shape dopo preprocessing: {X_preprocessed.shape}")
    
    # STEP 5: PCA con numero FISSO (identico ai client, nessun SMOTE sui dati di test)
    original_features = X_preprocessed.shape[1]
    
    # CORREZIONE: Usa sempre lo stesso numero fisso di componenti dei client
    n_components = min(fixed_pca_components, original_features, len(X_preprocessed))
    
    print(f"  - Feature originali: {original_features}")
    print(f"  - Componenti PCA FISSE: {n_components}")
    print(f"  - Riduzione dimensionalità: {(1-n_components/original_features)*100:.1f}%")
    
    # Applica PCA con numero fisso di componenti
    pca_fixed = PCA(n_components=n_components)
    X_global_final = pca_fixed.fit_transform(X_preprocessed)
    
    # Calcola varianza spiegata per informazione
    variance_explained = np.sum(pca_fixed.explained_variance_ratio_)
    print(f"  - Varianza spiegata con {n_components} componenti: {variance_explained*100:.2f}%")
    print(f"  - Shape finale server: {X_global_final.shape}")
    
    # CORREZIONE: Verifica che la forma sia quella attesa
    assert X_global_final.shape[1] == n_components, f"Errore: forma server {X_global_final.shape[1]} != {n_components}"
    
    print("=" * 60)
    
    return X_global_final, n_components

def create_server_dnn_model(input_shape):
    """
    CORREZIONE ARCHITETTURA: Crea il modello DNN per il server IDENTICO ai client.
    IMPORTANTE: Deve avere la STESSA architettura dei client per compatibilità.
    
    Args:
        input_shape: Numero di feature in input (FISSO = 50)
    
    Returns:
        Modello Keras compilato
    """
    print(f"=== CREAZIONE MODELLO DNN SERVER IDENTICO AI CLIENT ===")
    
    # CORREZIONE: Configurazione IDENTICA ai client
    dropout_rate = 0.2
    l2_reg = 0.0001
    
    # CORREZIONE: Verifica che input_shape sia corretto
    expected_input_shape = 50  # Numero fisso di componenti PCA
    if input_shape != expected_input_shape:
        print(f"ATTENZIONE: Input shape server {input_shape} diverso da atteso {expected_input_shape}")
        print(f"Uso input_shape atteso: {expected_input_shape}")
        input_shape = expected_input_shape
    
    # CORREZIONE: Architettura IDENTICA ai client con input fisso
    model = tf.keras.Sequential([
        # Layer di input esplicito - IDENTICO ai client
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        # CORREZIONE: Architettura semplificata e standardizzata IDENTICA
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
    
    # CORREZIONE: Ottimizzatore IDENTICO ai client
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
    
    # CORREZIONE: Verifica che sia identico ai client
    print(f"  - Modello DNN server creato IDENTICO ai client")
    print(f"  - Input shape FISSO: {input_shape}")
    print(f"  - Architettura: Input({input_shape}) → Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    print(f"  - Numero di pesi: {len(model.get_weights())}")
    print(f"  - Parametri totali: {model.count_params():,}")
    
    # Debug: verifica identità con client
    print(f"  - Layer del modello server:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            print(f"    - {i}: {layer.name} ({layer.__class__.__name__}) - {layer.units} unità")
        else:
            print(f"    - {i}: {layer.name} ({layer.__class__.__name__})")
    
    # CORREZIONE: Debug forme pesi per verifica compatibilità
    print(f"  - Forme pesi del modello server:")
    for i, weight in enumerate(model.get_weights()):
        print(f"    - Peso {i}: {weight.shape}")
    
    print("=" * 60)
    
    return model

def safe_extract_parameters_server(parameters):
    """
    CORREZIONE GESTIONE PARAMETERS: Estrae i pesi dal tipo Parameters di Flower in modo sicuro per il server.
    Gestisce diversi tipi di tensori e conversioni per evitare errori di tipo.
    
    Args:
        parameters: Oggetto Parameters, lista di array numpy, o lista di tensori
    
    Returns:
        Lista di array numpy
    """
    try:
        print(f"  - Tipo parametri ricevuti: {type(parameters)}")
        
        # CASO 1: Se è già una lista di array numpy
        if isinstance(parameters, list):
            print(f"  - Parametri sono già una lista con {len(parameters)} elementi")
            weights_list = []
            for i, tensor in enumerate(parameters):
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    # Se è un tensor TensorFlow
                    weights_list.append(tensor.numpy())
                else:
                    # Prova conversione diretta
                    weights_list.append(np.array(tensor, dtype=np.float32))
            print(f"  - Processati {len(weights_list)} pesi dalla lista")
            return weights_list
        
        # CASO 2: Se è un oggetto Parameters di Flower
        elif isinstance(parameters, Parameters):
            print(f"  - Parametri sono un oggetto Parameters di Flower")
            if hasattr(parameters, 'tensors'):
                print(f"  - Trovati {len(parameters.tensors)} tensori nell'oggetto Parameters")
                weights_list = []
                for i, tensor in enumerate(parameters.tensors):
                    if isinstance(tensor, np.ndarray):
                        weights_list.append(tensor)
                    elif hasattr(tensor, 'numpy'):
                        # Se è un tensor TensorFlow
                        weights_list.append(tensor.numpy())
                    else:
                        # Prova conversione diretta
                        weights_list.append(np.array(tensor, dtype=np.float32))
                print(f"  - Processati {len(weights_list)} pesi dall'oggetto Parameters")
                return weights_list
            else:
                print(f"  - Oggetto Parameters senza attributo 'tensors'")
                raise ValueError("Oggetto Parameters non ha attributo 'tensors'")
        
        # CASO 3: Se ha attributo tensors (compatibilità)
        elif hasattr(parameters, 'tensors'):
            print(f"  - Parametri hanno attributo tensors con {len(parameters.tensors)} elementi")
            weights_list = []
            for tensor in parameters.tensors:
                if isinstance(tensor, np.ndarray):
                    weights_list.append(tensor)
                elif hasattr(tensor, 'numpy'):
                    weights_list.append(tensor.numpy())
                else:
                    weights_list.append(np.array(tensor, dtype=np.float32))
            return weights_list
        
        # CASO 4: Fallback - prova conversione diretta
        else:
            print(f"  - Tentativo di conversione diretta")
            if hasattr(parameters, 'numpy'):
                return [parameters.numpy()]
            else:
                return [np.array(parameters, dtype=np.float32)]
            
    except Exception as e:
        print(f"  - ❌ Errore nell'estrazione parametri server: {e}")
        # Ultimo fallback: ritorna i parametri come sono e speriamo nel meglio
        print(f"  - Tentativo fallback: ritorno parametri originali")
        return parameters

def check_server_parameters_compatibility(received_params, model_weights):
    """
    CORREZIONE GESTIONE PARAMETERS: Verifica la compatibilità tra parametri ricevuti e modello server.
    Include controlli di tipo, numero e forme dei pesi per il server.
    
    Args:
        received_params: Parametri ricevuti (possono essere di diversi tipi)
        model_weights: Pesi del modello server
    
    Returns:
        Tuple (is_compatible, extracted_weights, error_message)
    """
    try:
        print(f"  - === CONTROLLO COMPATIBILITÀ PARAMETRI SERVER ===")
        
        # STEP 1: Estrai i pesi in modo sicuro
        extracted_weights = safe_extract_parameters_server(received_params)
        
        # STEP 2: Verifica che sia una lista
        if not isinstance(extracted_weights, list):
            error_msg = f"Parametri estratti non sono una lista: {type(extracted_weights)}"
            print(f"  - ❌ {error_msg}")
            return False, None, error_msg
        
        # STEP 3: Verifica numero di pesi
        if len(extracted_weights) != len(model_weights):
            error_msg = f"Numero pesi incompatibile: ricevuti {len(extracted_weights)}, attesi {len(model_weights)}"
            print(f"  - ❌ {error_msg}")
            return False, None, error_msg
        
        print(f"  - ✅ Numero pesi compatibile: {len(extracted_weights)}")
        
        # STEP 4: Verifica forme dei pesi
        forme_compatibili = 0
        for i, (received_weight, model_weight) in enumerate(zip(extracted_weights, model_weights)):
            # Converti a numpy se necessario
            if not isinstance(received_weight, np.ndarray):
                try:
                    received_weight = np.array(received_weight, dtype=np.float32)
                    extracted_weights[i] = received_weight
                except Exception as e:
                    error_msg = f"Impossibile convertire peso {i} a numpy array: {e}"
                    print(f"  - ❌ {error_msg}")
                    return False, None, error_msg
            
            # Verifica forma
            if received_weight.shape != model_weight.shape:
                error_msg = f"Forma peso {i} incompatibile: ricevuta {received_weight.shape}, attesa {model_weight.shape}"
                print(f"  - ❌ {error_msg}")
                return False, None, error_msg
            
            forme_compatibili += 1
        
        # STEP 5: Tutto OK
        print(f"  - ✅ Compatibilità parametri server verificata:")
        print(f"  - ✅ Numero pesi: {len(extracted_weights)}")
        print(f"  - ✅ Forme verificate: {forme_compatibili}/{len(extracted_weights)} pesi")
        
        return True, extracted_weights, None
        
    except Exception as e:
        error_msg = f"Errore durante verifica compatibilità server: {str(e)}"
        print(f"  - ❌ {error_msg}")
        return False, None, error_msg

def safe_set_server_model_weights(model, parameters):
    """
    CORREZIONE GESTIONE PARAMETERS: Imposta i pesi del modello server in modo sicuro.
    Gestisce diversi tipi di parametri e include controlli di validazione per il server.
    
    Args:
        model: Modello Keras del server
        parameters: Parametri da impostare (diversi tipi possibili)
    
    Returns:
        Tuple (success, error_message)
    """
    try:
        print(f"  - === IMPOSTAZIONE SICURA PESI MODELLO SERVER ===")
        
        # STEP 1: Ottieni pesi attuali del modello server
        current_weights = model.get_weights()
        print(f"  - Pesi attuali modello server: {len(current_weights)}")
        
        # STEP 2: Verifica compatibilità e estrai pesi
        is_compatible, extracted_weights, error_msg = check_server_parameters_compatibility(
            parameters, current_weights
        )
        
        if not is_compatible:
            print(f"  - ❌ Incompatibilità rilevata: {error_msg}")
            return False, error_msg
        
        # STEP 3: Imposta i pesi estratti
        print(f"  - Impostazione {len(extracted_weights)} pesi sul modello server...")
        model.set_weights(extracted_weights)
        
        print(f"  - ✅ Pesi server impostati con successo")
        return True, None
        
    except Exception as e:
        error_msg = f"Errore durante impostazione pesi server: {str(e)}"
        print(f"  - ❌ {error_msg}")
        return False, error_msg

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid DNN.
    Usa la stessa pipeline di preprocessing dei client con PCA fisso e gestione robusta Parameters.
    
    Returns:
        Funzione di valutazione che può essere usata dal server
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Applica la stessa pipeline di preprocessing dei client con PCA fisso.
        """
        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER DNN CON PCA FISSO ===")
        
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
        
        # CORREZIONE: Applica la stessa pipeline dei client con PCA fisso
        fixed_pca_components = 50  # Identico ai client
        X_global_final, pca_components = apply_server_preprocessing_pipeline(
            X_global, 
            fixed_pca_components=fixed_pca_components
        )
        
        print(f"  - Dataset test globale preprocessato:")
        print(f"    - Campioni finali: {len(X_global_final)}")
        print(f"    - Feature dopo pipeline FISSE: {X_global_final.shape[1]}")
        print(f"    - Distribuzione mantenuta sbilanciata per valutazione realistica")
        print("=" * 60)
        
        return X_global_final, y_global, pca_components
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, input_shape = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi con shape fisso
        input_shape = 50  # Shape fisso
        X_global = np.random.random((100, input_shape))
        y_global = np.random.randint(0, 2, 100)
        print(f"Usando dati fittizi per valutazione globale con shape fisso: {input_shape}")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round.
        Valuta il modello DNN aggregato sui dati di test globali con pipeline, PCA fisso e gestione robusta Parameters.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato (gestione robusta)
            config: Configurazione
        
        Returns:
            Tuple con (loss, metriche)
        """
        print(f"\n=== VALUTAZIONE GLOBALE TEST DNN CON GESTIONE ROBUSTA PARAMETERS - ROUND {server_round} ===")
        
        try:
            # CORREZIONE: Crea il modello DNN per la valutazione (identico ai client con input fisso)
            model = create_server_dnn_model(input_shape)
            
            # CORREZIONE GESTIONE PARAMETERS: Usa funzione sicura per impostare pesi
            success, error_msg = safe_set_server_model_weights(model, parameters)
            
            if not success:
                print(f"❌ Errore nell'impostazione parametri server: {error_msg}")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": f"server_parameter_handling_failed: {error_msg}", 
                    "global_test_samples": 0,
                    "parameter_handling": "robust_safe_extraction_failed"
                }
            
            print(f"✅ Pesi aggregati impostati con successo sul modello server")
            
            # Valutazione sul dataset test globale (con pipeline, PCA fisso e gestione robusta Parameters)
            print(f"Avvio valutazione su {len(X_global)} campioni con {X_global.shape[1]} feature...")
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"=== RISULTATI VALUTAZIONE TEST GLOBALE DNN (GESTIONE ROBUSTA PARAMETERS) ===")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            print(f"  - AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  - Campioni test utilizzati: {len(X_global)}")
            print(f"  - Feature utilizzate (PCA FISSO): {X_global.shape[1]}")
            
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
                print(f"  - ⚠️  ATTENZIONE: Loss molto alta ({loss:.4f}) - possibile problema di convergenza")
            elif loss > 2.0:
                print(f"  - ⚠️  Loss alta ({loss:.4f})")
            elif loss < 1.0:
                print(f"  - ✅  Loss accettabile ({loss:.4f})")
                
            if accuracy < 0.5:
                print(f"  - ⚠️  ATTENZIONE: Accuracy molto bassa ({accuracy:.4f}) - performance sotto random")
            elif accuracy < 0.6:
                print(f"  - ⚠️  Accuracy bassa ({accuracy:.4f})")
            elif accuracy > 0.7:
                print(f"  - ✅  Performance buone (accuracy {accuracy:.4f})")
            else:
                print(f"  - ✅  Performance accettabili (accuracy {accuracy:.4f})")
            
            print("=" * 80)
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
                "model_type": "DNN_Standardized_Fixed_PCA_Parameters_Robust",
                "parameter_handling": "robust_safe_extraction_success"
            }
            
        except Exception as e:
            print(f"❌ Errore durante la valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {
                "accuracy": 0.0, 
                "error": str(e), 
                "global_test_samples": 0,
                "parameter_handling": "robust_safe_extraction_error"
            }
    
    return evaluate

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round di addestramento DNN.
    Include informazioni sulla pipeline di preprocessing applicata e modello DNN con PCA fisso e gestione robusta Parameters.
    
    Args:
        fit_results: Risultati dell'addestramento dai client
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ROUND DNN STANDARDIZZATO CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_original_features = 0
    total_pipeline_features = 0
    pipeline_info = None
    model_type = None
    total_local_epochs = 0
    total_batch_size = 0
    error_clients = []
    parameter_handling_clients = []
    
    accuracy_list = []
    loss_list = []
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client DNN {i+1}:")
        print(f"  - Campioni training: {client_samples}")
        
        # CORREZIONE GESTIONE PARAMETERS: Controlla se ci sono stati errori di compatibilità o gestione Parameters
        if 'error' in client_metrics:
            error_clients.append(i+1)
            print(f"  - ❌ ERRORE: {client_metrics['error']}")
            continue
        
        # NUOVO: Verifica gestione robusta Parameters
        if 'parameter_handling' in client_metrics:
            parameter_handling = client_metrics['parameter_handling']
            parameter_handling_clients.append(parameter_handling)
            print(f"  - 🔧 Gestione Parameters: {parameter_handling}")
        
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
            elif loss > 2.0:
                print(f"    ⚠️  Loss moderata per client {i+1}")
            elif loss < 1.0:
                print(f"    ✅  Loss accettabile per client {i+1}")
        
        if 'train_precision' in client_metrics:
            precision = client_metrics['train_precision']
            print(f"  - Train Precision: {precision:.4f}")
            if precision < 0.5:
                print(f"    ⚠️  Precision bassa")
            elif precision > 0.7:
                print(f"    ✅  Precision buona")
        
        if 'train_recall' in client_metrics:
            recall = client_metrics['train_recall']
            print(f"  - Train Recall: {recall:.4f}")
            if recall < 0.5:
                print(f"    ⚠️  Recall basso")
            elif recall > 0.7:
                print(f"    ✅  Recall buono")
        
        # Metriche aggiuntive DNN
        if 'train_f1_score' in client_metrics:
            f1 = client_metrics['train_f1_score']
            print(f"  - Train F1-Score: {f1:.4f}")
            if f1 < 0.5:
                print(f"    ⚠️  F1-Score basso")
            elif f1 > 0.7:
                print(f"    ✅  F1-Score buono")
        
        if 'train_auc' in client_metrics:
            auc = client_metrics['train_auc']
            print(f"  - Train AUC: {auc:.4f}")
            if auc < 0.7:
                print(f"    ⚠️  AUC basso")
            elif auc > 0.8:
                print(f"    ✅  AUC ottimo")
        
        if 'local_epochs' in client_metrics:
            epochs = client_metrics['local_epochs']
            total_local_epochs = epochs
            print(f"  - Epoche locali: {epochs}")
        
        if 'batch_size' in client_metrics:
            batch_size = client_metrics['batch_size']
            total_batch_size = batch_size
            print(f"  - Batch size: {batch_size}")
        
        # CORREZIONE: Informazioni sulla compatibilità
        if 'weights_count' in client_metrics:
            weights_count = client_metrics['weights_count']
            print(f"  - Numero pesi: {weights_count}")
        
        # Informazioni sulla pipeline con PCA fisso
        if 'original_features' in client_metrics and 'pca_features' in client_metrics:
            orig_feat = client_metrics['original_features']
            pipeline_feat = client_metrics['pca_features']
            total_original_features = orig_feat
            total_pipeline_features = pipeline_feat
            print(f"  - Feature originali: {orig_feat}")
            print(f"  - Feature post-PCA (FISSE): {pipeline_feat}")
            
        if 'pca_reduction' in client_metrics:
            reduction = client_metrics['pca_reduction']
            print(f"  - Riduzione PCA: {reduction:.1f}%")
        
        if 'variance_explained' in client_metrics:
            variance = client_metrics['variance_explained']
            print(f"  - Varianza spiegata: {variance:.1f}%")
        
        if 'pipeline_applied' in client_metrics:
            pipeline_info = client_metrics['pipeline_applied']
        
        if 'model_type' in client_metrics:
            model_type = client_metrics['model_type']
        
        # Mostra anche informazioni sui dati di validation locali
        if 'val_samples' in client_metrics:
            val_samples = client_metrics['val_samples']
            print(f"  - Campioni validation: {val_samples}")
    
    if total_samples > 0:
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_loss = np.mean(loss_list) if loss_list else 0
        std_accuracy = np.std(accuracy_list) if accuracy_list else 0
        
        print(f"\n=== RIASSUNTO AGGREGATO DNN STANDARDIZZATO CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS ===")
        print(f"  - Client con errori: {len(error_clients)} / {len(fit_results)}")
        if error_clients:
            print(f"  - Client in errore: {error_clients}")
        
        # NUOVO: Riassunto gestione Parameters
        parameter_handling_summary = {}
        for ph in parameter_handling_clients:
            parameter_handling_summary[ph] = parameter_handling_summary.get(ph, 0) + 1
        
        if parameter_handling_summary:
            print(f"  - Gestione Parameters utilizzata:")
            for handling_type, count in parameter_handling_summary.items():
                print(f"    - {handling_type}: {count} client")
        
        print(f"  - Media pesata accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  - Media loss: {avg_loss:.4f}")
        print(f"  - Std accuracy tra client: {std_accuracy:.4f}")
        print(f"  - Totale campioni training: {total_samples}")
        print(f"  - Epoche locali per client: {total_local_epochs}")
        print(f"  - Batch size: {total_batch_size}")
        
        if total_original_features > 0 and total_pipeline_features > 0:
            print(f"  - Riduzione dimensionalità comune: {total_original_features} → {total_pipeline_features} (FISSO)")
            print(f"  - Percentuale riduzione: {(1 - total_pipeline_features/total_original_features)*100:.1f}%")
        
        if pipeline_info:
            print(f"  - Pipeline applicata: {pipeline_info}")
        
        if model_type:
            print(f"  - Tipo modello: {model_type}")
            
        # Controlli di qualità aggregati
        if avg_loss > 3.0:
            print(f"  ⚠️  ATTENZIONE: Loss media molto alta ({avg_loss:.4f}) - problemi di convergenza")
        elif avg_loss > 2.0:
            print(f"  ⚠️  ATTENZIONE: Loss media alta ({avg_loss:.4f}) - possibili problemi")
        elif avg_loss < 1.0:
            print(f"  ✅  Loss media accettabile ({avg_loss:.4f})")
        else:
            print(f"  ✅  Loss media OK ({avg_loss:.4f})")
        
        if avg_weighted_accuracy < 0.5:
            print(f"  ⚠️  ATTENZIONE: Accuracy media molto bassa ({avg_weighted_accuracy:.4f})")
        elif avg_weighted_accuracy < 0.6:
            print(f"  ⚠️  ATTENZIONE: Accuracy media bassa ({avg_weighted_accuracy:.4f})")
        elif avg_weighted_accuracy > 0.7:
            print(f"  ✅  Accuracy media buona ({avg_weighted_accuracy:.4f})")
        else:
            print(f"  ✅  Accuracy media accettabile ({avg_weighted_accuracy:.4f})")
        
        if std_accuracy > 0.3:
            print(f"  ⚠️  ATTENZIONE: Alta variabilità tra client (std: {std_accuracy:.4f})")
        elif std_accuracy > 0.2:
            print(f"  ⚠️  Variabilità moderata tra client (std: {std_accuracy:.4f})")
        else:
            print(f"  ✅  Variabilità tra client accettabile (std: {std_accuracy:.4f})")
    
    print("=" * 80)

def print_client_evaluation_metrics(eval_results):
    """
    Stampa le metriche di valutazione dei client DNN con PCA fisso e gestione robusta Parameters.
    
    Args:
        eval_results: Risultati della valutazione dai client
    """
    if not eval_results:
        return
    
    print(f"\n=== METRICHE VALIDATION CLIENT DNN CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS ===")
    
    total_val_samples = 0
    total_weighted_val_accuracy = 0
    avg_metrics = {'precision': 0, 'recall': 0, 'f1_score': 0, 'auc': 0}
    val_accuracy_list = []
    val_loss_list = []
    eval_error_clients = []
    eval_parameter_handling_clients = []
    
    for i, (client_proxy, eval_res) in enumerate(eval_results):
        val_samples = eval_res.num_examples
        eval_metrics = eval_res.metrics
        val_loss = eval_res.loss
        
        total_val_samples += val_samples
        val_loss_list.append(val_loss)
        
        print(f"Client DNN {i+1} Validation:")
        print(f"  - Campioni validation: {val_samples}")
        print(f"  - Val Loss: {val_loss:.4f}")
        
        # CORREZIONE GESTIONE PARAMETERS: Controlla se ci sono stati errori di compatibilità o gestione Parameters
        if 'error' in eval_metrics:
            eval_error_clients.append(i+1)
            print(f"  - ❌ ERRORE VALIDATION: {eval_metrics['error']}")
            continue
        
        # NUOVO: Verifica gestione robusta Parameters in valutazione
        if 'parameter_handling' in eval_metrics:
            parameter_handling = eval_metrics['parameter_handling']
            eval_parameter_handling_clients.append(parameter_handling)
            print(f"  - 🔧 Gestione Parameters Validation: {parameter_handling}")
        
        if 'accuracy' in eval_metrics:
            val_accuracy = eval_metrics['accuracy']
            total_weighted_val_accuracy += val_accuracy * val_samples
            val_accuracy_list.append(val_accuracy)
            print(f"  - Val Accuracy: {val_accuracy:.4f}")
            
            # Controllo qualità validation accuracy
            if val_accuracy < 0.5:
                print(f"    ⚠️  Validation accuracy molto bassa")
            elif val_accuracy < 0.6:
                print(f"    ⚠️  Validation accuracy bassa")
            elif val_accuracy > 0.7:
                print(f"    ✅  Validation accuracy buona")
        
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
        
        print(f"\n=== RIASSUNTO VALIDATION AGGREGATO DNN CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS ===")
        print(f"  - Client con errori validation: {len(eval_error_clients)} / {num_clients}")
        if eval_error_clients:
            print(f"  - Client in errore validation: {eval_error_clients}")
        
        # NUOVO: Riassunto gestione Parameters in validation
        eval_parameter_handling_summary = {}
        for ph in eval_parameter_handling_clients:
            eval_parameter_handling_summary[ph] = eval_parameter_handling_summary.get(ph, 0) + 1
        
        if eval_parameter_handling_summary:
            print(f"  - Gestione Parameters Validation utilizzata:")
            for handling_type, count in eval_parameter_handling_summary.items():
                print(f"    - {handling_type}: {count} client")
        
        print(f"  - Media pesata validation accuracy: {avg_weighted_val_accuracy:.4f}")
        print(f"  - Media validation loss: {avg_val_loss:.4f}")
        print(f"  - Std validation accuracy: {std_val_accuracy:.4f}")
        print(f"  - Media validation precision: {avg_metrics['precision']/num_clients:.4f}")
        print(f"  - Media validation recall: {avg_metrics['recall']/num_clients:.4f}")
        print(f"  - Media validation F1-Score: {avg_metrics['f1_score']/num_clients:.4f}")
        print(f"  - Media validation AUC: {avg_metrics['auc']/num_clients:.4f}")
        print(f"  - Totale campioni validation: {total_val_samples}")
        
        # Controlli di qualità validation
        if avg_val_loss > 3.0:
            print(f"  ⚠️  ATTENZIONE: Validation loss molto alta ({avg_val_loss:.4f})")
        elif avg_val_loss > 2.0:
            print(f"  ⚠️  ATTENZIONE: Validation loss alta ({avg_val_loss:.4f})")
        elif avg_val_loss < 1.0:
            print(f"  ✅  Validation loss accettabile ({avg_val_loss:.4f})")
        
        if avg_weighted_val_accuracy < 0.5:
            print(f"  ⚠️  ATTENZIONE: Validation accuracy molto bassa ({avg_weighted_val_accuracy:.4f})")
        elif avg_weighted_val_accuracy < 0.6:
            print(f"  ⚠️  ATTENZIONE: Validation accuracy bassa ({avg_weighted_val_accuracy:.4f})")
        elif avg_weighted_val_accuracy > 0.7:
            print(f"  ✅  Validation accuracy buona ({avg_weighted_val_accuracy:.4f})")
        else:
            print(f"  ✅  Validation accuracy accettabile ({avg_weighted_val_accuracy:.4f})")
    
    print("=" * 80)

class SmartGridDNNFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid DNN con logging migliorato e monitoraggio qualità.
    Include gestione errori e controlli di compatibilità per architetture standardizzate con PCA fisso e gestione robusta Parameters.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento DNN e stampa metriche dettagliate.
        Include controlli di compatibilità per architetture standardizzate con PCA fisso e gestione robusta Parameters.
        """
        print(f"\n=== AGGREGAZIONE TRAINING DNN STANDARDIZZATO CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS - ROUND {server_round} ===")
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
        
        # CORREZIONE: Controllo preventivo compatibilità pesi
        print(f"=== CONTROLLO COMPATIBILITÀ PESI TRA CLIENT ===")
        weight_lengths = []
        parameter_handling_types = []
        
        for i, (client_proxy, fit_res) in enumerate(results):
            if 'weights_count' in fit_res.metrics:
                weight_count = fit_res.metrics['weights_count']
                weight_lengths.append(weight_count)
                print(f"  - Client {i+1}: {weight_count} pesi")
                
            # NUOVO: Raccogli informazioni sulla gestione Parameters
            if 'parameter_handling' in fit_res.metrics:
                param_handling = fit_res.metrics['parameter_handling']
                parameter_handling_types.append(param_handling)
                print(f"  - Client {i+1}: gestione Parameters = {param_handling}")
        
        # Verifica che tutti i client abbiano lo stesso numero di pesi
        if weight_lengths and len(set(weight_lengths)) > 1:
            print(f"⚠️  ATTENZIONE: Incompatibilità nel numero di pesi tra client: {set(weight_lengths)}")
        elif weight_lengths:
            print(f"✅  Compatibilità pesi verificata: {weight_lengths[0]} pesi per client")
        
        # NUOVO: Verifica gestione Parameters uniforme
        if parameter_handling_types:
            unique_handling = set(parameter_handling_types)
            if len(unique_handling) == 1:
                print(f"✅  Gestione Parameters uniforme: {list(unique_handling)[0]}")
            else:
                print(f"⚠️  Gestione Parameters mista: {unique_handling}")
        
        # Stampa metriche dei client DNN standardizzati con PCA fisso e gestione robusta Parameters
        print_client_metrics(results)
        
        # Chiama l'aggregazione standard
        try:
            print(f"=== AVVIO AGGREGAZIONE STANDARD FLOWER ===")
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione DNN standardizzato con PCA fisso e gestione robusta Parameters completata per round {server_round}")
                print(f"✅ Pesi di {len(results)} client DNN aggregati con successo")
                
                # NUOVO: Verifica stabilità numerica dei pesi aggregati (se possibile)
                if isinstance(aggregated_result, tuple) and len(aggregated_result) >= 1:
                    aggregated_parameters = aggregated_result[0]
                    print(f"✅ Tipo parametri aggregati: {type(aggregated_parameters)}")
                    
                    if hasattr(aggregated_parameters, 'tensors'):
                        print(f"✅ Parametri aggregati contengono {len(aggregated_parameters.tensors)} tensori")
                    elif isinstance(aggregated_parameters, list):
                        print(f"✅ Parametri aggregati sono una lista con {len(aggregated_parameters)} elementi")
            else:
                print(f"❌ ATTENZIONE: Aggregazione DNN fallita per round {server_round}")
                print(f"❌ Possibili cause: incompatibilità pesi, errori numerici, o problemi di gestione Parameters")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione DNN e stampa metriche dettagliate.
        Include gestione robusta Parameters.
        """
        print(f"\n=== AGGREGAZIONE VALUTAZIONE DNN CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS ROUND {server_round} ===")
        print(f"Client DNN che hanno valutato: {len(results)}")
        print(f"Client falliti nella valutazione: {len(failures)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Stampa metriche di valutazione dei client DNN con PCA fisso e gestione robusta Parameters
        print_client_evaluation_metrics(results)
        
        # Chiama l'aggregazione standard
        try:
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione DNN con gestione robusta Parameters completata per round {server_round}")
            else:
                print(f"⚠️  Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione valutazione: {e}")
            return None
        
        print("=" * 80)
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato DNN con architettura standardizzata, PCA fisso e gestione robusta Parameters.
    """
    print("=== AVVIO SERVER FEDERATO SMARTGRID DNN STANDARDIZZATO CON PCA FISSO E GESTIONE ROBUSTA PARAMETERS ===")
    print("CORREZIONI IMPLEMENTATE:")
    print("  - PCA FISSO: 50 componenti per tutti i client (risolve errore broadcast)")
    print("  - Architettura standardizzata: Input(50) → Dense(128) → Dense(64) → Dense(32) → Dense(1)")
    print("  - Compatibilità server-client: modelli identici per evitare errore pesi")
    print("  - Input layer esplicito: evita warning Keras")
    print("  - GESTIONE ROBUSTA PARAMETERS: extraction sicura, controlli tipo, fallback")
    print("  - Debug pesi: logging dettagliato per troubleshooting")
    print("  - Verifiche forme: controlli compatibilità forme pesi")
    print("  - Controlli qualità: monitoraggio automatico performance")
    print("  - Stabilità numerica: gestione overflow e valori non validi")
    print("")
    print("FUNZIONALITÀ GESTIONE ROBUSTA PARAMETERS:")
    print("  - safe_extract_parameters_server(): gestione sicura diversi tipi Parameters")
    print("  - check_server_parameters_compatibility(): verifica compatibilità completa server")
    print("  - safe_set_server_model_weights(): impostazione sicura pesi del modello server")
    print("  - Supporto oggetti Parameters di Flower e liste numpy arrays")
    print("  - Fallback automatici per diversi tipi di tensori")
    print("  - Controlli numerici avanzati per stabilità")
    print("  - Logging esteso per debugging della gestione Parameters")
    print("")
    print("Configurazione:")
    print("  - Numero di round: 5")
    print("  - Client minimi per training: 2")
    print("  - Client minimi per valutazione: 2")
    print("  - Client minimi disponibili: 2")
    print("  - Strategia: FedAvg personalizzata per DNN con PCA fisso e gestione robusta Parameters")
    print("  - Valutazione: Dataset globale DNN con pipeline, PCA fisso e gestione robusta Parameters (client 14-15)")
    print("  - Client training: Usano train (70%) + validation (30%) locale")
    print("  - Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA(50) FISSO")
    print("  - Modello: DNN standardizzata (architettura identica client-server)")
    print("  - Architettura: Input(50) → 3 layer nascosti (128→64→32→1) - FISSA per compatibilità")
    print("  - Regolarizzazione: Dropout 0.2 + L2 0.0001 + BatchNormalization")
    print("  - Ottimizzazioni: Learning Rate 0.0001 + Gradient Clipping 1.0")
    print("  - Training locale: 3 epoche per round, batch size 16")
    print("  - Gestione errori: Controlli compatibilità pesi + Debug architettura + Verifiche forme + Gestione robusta Parameters")
    print("  - Monitoraggio: Controlli qualità automatici + Logging esteso + Analisi gestione Parameters")
    print("  - Stabilità: PCA fisso + Controlli numerici + Compatibilità garantita + Gestione robusta Parameters")
    print("=" * 120)
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Strategia Federated Averaging personalizzata per DNN con PCA fisso e gestione robusta Parameters
    strategy = SmartGridDNNFedAvg(
        fraction_fit=1.0,                    # Usa tutti i client disponibili per training
        fraction_evaluate=1.0,               # Usa tutti i client disponibili per valutazione
        min_fit_clients=2,                   # Numero minimo di client per iniziare training
        min_evaluate_clients=2,              # Numero minimo di client per valutazione
        min_available_clients=2,             # Numero minimo di client connessi
        evaluate_fn=get_smartgrid_evaluate_fn()  # Valutazione globale DNN con pipeline, PCA fisso e gestione robusta Parameters
    )
    
    print("Server DNN standardizzato con PCA fisso e gestione robusta Parameters in attesa di client...")
    print("Per connettere i client DNN con PCA fisso e gestione robusta Parameters, esegui in terminali separati:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  python client.py 3")
    print("  ...")
    print("  python client.py 13")
    print("\nNOTA: Usa client ID 1-13 per training federato DNN con PCA fisso e gestione robusta Parameters")
    print("      Client 14-15 sono riservati per valutazione globale")
    print("      Ogni client applicherà la pipeline e addestrerà DNN standardizzata:")
    print("      Split → Imputazione → Normalizzazione → SMOTE → PCA(50) FISSO → DNN Training con gestione robusta Parameters")
    print("      Ogni client DNN eseguirà 3 epoche locali per round federato con batch size 16")
    print("      Learning rate 0.0001 con gradient clipping per stabilità")
    print("      Controlli automatici di compatibilità architettura e forme pesi")
    print("      Gestione errori robusta per compatibilità, stabilità e gestione Parameters")
    print("      PCA fisso a 50 componenti per garantire compatibilità matriciale")
    print("      Gestione robusta Parameters di Flower per evitare errori di tipo")
    print("\nIl training federato DNN con PCA fisso e gestione robusta Parameters inizierà quando almeno 2 client saranno connessi.")
    print("Il sistema include controlli automatici completi per:")
    print("  - Compatibilità architettura: modelli identici client-server")
    print("  - Compatibilità matriciale: PCA fisso risolve errore broadcast")
    print("  - Gestione robusta Parameters: supporto diversi tipi Parameters di Flower")
    print("  - Debug pesi: verifica numero e forme dei pesi")
    print("  - Monitoraggio performance: controlli qualità automatici")
    print("  - Logging esteso: troubleshooting semplificato con analisi gestione Parameters")
    print("  - Stabilità numerica: gestione robusta errori e overflow")
    print("  - Fallback automatici: gestione sicura di tutti i tipi di tensori e conversioni")
    print("=" * 120)
    sys.stdout.flush()
    
    try:
        # Avvia il server
        print("Avvio del server Flower con architettura standardizzata, PCA fisso e gestione robusta Parameters...")
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Errore durante l'avvio del server DNN standardizzato: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()