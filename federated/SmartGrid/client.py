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

def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico.
    Ogni client ha accesso solo al proprio file CSV.
    
    Args:
        client_id: ID del client (1-15)
    
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
    
    # STEP 5: PCA (fit sui dati training bilanciati)
    print(f"\n=== STEP 5: RIDUZIONE DIMENSIONALE CON PCA ===")
    
    original_features = X_train_balanced.shape[1]
    
    # Analisi PCA per selezione automatica componenti
    pca_full = PCA()
    pca_full.fit(X_train_balanced)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    n_components = min(n_components, original_features, len(X_train_balanced))
    
    print(f"  - Feature originali: {original_features}")
    print(f"  - Componenti selezionate: {n_components}")
    print(f"  - Varianza spiegata: {cumulative_variance[n_components-1]*100:.2f}%")
    print(f"  - Riduzione: {(1-n_components/original_features)*100:.1f}%")
    
    # Applica PCA
    pca_optimal = PCA(n_components=n_components)
    X_train_final = pca_optimal.fit_transform(X_train_balanced)
    X_val_final = pca_optimal.transform(X_val_preprocessed)
    
    print(f"  - Training final shape: {X_train_final.shape}")
    print(f"  - Validation final shape: {X_val_final.shape}")
    
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
        'pca_reduction': (1 - n_components / original_features) * 100
    }
    
    print("=" * 60)
    
    return X_train_final, y_train_balanced, X_val_final, y_val, dataset_info

def create_smartgrid_client_dnn_model(input_shape):
    """
    Crea il modello DNN SmartGrid per il client.
    Identico al modello centralizzato per garantire compatibilità federata.
    
    Args:
        input_shape: Numero di feature in input (dopo PCA)
    
    Returns:
        Modello Keras compilato
    """
    print(f"=== CREAZIONE MODELLO DNN CLIENT ===")
    
    # Configurazione identica al modello centralizzato
    dropout_rate = 0.3
    l2_reg = 0.001
    
    # Architettura DNN identica per compatibilità federata
    model = keras.Sequential([
        # Layer di input
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        # Primo blocco: Estrazione feature di alto livello
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Secondo blocco: Raffinamento pattern
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Terzo blocco: Specializzazione per sicurezza
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(dropout_rate, name='dropout_3'),
        
        # Quarto blocco: Consolidamento pattern
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='dense_4'),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(dropout_rate / 2, name='dropout_4'),
        
        # Layer finale: Classificazione binaria
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # Ottimizzatore identico al centralizzato
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
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
    
    print(f"  - Modello DNN client creato")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Architettura: 4 layer nascosti (256→128→64→32)")
    print(f"  - Parametri totali: {model.count_params():,}")
    print("=" * 60)
    
    return model

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
    per la rilevazione di intrusioni in smart grid con DNN e pipeline corretta.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello DNN locale.
        """
        return model.get_weights()

    def fit(self, parameters, config):
        """
        Addestra il modello DNN sui dati locali del client (con pipeline completa).
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO DNN ===")
        print(f"[Client {client_id}] Ricevuti parametri dal server, avvio addestramento locale...")
        sys.stdout.flush()
        
        # Imposta i pesi ricevuti dal server
        model.set_weights(parameters)
        
        # Verifica che ci siano dati di training
        if len(X_train) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Addestra il modello DNN localmente per più epoche (federato necessita più training locale)
        local_epochs = 5  # Più epoche per DNN in contesto federato
        
        print(f"[Client {client_id}] Addestramento DNN su {len(X_train)} campioni per {local_epochs} epoche...")
        print(f"[Client {client_id}] Feature utilizzate (post-pipeline): {X_train.shape[1]}")
        
        history = model.fit(
            X_train, y_train,
            epochs=local_epochs,
            batch_size=32,      # Batch size più piccolo per client
            verbose=0,
            shuffle=True
        )
        
        # Estrai le metriche dall'addestramento (ultima epoca)
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        train_precision = history.history.get('precision', [0])[-1]
        train_recall = history.history.get('recall', [0])[-1]
        train_auc = history.history.get('auc', [0])[-1]
        
        # Calcola F1-score
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        
        print(f"[Client {client_id}] Addestramento DNN completato:")
        print(f"[Client {client_id}]   - Loss: {train_loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {train_accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {train_precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {train_recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {train_f1:.4f}")
        print(f"[Client {client_id}]   - AUC: {train_auc:.4f}")
        
        # Metriche da inviare al server
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            'train_auc': float(train_auc),
            'local_epochs': int(local_epochs),
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
            'pipeline_applied': 'split_impute_scale_smote_pca',
            'model_type': 'DNN'
        }
        
        print(f"[Client {client_id}] Invio pesi DNN aggiornati al server...")
        sys.stdout.flush()
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello DNN sui dati di validation locali del client (post-pipeline).
        """
        global model, X_val, y_val
        
        print(f"[Client {client_id}] === VALUTAZIONE LOCALE DNN ===")
        
        # Imposta i pesi da valutare
        model.set_weights(parameters)
        
        # Verifica che ci siano dati di validation
        if len(X_val) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di validation disponibile!")
            return 0.0, 0, {"accuracy": 0.0}
        
        # Valuta sui dati di validation locali
        results = model.evaluate(X_val, y_val, verbose=0)
        loss, accuracy, precision, recall, auc = results
        
        # Calcola F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"[Client {client_id}] Valutazione DNN locale completata:")
        print(f"[Client {client_id}]   - Loss: {loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {f1_score:.4f}")
        print(f"[Client {client_id}]   - AUC: {auc:.4f}")
        print(f"[Client {client_id}]   - Campioni validation: {len(X_val)}")
        print(f"[Client {client_id}]   - Feature utilizzate (post-pipeline): {X_val.shape[1]}")
        
        # Metriche da restituire
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc": auc,
            "val_samples": len(X_val)
        }
        
        return loss, len(X_val), metrics

def main():
    """
    Funzione principale per avviare il client SmartGrid DNN con pipeline corretta.
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
    
    print(f"AVVIO CLIENT SMARTGRID DNN {client_id} CON PIPELINE CORRETTA")
    print("Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA")
    print("Modello: Deep Neural Network (4 layer nascosti) con regolarizzazione")
    print("Configurazione: train (70%) + validation locale (30%)")
    print("=" * 70)
    
    try:
        # 1. Carica i dati locali del client con pipeline completa
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
        
        # 2. Crea il modello DNN locale (con input shape delle feature ridotte)
        print(f"[Client {client_id}] Creazione modello DNN locale...")
        model = create_smartgrid_client_dnn_model(X_train.shape[1])
        print(f"[Client {client_id}] Modello DNN creato con {X_train.shape[1]} feature di input (post-pipeline)")
        
        # 3. Stampa riassunto del client
        print(f"[Client {client_id}] === RIASSUNTO CLIENT DNN ===")
        print(f"[Client {client_id}] Dataset info:")
        for key, value in dataset_info.items():
            if key == 'pca_reduction':
                print(f"[Client {client_id}]   - {key}: {value:.1f}%")
            else:
                print(f"[Client {client_id}]   - {key}: {value}")
        
        print(f"[Client {client_id}] Modello: DNN con 4 layer nascosti (256→128→64→32)")
        print(f"[Client {client_id}] Parametri totali: {model.count_params():,}")
        print(f"[Client {client_id}] Pipeline applicata: Split → Imputazione → Normalizzazione → SMOTE → PCA")
        print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
        sys.stdout.flush()
        
        # 4. Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridDNNClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()