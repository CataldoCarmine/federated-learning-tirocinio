import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico.
    Ogni client ha accesso solo al proprio file CSV.
    
    Args:
        client_id: ID del client (1-15)
    
    Returns:
        Tuple con (X_train, y_train, X_test, y_test, scaler, dataset_info)
    """
    print(f"=== CARICAMENTO DATI CLIENT {client_id} SMARTGRID ===")
    
    # Directory contenente questo script (ad es. federated/SmartGrid/client.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path assoluto al file CSV per questo client
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    # Carica il dataset del client
    df = pd.read_csv(file_path)
    
    print(f"Dataset del client {client_id}:")
    print(f"  - Totale campioni: {len(df)}")
    print(f"  - Feature: {df.shape[1] - 1}")  # -1 per escludere la colonna marker
    
    # Separa feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale
    
    # Statistiche del dataset locale
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    print(f"  - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"  - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    
    # === PREPROCESSING STEP 1: GESTIONE VALORI MANCANTI ===
    print(f"Pulizia dei dati:")
    initial_samples = len(X)

    # Gestisci valori infiniti e NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = X.isnull().sum().sum()
    print(f"  - Valori NaN trovati: {nan_count}")

    # MODIFICA: Imputazione dei NaN con la mediana (invece di rimozione)
    if nan_count > 0:
        print(f"  - Applicazione imputazione con mediana...")
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        print(f"  - Imputazione completata")

    # y resta allineato (nessuna riga viene rimossa)
    final_samples = len(X)
    removed_samples = initial_samples - final_samples  # Sarà 0 ora
    print(f"  - Campioni rimossi: {removed_samples}")
    print(f"  - Campioni finali: {final_samples}")

    if final_samples == 0:
        raise ValueError(f"Nessun campione valido rimasto per il client {client_id}")
    
    # === PREPROCESSING STEP 2: NORMALIZZAZIONE ===
    print(f"Normalizzazione feature locali...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Suddivisione in train/test locale PRIMA del bilanciamento
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None  # Stratify solo se ci sono entrambe le classi
    )
    
    print(f"Suddivisione locale train/test:")
    print(f"  - Training set: {len(X_train)} campioni")
    print(f"  - Test set: {len(X_test)} campioni")
    
    # Verifica distribuzione delle classi prima del bilanciamento
    if len(X_train) > 0:
        train_attack_ratio = y_train.mean() if len(y_train) > 0 else 0
        print(f"  - Proporzione attacchi training (originale): {train_attack_ratio*100:.2f}%")
    
    if len(X_test) > 0:
        test_attack_ratio = y_test.mean() if len(y_test) > 0 else 0
        print(f"  - Proporzione attacchi test: {test_attack_ratio*100:.2f}%")
    
    # === PREPROCESSING STEP 3: GESTIONE SQUILIBRIO CLASSI (SOLO TRAINING) ===
    print(f"\n=== BILANCIAMENTO CLASSI SUL TRAINING SET ===")
    
    # Calcola il rapporto di squilibrio nel training set
    if len(y_train) > 0:
        train_attack_ratio = y_train.mean()
        minority_class_ratio = min(train_attack_ratio, 1 - train_attack_ratio)
        unique_classes = len(np.unique(y_train))
        
        print(f"  - Distribuzione training PRIMA del bilanciamento:")
        print(f"    - Classe 0 (Natural): {(y_train == 0).sum()} campioni ({(1-train_attack_ratio)*100:.2f}%)")
        print(f"    - Classe 1 (Attack): {(y_train == 1).sum()} campioni ({train_attack_ratio*100:.2f}%)")
        print(f"    - Rapporto classe minoritaria: {minority_class_ratio*100:.2f}%")
        
        # Applica SMOTE solo se lo squilibrio è significativo e ci sono entrambe le classi
        if minority_class_ratio < 0.4 and unique_classes > 1:
            print(f"  - Squilibrio significativo rilevato, applicazione SMOTE al training set...")
            
            try:
                # Configura SMOTE con k_neighbors adattivo
                min_samples_per_class = min((y_train == 0).sum(), (y_train == 1).sum())
                k_neighbors = min(5, min_samples_per_class - 1) if min_samples_per_class > 1 else 1
                
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=42,
                    k_neighbors=k_neighbors
                )
                
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                # Statistiche dopo il bilanciamento
                print(f"  - SMOTE applicato con successo al training set")
                print(f"  - Distribuzione training DOPO il bilanciamento:")
                print(f"    - Classe 0 (Natural): {(y_train_balanced == 0).sum()} campioni ({(y_train_balanced == 0).mean()*100:.2f}%)")
                print(f"    - Classe 1 (Attack): {(y_train_balanced == 1).sum()} campioni ({(y_train_balanced == 1).mean()*100:.2f}%)")
                print(f"    - Campioni sintetici generati: {len(X_train_balanced) - len(X_train)}")
                
                # Usa i dati di training bilanciati
                X_train = X_train_balanced
                y_train = y_train_balanced
                
                print(f"  - Test set rimane sbilanciato per valutazione realistica")
                
            except Exception as e:
                print(f"  - Errore durante l'applicazione di SMOTE: {e}")
                print(f"  - Continuazione con dati di training originali")
        else:
            if unique_classes == 1:
                print(f"  - ATTENZIONE: Presente solo una classe nei dati, SMOTE non applicabile")
            else:
                print(f"  - Squilibrio accettabile, SMOTE non necessario")
    
    # Informazioni del dataset per reporting
    dataset_info = {
        'client_id': client_id,
        'total_samples': final_samples,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio,
        'features': X.shape[1]
    }
    
    print("=" * 60)
    
    return X_train, y_train, X_test, y_test, scaler, dataset_info

def create_smartgrid_client_model(input_shape):
    """
    Crea il modello SmartGrid per il client.
    Identico al modello centralizzato per garantire compatibilità.
    
    Args:
        input_shape: Numero di feature in input
    
    Returns:
        Modello Keras compilato
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", "precision", "recall"]
    )
    
    return model

# Variabili globali per il client
client_id = None
model = None
X_train = None
y_train = None
X_test = None
y_test = None
dataset_info = None

class SmartGridClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid che implementa l'addestramento federato
    per la rilevazione di intrusioni in smart grid.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello locale.
        
        Args:
            config: Configurazione dal server
        
        Returns:
            Lista dei pesi del modello
        """
        return model.get_weights()

    def fit(self, parameters, config):
        """
        Addestra il modello sui dati locali del client.
        
        Args:
            parameters: Pesi del modello globale ricevuti dal server
            config: Configurazione dell'addestramento
        
        Returns:
            Tuple con (pesi_aggiornati, numero_campioni, metriche)
        """
        global model, X_train, y_train, dataset_info
        
        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO ===")
        print(f"[Client {client_id}] Ricevuti parametri dal server, avvio addestramento locale...")
        sys.stdout.flush()
        
        # Imposta i pesi ricevuti dal server
        model.set_weights(parameters)
        
        # Verifica che ci siano dati di training
        if len(X_train) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Addestra il modello localmente per 1 epoca
        print(f"[Client {client_id}] Addestramento su {len(X_train)} campioni per 1 epoca...")
        
        history = model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        # Estrai le metriche dall'addestramento
        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        train_precision = history.history.get('precision', [0])[0]
        train_recall = history.history.get('recall', [0])[0]
        
        print(f"[Client {client_id}] Addestramento completato:")
        print(f"[Client {client_id}]   - Loss: {train_loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {train_accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {train_precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {train_recall:.4f}")
        
        # Metriche da inviare al server (solo tipi Python nativi, niente dizionari nested)
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'client_id': int(dataset_info['client_id']),
            'total_samples': int(dataset_info['total_samples']),
            'train_samples': int(dataset_info['train_samples']),
            'test_samples': int(dataset_info['test_samples']),
            'attack_samples': int(dataset_info['attack_samples']),
            'natural_samples': int(dataset_info['natural_samples']),
            'attack_ratio': float(dataset_info['attack_ratio']),
            'features': int(dataset_info['features']),
        }
        
        print(f"[Client {client_id}] Invio pesi aggiornati al server...")
        sys.stdout.flush()
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello sui dati di test locali del client.
        
        Args:
            parameters: Pesi del modello da valutare
            config: Configurazione della valutazione
        
        Returns:
            Tuple con (loss, numero_campioni, metriche)
        """
        global model, X_test, y_test
        
        print(f"[Client {client_id}] === VALUTAZIONE LOCALE ===")
        
        # Imposta i pesi da valutare
        model.set_weights(parameters)
        
        # Verifica che ci siano dati di test
        if len(X_test) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di test disponibile!")
            return 0.0, 0, {"accuracy": 0.0}
        
        # Valuta sui dati di test locali
        results = model.evaluate(X_test, y_test, verbose=0)
        loss, accuracy, precision, recall = results
        
        # Calcola F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"[Client {client_id}] Valutazione locale completata:")
        print(f"[Client {client_id}]   - Loss: {loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {f1_score:.4f}")
        print(f"[Client {client_id}]   - Campioni test: {len(X_test)}")
        
        # Metriche da restituire
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "test_samples": len(X_test)
        }
        
        return loss, len(X_test), metrics

def main():
    """
    Funzione principale per avviare il client SmartGrid.
    """
    global client_id, model, X_train, y_train, X_test, y_test, dataset_info
    
    # Verifica argomenti della riga di comando
    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        print("Esempio: python client.py 1")
        print("Client ID validi: 1-15 (corrispondenti ai file data1.csv - data15.csv)")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"AVVIO CLIENT SMARTGRID {client_id}")
    print("=" * 50)
    
    try:
        # 1. Carica i dati locali del client
        X_train, y_train, X_test, y_test, scaler, dataset_info = load_client_smartgrid_data(client_id)
        
        # 2. Crea il modello locale
        print(f"[Client {client_id}] Creazione modello locale...")
        model = create_smartgrid_client_model(X_train.shape[1])
        print(f"[Client {client_id}] Modello creato con {X_train.shape[1]} feature di input")
        
        # 3. Stampa riassunto del client
        print(f"[Client {client_id}] === RIASSUNTO CLIENT ===")
        print(f"[Client {client_id}] Dataset info:")
        for key, value in dataset_info.items():
            print(f"[Client {client_id}]   - {key}: {value}")
        
        print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
        sys.stdout.flush()
        
        # 4. Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()