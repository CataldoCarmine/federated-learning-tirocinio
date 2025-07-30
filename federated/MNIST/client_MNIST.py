import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np

def load_client_data(client_id, num_clients=3):
    """
    Carica e distribuisce i dati MNIST tra i client.
    Ogni client riceve una porzione diversa e non sovrapposta del dataset.
    
    Args:
        client_id: ID del client (0, 1, 2, ...)
        num_clients: Numero totale di client
    
    Returns:
        Tuple con (x_train, y_train, x_test, y_test) per questo client
    """
    # Carica il dataset MNIST completo
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()
    x_train_full, x_test_full = x_train_full / 255.0, x_test_full / 255.0
    
    # Riduci il dataset totale per velocizzare i test
    total_train_samples = 3000  # 1000 campioni per client (3 client)
    total_test_samples = 600    # 200 campioni per client (3 client)
    
    x_train_reduced = x_train_full[:total_train_samples]
    y_train_reduced = y_train_full[:total_train_samples]
    x_test_reduced = x_test_full[:total_test_samples]
    y_test_reduced = y_test_full[:total_test_samples]
    
    # Calcola gli indici per questo client
    train_samples_per_client = total_train_samples // num_clients
    test_samples_per_client = total_test_samples // num_clients
    
    # Indici di inizio e fine per i dati di training
    train_start = client_id * train_samples_per_client
    train_end = (client_id + 1) * train_samples_per_client
    
    # Indici di inizio e fine per i dati di test
    test_start = client_id * test_samples_per_client
    test_end = (client_id + 1) * test_samples_per_client
    
    # Estrai i dati specifici per questo client
    x_train_client = x_train_reduced[train_start:train_end]
    y_train_client = y_train_reduced[train_start:train_end]
    x_test_client = x_test_reduced[test_start:test_end]
    y_test_client = y_test_reduced[test_start:test_end]
    
    print(f"[Client {client_id}] Dati ricevuti:")
    print(f"  - Training samples: {len(x_train_client)} (indici {train_start}-{train_end-1})")
    print(f"  - Test samples: {len(x_test_client)} (indici {test_start}-{test_end-1})")
    print(f"  - Classi nel training set: {np.unique(y_train_client)}")
    
    return x_train_client, y_train_client, x_test_client, y_test_client

# Ottieni l'ID del client dalla riga di comando
if len(sys.argv) != 2:
    print("Uso: python client_MNIST.py <client_id>")
    print("Esempio: python client_MNIST.py 0")
    sys.exit(1)

client_id = int(sys.argv[1])
num_clients = 3  # Numero totale di client

# Carica i dati specifici per questo client
x_train, y_train, x_test, y_test = load_client_data(client_id, num_clients)

# Crea il modello (identico per tutti i client)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10)
])
model.compile("adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Definisci il client Flower
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Restituisce i pesi attuali del modello"""
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
        print(f"[Client {client_id}] Ricevuti nuovi parametri dal server, avvio addestramento per 1 epoca...")
        sys.stdout.flush()
        
        # Imposta i pesi ricevuti dal server
        model.set_weights(parameters)
        
        # Addestra il modello sui dati locali
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        print(f"[Client {client_id}] Addestramento completato. Loss finale: {history.history['loss'][-1]:.4f}")
        sys.stdout.flush()
        
        # Restituisce i pesi aggiornati e il numero di campioni utilizzati
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        """
        Valuta il modello sui dati di test locali del client.
        
        Args:
            parameters: Pesi del modello da valutare
            config: Configurazione della valutazione
        
        Returns:
            Tuple con (loss, numero_campioni, metriche)
        """
        # Imposta i pesi da valutare
        model.set_weights(parameters)
        
        # Valuta sui dati di test locali
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        print(f"[Client {client_id}] Valutazione locale - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(x_test), {"accuracy": accuracy}

# Avvia il client
if __name__ == '__main__':
    print(f"[Client {client_id}] Avviando il client...")
    print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
    sys.stdout.flush()
    
    # Connessione al server Flower
    fl.client.start_numpy_client(
        server_address="localhost:8080", 
        client=MnistClient()
    )