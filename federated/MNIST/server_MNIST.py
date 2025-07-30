import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
import sys

def get_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server.
    Usa un dataset di test separato per valutare il modello aggregato.
    
    Returns:
        Funzione di valutazione che può essere usata dal server
    """
    # Carica il dataset di test globale (diverso da quello dei client)
    (_, _), (x_test_global, y_test_global) = tf.keras.datasets.mnist.load_data()
    x_test_global = x_test_global / 255.0
    
    # Usa un subset per il test globale (diverso da quello distribuito ai client)
    x_test_global = x_test_global[600:1000]  # Campioni 600-999 (non usati dai client)
    y_test_global = y_test_global[600:1000]
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione che viene chiamata ad ogni round.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato
            config: Configurazione
        
        Returns:
            Tuple con (loss, metriche)
        """
        # Crea il modello base (identico a quello dei client)
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10)
        ])
        
        # Compila il modello
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Imposta i pesi aggregati ricevuti dai client
        model.set_weights(parameters)
        
        # Valutazione sul dataset globale
        loss, accuracy = model.evaluate(x_test_global, y_test_global, verbose=0)
        
        print(f"\n=== ROUND {server_round} - VALUTAZIONE GLOBALE ===")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Campioni di test utilizzati: {len(x_test_global)}")
        print("=" * 50)
        sys.stdout.flush()
        
        return loss, {"accuracy": accuracy}
    
    return evaluate

def main():
    """Funzione principale per avviare il server federato"""
    print("=== AVVIO SERVER FEDERATO MNIST ===")
    print("Configurazione:")
    print("  - Numero di round: 5")
    print("  - Client minimi per il training: 3")
    print("  - Client minimi disponibili: 3")
    print("  - Strategia: FedAvg (Federated Averaging)")
    print("=" * 40)
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Strategia Federated Averaging
    strategy = FedAvg(
        fraction_fit=1.0,           # Usa tutti i client disponibili per il training
        fraction_evaluate=1.0,      # Usa tutti i client disponibili per la valutazione
        min_fit_clients=3,          # Numero minimo di client per iniziare il training
        min_evaluate_clients=3,     # Numero minimo di client per la valutazione
        min_available_clients=3,    # Numero minimo di client che devono essere connessi
        evaluate_fn=get_evaluate_fn()  # Funzione di valutazione globale
    )
    
    print("Server in attesa di client...")
    print("Per connettere i client, esegui in terminali separati:")
    print("  python client_MNIST.py 0")
    print("  python client_MNIST.py 1")
    print("  python client_MNIST.py 2")
    print("\nIl training inizierà automaticamente quando tutti i client saranno connessi.")
    print("=" * 40)
    sys.stdout.flush()
    
    # Avvia il server
    fl.server.start_server(
        server_address="localhost:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()