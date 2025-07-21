import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf

# Crea una funzione di valutazione
def evaluate_model(server_round, parameters, fit_results):
    # Usa lo stesso dataset ridotto della versione centralizzata
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test, y_test = x_test[:200] / 255.0, y_test[:200]

    # Crea il modello base (uguale alla versione centralizzata)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])
    
    # Imposta i pesi ricevuti dal client
    model.set_weights(parameters)
    
    # Compila il modello prima della valutazione
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Valutazione
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Round {server_round}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, {"accuracy": accuracy}

# Configurazione del server
config = fl.server.ServerConfig(num_rounds=3)

# Strategia Federated Averaging
strategy = FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_fn=evaluate_model
)

# Avvia il server
fl.server.start_server(
    server_address="localhost:8080",
    config=config,
    strategy=strategy,
)
