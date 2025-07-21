import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Riduci il dataset per simulare un client
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:200], y_test[:200]

# Crea il modello
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
        return model.get_weights()

    def fit(self, parameters, config):
        print("Ricevuti nuovi parametri dal server, avvio addestramento per 1 epoca...")
        sys.stdout.flush()
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        print("Addestramento completato, invio dei pesi aggiornati al server.")
        sys.stdout.flush()
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}


# Blocca principale per avviare il client
if __name__ == '__main__':
    print("Client in esecuzione, sto per connettermi al server...")
    sys.stdout.flush()
    fl.client.start_numpy_client(server_address="localhost:8080", client=MnistClient())
