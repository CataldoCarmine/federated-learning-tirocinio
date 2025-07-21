import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ottieni l'ID del client
client_id = int(sys.argv[1])

# Carica il CSV del client
df = pd.read_csv(f"data/SmartGrid/data{client_id}.csv")

# Prepara X e y
X = df.drop(columns=["marker"])
y = (df["marker"] != "Natural").astype(int)

# Pulizia dei dati
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]

# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split in train e test
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Crea il modello (rete semplice per classificazione binaria)
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam",
              loss=keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

# Definizione del client Flower
class SmartGridClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {client_id}] Ricevuti parametri, avvio addestramento...")
        sys.stdout.flush()
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        print(f"[Client {client_id}] Addestramento completato.")
        sys.stdout.flush()
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}

# Avvia il client
if __name__ == "__main__":
    print(f"[Client {client_id}] In esecuzione...")
    sys.stdout.flush()
    fl.client.start_numpy_client(server_address="localhost:8080", client=SmartGridClient())
