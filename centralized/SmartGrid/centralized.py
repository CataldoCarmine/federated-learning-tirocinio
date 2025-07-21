import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carica e unisci tutti i CSV dei client
df_list = [pd.read_csv(f"data/SmartGrid/data{i}.csv") for i in range(1, 16)]
df = pd.concat(df_list, ignore_index=True)

# Pre-elaborazione: separa feature e target
X = df.drop(columns=["marker"])
y = (df["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale

# Rimuovi eventuali valori anomali
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]

# Standardizza le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Suddividi in train/test
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crea il modello
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compila il modello
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Addestra il modello
model.fit(x_train, y_train, epochs=5)

# Valuta il modello
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
