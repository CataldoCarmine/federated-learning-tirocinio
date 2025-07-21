import tensorflow as tf
from tensorflow import keras
import numpy as np

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizza le immagini
x_train, x_test= x_train / 255.0, x_test / 255.0

# Riduci il dataset per velocizzare i test
x_train_small, y_train_small = x_train[:1000], y_train[:1000]
x_test_small, y_test_small = x_test[:200], y_test[:200]

# Crea il modello
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compila il modello
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Addestra il modello
model.fit(x_train_small, y_train_small, epochs=5)

# Valuta il modello
loss, accuracy = model.evaluate(x_test_small, y_test_small)
print(f"Test accuracy: {accuracy}")
