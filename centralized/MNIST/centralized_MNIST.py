import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys

def load_centralized_data():
    """
    Carica e prepara il dataset MNIST per l'addestramento centralizzato.
    Usa la stessa quantità di dati che verrebbe distribuita tra i client federati.
    
    Returns:
        Tuple con (x_train, y_train, x_test, y_test)
    """
    print("=== CARICAMENTO DATASET MNIST CENTRALIZZATO ===")
    
    # Carica il dataset MNIST completo
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()
    
    # Normalizza le immagini (valori tra 0 e 1)
    x_train_full = x_train_full / 255.0
    x_test_full = x_test_full / 255.0
    
    # Usa la stessa quantità di dati del caso federato per un confronto equo
    # Nel federato: 3 client × 1000 campioni = 3000 campioni training totali
    # Nel federato: 3 client × 200 campioni = 600 campioni test totali
    total_train_samples = 3000
    total_test_samples = 600
    
    # Estrai i primi N campioni per training e test
    x_train = x_train_full[:total_train_samples]
    y_train = y_train_full[:total_train_samples]
    x_test = x_test_full[:total_test_samples]
    y_test = y_test_full[:total_test_samples]
    
    print(f"Dati di training: {len(x_train)} campioni")
    print(f"Dati di test: {len(x_test)} campioni")
    print(f"Classi nel dataset: {np.unique(y_train)}")
    print(f"Distribuzione delle classi nel training:")
    
    # Mostra la distribuzione delle classi
    unique, counts = np.unique(y_train, return_counts=True)
    for classe, count in zip(unique, counts):
        print(f"  Classe {classe}: {count} campioni ({count/len(y_train)*100:.1f}%)")
    
    print("=" * 50)
    
    return x_train, y_train, x_test, y_test

def create_model():
    """
    Crea il modello di rete neurale.
    Identico a quello usato nell'approccio federato per garantire un confronto equo.
    
    Returns:
        Modello Keras compilato
    """
    print("=== CREAZIONE MODELLO ===")
    
    # Crea il modello (identico a quello federato)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # Appiattisce l'immagine 28x28 in vettore 784
        keras.layers.Dense(128, activation='relu'),   # Layer denso con 128 neuroni e attivazione ReLU
        keras.layers.Dense(10)                        # Layer di output con 10 neuroni (una per classe)
    ])
    
    # Compila il modello
    model.compile(
        optimizer='adam',                                                    # Ottimizzatore Adam
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Loss per classificazione multi-classe
        metrics=['accuracy']                                                 # Metrica di valutazione
    )
    
    print("Architettura del modello:")
    model.summary()
    print("=" * 50)
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Addestra il modello sui dati centralizzati.
    
    Args:
        model: Modello Keras da addestrare
        x_train, y_train: Dati di training
        x_test, y_test: Dati di test
    
    Returns:
        History dell'addestramento
    """
    print("=== ADDESTRAMENTO CENTRALIZZATO ===")
    
    # Calcola il numero equivalente di epoche del caso federato
    # Federato: 5 round × 1 epoca per client = 5 epoche equivalenti
    epochs = 5
    batch_size = 32  # Stesso batch size del federato
    
    print(f"Configurazione addestramento:")
    print(f"  - Epoche: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Campioni per epoca: {len(x_train)}")
    print(f"  - Batch per epoca: {len(x_train) // batch_size}")
    print("=" * 50)
    
    # Registra il tempo di inizio
    start_time = time.time()
    
    # Addestra il modello
    print("Inizio addestramento...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),  # Valutazione ad ogni epoca
        verbose=1  # Mostra il progresso
    )
    
    # Calcola il tempo totale
    training_time = time.time() - start_time
    
    print(f"\nAddestramento completato in {training_time:.2f} secondi")
    print("=" * 50)
    
    return history

def evaluate_model(model, x_test, y_test):
    """
    Valuta il modello addestrato sui dati di test.
    
    Args:
        model: Modello addestrato
        x_test, y_test: Dati di test
    
    Returns:
        Tuple con (loss, accuracy)
    """
    print("=== VALUTAZIONE FINALE ===")
    
    # Valutazione finale
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"Risultati finali:")
    print(f"  - Test Loss: {loss:.4f}")
    print(f"  - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Campioni di test utilizzati: {len(x_test)}")
    
    # Predizioni per analisi aggiuntiva
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calcola accuracy per classe
    print(f"\nAccuracy per classe:")
    for classe in range(10):
        # Trova tutti i campioni di questa classe
        class_mask = (y_test == classe)
        if np.sum(class_mask) > 0:
            # Calcola accuracy per questa classe specifica
            class_predictions = predicted_classes[class_mask]
            class_accuracy = np.mean(class_predictions == classe)
            class_count = np.sum(class_mask)
            print(f"  Classe {classe}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} campioni")
    
    print("=" * 50)
    
    return loss, accuracy

def print_training_summary(history, final_loss, final_accuracy):
    """
    Stampa un riassunto dell'addestramento per facilitare il confronto.
    
    Args:
        history: History dell'addestramento
        final_loss, final_accuracy: Metriche finali
    """
    print("=== RIASSUNTO ADDESTRAMENTO CENTRALIZZATO ===")
    
    # Estrai le metriche dalla history
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    
    print(f"Evoluzione delle metriche per epoca:")
    print(f"{'Epoca':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 60)
    
    for epoch in range(len(train_loss)):
        print(f"{epoch+1:<6} {train_loss[epoch]:<12.4f} {train_accuracy[epoch]:<12.4f} "
              f"{val_loss[epoch]:<12.4f} {val_accuracy[epoch]:<12.4f}")
    
    print(f"\nRisultati finali:")
    print(f"  - Loss finale: {final_loss:.4f}")
    print(f"  - Accuracy finale: {final_accuracy:.4f}")
    print(f"  - Miglioramento accuracy: {(final_accuracy - train_accuracy[0]):.4f}")
    
    print("\n" + "=" * 60)
    print("ADDESTRAMENTO CENTRALIZZATO COMPLETATO")
    print("Ora puoi confrontare questi risultati con l'approccio federato.")
    print("=" * 60)

def main():
    """
    Funzione principale per l'addestramento centralizzato.
    """
    print("INIZIO ADDESTRAMENTO CENTRALIZZATO MNIST")
    print("Questo script addestra un modello di classificazione MNIST")
    print("usando un approccio centralizzato tradizionale.")
    print("=" * 60)
    
    try:
        # 1. Carica i dati
        x_train, y_train, x_test, y_test = load_centralized_data()
        
        # 2. Crea il modello
        model = create_model()
        
        # 3. Addestra il modello
        history = train_model(model, x_train, y_train, x_test, y_test)
        
        # 4. Valuta il modello
        final_loss, final_accuracy = evaluate_model(model, x_test, y_test)
        
        # 5. Stampa riassunto
        print_training_summary(history, final_loss, final_accuracy)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()