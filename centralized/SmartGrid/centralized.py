import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import sys
import os

def load_centralized_smartgrid_data():
    """
    Carica e unisce tutti i dati SmartGrid per l'addestramento centralizzato.
    Simula il caso tradizionale dove tutti i dati sono disponibili centralmente.
    
    Returns:
        Tuple con (X_scaled, y, scaler, dataset_info)
    """
    print("=== CARICAMENTO DATASET SMARTGRID CENTRALIZZATO ===")

    # Path assoluto alla cartella dei dati
    data_dir = os.path.join(script_dir, "..", "..", "data", "SmartGrid")
    
    # Lista per contenere tutti i dataframe
    df_list = []
    files_loaded = []
    
    # Carica tutti i file CSV disponibili (data1.csv a data15.csv)
    for file_id in range(1, 16):
        file_path = os.path.join(data_dir, f"data{file_id}.csv")
        
        if os.path.exists(file_path):
            try:
                df_file = pd.read_csv(file_path)
                df_list.append(df_file)
                files_loaded.append(file_id)
                print(f"  - Caricato data{file_id}.csv: {len(df_file)} campioni")
            except Exception as e:
                print(f"  - Errore nel caricamento di data{file_id}.csv: {e}")
        else:
            print(f"  - File data{file_id}.csv non trovato")
    
    if not df_list:
        raise FileNotFoundError("Nessun file di dati SmartGrid trovato nella cartella data/SmartGrid/")
    
    # Unisci tutti i dataframe in un unico dataset centralizzato
    df_combined = pd.concat(df_list, ignore_index=True)
    
    print(f"\nDataset centralizzato combinato:")
    print(f"  - File caricati: {len(files_loaded)} ({files_loaded})")
    print(f"  - Totale campioni: {len(df_combined)}")
    print(f"  - Feature totali: {df_combined.shape[1] - 1}")  # -1 per escludere 'marker'
    
    # Separa feature e target
    X = df_combined.drop(columns=["marker"])
    y = (df_combined["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale
    
    # Statistiche del dataset
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    print(f"  - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"  - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    
    # Distribuzione delle classi per scenario (basata sulla colonna marker originale)
    marker_distribution = df_combined["marker"].value_counts()
    print(f"\nDistribuzione per tipo di scenario:")
    for marker, count in marker_distribution.items():
        percentage = (count / len(df_combined)) * 100
        print(f"  - {marker}: {count} campioni ({percentage:.2f}%)")
    
    # Pulizia dei dati: gestione di valori infiniti e NaN
    print(f"\nPulizia dei dati:")
    initial_samples = len(X)
    
    # Sostituisci valori infiniti con NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Conta i NaN per feature
    nan_counts = X.isnull().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        print(f"  - Valori NaN trovati: {total_nans}")
        features_with_nans = nan_counts[nan_counts > 0]
        print(f"  - Feature con NaN: {len(features_with_nans)}")
        if len(features_with_nans) <= 10:  # Mostra solo se poche
            for feature, count in features_with_nans.items():
                print(f"    - {feature}: {count} NaN")
    else:
        print(f"  - Nessun valore NaN trovato")
    
    # Rimuovi righe con NaN
    X.dropna(inplace=True)
    y = y.loc[X.index]
    
    final_samples = len(X)
    removed_samples = initial_samples - final_samples
    
    print(f"  - Campioni rimossi: {removed_samples}")
    print(f"  - Campioni finali: {final_samples}")
    
    if final_samples == 0:
        raise ValueError("Nessun campione valido rimasto dopo la pulizia dei dati")
    
    # Normalizzazione delle feature
    print(f"\nNormalizzazione feature...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Verifica della normalizzazione (mostra solo le prime 5 feature)
    feature_means = np.mean(X_scaled, axis=0)
    feature_stds = np.std(X_scaled, axis=0)
    
    print(f"  - Verifica normalizzazione (prime 5 feature):")
    print(f"    - Medie: {feature_means[:5]}")
    print(f"    - Deviazioni standard: {feature_stds[:5]}")
    
    # Informazioni del dataset per il riassunto finale
    dataset_info = {
        'files_loaded': files_loaded,
        'total_files': len(files_loaded),
        'initial_samples': initial_samples,
        'final_samples': final_samples,
        'removed_samples': removed_samples,
        'features': X.shape[1],
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio
    }
    
    print("=" * 60)
    
    return X_scaled, y, scaler, dataset_info

def create_smartgrid_model(input_shape):
    """
    Crea il modello per la classificazione binaria SmartGrid.
    Architettura identica a quella della versione MNIST centralizzata per consistenza.
    
    Args:
        input_shape: Numero di feature in input
    
    Returns:
        Modello Keras compilato
    """
    print("=== CREAZIONE MODELLO SMARTGRID ===")
    
    # Crea il modello (architettura semplice per classificazione binaria)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),    # Layer di input
        keras.layers.Dense(64, activation='relu'),   # Layer nascosto con 64 neuroni e ReLU
        keras.layers.Dense(1, activation='sigmoid')  # Layer di output con attivazione sigmoid per classificazione binaria
    ])
    
    # Compila il modello
    model.compile(
        optimizer='adam',                           # Ottimizzatore Adam (stesso di MNIST)
        loss=tf.keras.losses.BinaryCrossentropy(),  # Loss per classificazione binaria
        metrics=['accuracy']                        # Metrica principale (come MNIST)
    )
    
    print("Architettura del modello:")
    model.summary()
    print("=" * 60)
    
    return model

def train_smartgrid_model(model, X_train, y_train, X_test, y_test):
    """
    Addestra il modello SmartGrid sui dati centralizzati.
    Configurazione identica alla versione MNIST centralizzata.
    
    Args:
        model: Modello Keras da addestrare
        X_train, y_train: Dati di training
        X_test, y_test: Dati di test per validazione
    
    Returns:
        History dell'addestramento
    """
    print("=== ADDESTRAMENTO CENTRALIZZATO SMARTGRID ===")
    
    # Configurazione identica alla versione MNIST
    epochs = 5
    batch_size = 32  # Stesso batch size di MNIST
    
    print(f"Configurazione addestramento:")
    print(f"  - Epoche: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Campioni training: {len(X_train)}")
    print(f"  - Campioni test: {len(X_test)}")
    print(f"  - Batch per epoca: {len(X_train) // batch_size}")
    
    # Distribuzione delle classi nei set di training e test
    train_attacks = y_train.sum()
    train_naturals = (y_train == 0).sum()
    test_attacks = y_test.sum()
    test_naturals = (y_test == 0).sum()
    
    print(f"  - Distribuzione training: {train_attacks} attacchi, {train_naturals} naturali")
    print(f"  - Distribuzione test: {test_attacks} attacchi, {test_naturals} naturali")
    
    print("=" * 60)
    
    # Registra il tempo di inizio
    start_time = time.time()
    
    # Addestra il modello con validation data (come MNIST)
    print("Inizio addestramento...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),  # Validazione ad ogni epoca
        verbose=1  # Mostra il progresso dettagliato
    )
    
    # Calcola il tempo totale
    training_time = time.time() - start_time
    
    print(f"\nAddestramento completato in {training_time:.2f} secondi")
    print("=" * 60)
    
    return history

def evaluate_smartgrid_model(model, X_test, y_test):
    """
    Valuta il modello SmartGrid sui dati di test.
    Output simile alla versione MNIST per facilità di confronto.
    
    Args:
        model: Modello addestrato
        X_test, y_test: Dati di test
    
    Returns:
        Tuple con (loss, accuracy)
    """
    print("=== VALUTAZIONE FINALE SMARTGRID ===")
    
    # Valutazione finale
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Risultati finali:")
    print(f"  - Test Loss: {loss:.4f}")
    print(f"  - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Campioni di test utilizzati: {len(X_test)}")
    
    # Predizioni per analisi dettagliata (come MNIST con accuracy per classe)
    predictions_prob = model.predict(X_test, verbose=0)
    predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
    
    # Analisi per classe (simile all'analisi per classe di MNIST)
    print(f"\nAccuracy per classe:")
    
    # Classe 0 (Natural/Normale)
    natural_mask = (y_test == 0)
    if np.sum(natural_mask) > 0:
        natural_predictions = predictions_binary[natural_mask]
        natural_accuracy = np.mean(natural_predictions == 0)  # Predizioni corrette per classe 0
        natural_count = np.sum(natural_mask)
        print(f"  Classe 0 (Natural): {natural_accuracy:.4f} ({natural_accuracy*100:.2f}%) - {natural_count} campioni")
    
    # Classe 1 (Attack/Attacco)
    attack_mask = (y_test == 1)
    if np.sum(attack_mask) > 0:
        attack_predictions = predictions_binary[attack_mask]
        attack_accuracy = np.mean(attack_predictions == 1)  # Predizioni corrette per classe 1
        attack_count = np.sum(attack_mask)
        print(f"  Classe 1 (Attack): {attack_accuracy:.4f} ({attack_accuracy*100:.2f}%) - {attack_count} campioni")
    
    # Matrice di confusione semplificata (informazioni aggiuntive)
    true_negatives = np.sum((y_test == 0) & (predictions_binary == 0))
    false_positives = np.sum((y_test == 0) & (predictions_binary == 1))
    false_negatives = np.sum((y_test == 1) & (predictions_binary == 0))
    true_positives = np.sum((y_test == 1) & (predictions_binary == 1))
    
    print(f"\nMatrice di confusione:")
    print(f"  - True Negatives (TN): {true_negatives}")
    print(f"  - False Positives (FP): {false_positives}") 
    print(f"  - False Negatives (FN): {false_negatives}")
    print(f"  - True Positives (TP): {true_positives}")
    
    # Metriche aggiuntive per sistemi di sicurezza
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetriche aggiuntive:")
    print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    print("=" * 60)
    
    return loss, accuracy

def print_training_summary(history, final_loss, final_accuracy, dataset_info):
    """
    Stampa un riassunto dell'addestramento SmartGrid.
    Formato identico alla versione MNIST per facilità di confronto.
    
    Args:
        history: History dell'addestramento
        final_loss, final_accuracy: Metriche finali
        dataset_info: Informazioni sul dataset
    """
    print("=== RIASSUNTO ADDESTRAMENTO CENTRALIZZATO SMARTGRID ===")
    
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
    
    # Informazioni sul dataset utilizzato
    print(f"\nInformazioni dataset:")
    print(f"  - File utilizzati: {dataset_info['total_files']} (data{min(dataset_info['files_loaded'])}.csv - data{max(dataset_info['files_loaded'])}.csv)")
    print(f"  - Campioni totali processati: {dataset_info['initial_samples']}")
    print(f"  - Campioni utilizzati: {dataset_info['final_samples']}")
    print(f"  - Campioni rimossi (pulizia): {dataset_info['removed_samples']}")
    print(f"  - Feature utilizzate: {dataset_info['features']}")
    print(f"  - Proporzione attacchi: {dataset_info['attack_ratio']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("ADDESTRAMENTO CENTRALIZZATO SMARTGRID COMPLETATO")
    print("Ora puoi confrontare questi risultati con l'approccio federato.")
    print("=" * 70)

def main():
    """
    Funzione principale per l'addestramento centralizzato SmartGrid.
    Struttura identica alla versione MNIST centralizzata.
    """
    print("INIZIO ADDESTRAMENTO CENTRALIZZATO SMARTGRID")
    print("Questo script addestra un modello di rilevamento intrusioni SmartGrid")
    print("usando un approccio centralizzato tradizionale.")
    print("=" * 70)
    
    try:
        # 1. Carica e prepara tutti i dati centralizzati
        X_scaled, y, scaler, dataset_info = load_centralized_smartgrid_data()
        
        # 2. Suddividi in train/test (come MNIST)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.2,      # Stessa proporzione di MNIST  
            random_state=42,    # Stesso seed per riproducibilità
            stratify=y          # Mantieni la proporzione delle classi
        )
        
        print(f"Suddivisione train/test:")
        print(f"  - Training set: {len(X_train)} campioni")
        print(f"  - Test set: {len(X_test)} campioni")
        print(f"  - Proporzione attacchi training: {y_train.mean()*100:.2f}%")
        print(f"  - Proporzione attacchi test: {y_test.mean()*100:.2f}%")
        print("=" * 70)
        
        # 3. Crea il modello
        model = create_smartgrid_model(X_train.shape[1])
        
        # 4. Addestra il modello
        history = train_smartgrid_model(model, X_train, y_train, X_test, y_test)
        
        # 5. Valuta il modello
        final_loss, final_accuracy = evaluate_smartgrid_model(model, X_test, y_test)
        
        # 6. Stampa riassunto finale
        print_training_summary(history, final_loss, final_accuracy, dataset_info)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()