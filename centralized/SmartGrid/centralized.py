import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import time
import sys
import os

def load_centralized_smartgrid_data():
    """
    Carica e unisce tutti i dati SmartGrid per l'addestramento centralizzato.
    Simula il caso tradizionale dove tutti i dati sono disponibili centralmente.
    
    Returns:
        Tuple con (X, y, dataset_info)
    """
    print("=== CARICAMENTO DATASET SMARTGRID CENTRALIZZATO ===")

    # Directory contenente questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
    
    # Gestione preliminare valori infiniti
    print(f"\nPulizia preliminare valori infiniti...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Informazioni del dataset per il riassunto finale
    dataset_info = {
        'files_loaded': files_loaded,
        'total_files': len(files_loaded),
        'total_samples': len(df_combined),
        'features': X.shape[1],
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio
    }
    
    print("=" * 60)
    
    return X, y, dataset_info

def split_train_validation_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    STEP 1: Suddivide il dataset in train (70%), validation (15%) e test (15%).
    Questo viene fatto PRIMA di qualsiasi preprocessing per evitare data leakage.
    
    Args:
        X: Feature del dataset
        y: Target del dataset
        train_size: Proporzione per il training set (default: 0.7)
        val_size: Proporzione per il validation set (default: 0.15)
        test_size: Proporzione per il test set (default: 0.15)
        random_state: Seed per riproducibilità
    
    Returns:
        Tuple con (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"=== STEP 1: SUDDIVISIONE TRAIN/VALIDATION/TEST (PRIMA DEL PREPROCESSING) ===")
    
    # Verifica che le proporzioni sommino a 1
    total_size = train_size + val_size + test_size
    if abs(total_size - 1.0) > 0.001:
        raise ValueError(f"Le proporzioni devono sommare a 1.0, ricevuto: {total_size}")
    
    # Prima divisione: separa il training set dal resto (validation + test)
    temp_val_test_size = val_size + test_size  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_val_test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Seconda divisione: separa validation e test dal resto
    # Calcola la proporzione relativa tra validation e test
    relative_test_size = test_size / temp_val_test_size  # 0.15 / 0.30 = 0.5
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Stampa informazioni sulla suddivisione
    print(f"  - Training set: {len(X_train)} campioni ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Validation set: {len(X_val)} campioni ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  - Test set: {len(X_test)} campioni ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verifica distribuzione delle classi
    train_attack_ratio = y_train.mean()
    val_attack_ratio = y_val.mean()
    test_attack_ratio = y_test.mean()
    
    print(f"  - Proporzione attacchi training: {train_attack_ratio*100:.2f}%")
    print(f"  - Proporzione attacchi validation: {val_attack_ratio*100:.2f}%")
    print(f"  - Proporzione attacchi test: {test_attack_ratio*100:.2f}%")
    
    print("=" * 60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessing_pipeline(variance_threshold=0.95):
    """
    Crea la pipeline di preprocessing scikit-learn.
    STEP 2-3: Imputazione e Normalizzazione
    
    Args:
        variance_threshold: Soglia per PCA
    
    Returns:
        Pipeline di preprocessing
    """
    print(f"=== CREAZIONE PIPELINE PREPROCESSING ===")
    print(f"  - Step 2: Imputazione con mediana")
    print(f"  - Step 3: Normalizzazione con StandardScaler")
    print(f"  - PCA verrà applicata successivamente dopo SMOTE")
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    print("=" * 60)
    
    return pipeline

def apply_smote_balancing(X_train, y_train):
    """
    STEP 4: Applica SMOTE per bilanciare le classi solo sul training set.
    
    Args:
        X_train: Feature di training (già preprocessate)
        y_train: Target di training
    
    Returns:
        Tuple (X_train_balanced, y_train_balanced)
    """
    print(f"=== STEP 4: BILANCIAMENTO CLASSI CON SMOTE ===")
    
    # Calcola il rapporto di squilibrio nel training set
    train_attack_ratio = y_train.mean()
    minority_class_ratio = min(train_attack_ratio, 1 - train_attack_ratio)
    
    print(f"  - Distribuzione training PRIMA del bilanciamento:")
    print(f"    - Classe 0 (Natural): {(y_train == 0).sum()} campioni ({(1-train_attack_ratio)*100:.2f}%)")
    print(f"    - Classe 1 (Attack): {(y_train == 1).sum()} campioni ({train_attack_ratio*100:.2f}%)")
    print(f"    - Rapporto classe minoritaria: {minority_class_ratio*100:.2f}%")
    
    # Applica SMOTE solo se lo squilibrio è significativo (< 40%)
    if minority_class_ratio < 0.4:
        print(f"  - Squilibrio significativo rilevato, applicazione SMOTE...")
        
        # Configura SMOTE
        smote = SMOTE(
            sampling_strategy='auto',  # Bilancia automaticamente alla classe maggioritaria
            random_state=42,          # Per riproducibilità
            k_neighbors=5             # Numero di vicini per generare campioni sintetici
        )
        
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Statistiche dopo il bilanciamento
            print(f"  - SMOTE applicato con successo")
            print(f"  - Distribuzione training DOPO il bilanciamento:")
            print(f"    - Classe 0 (Natural): {(y_train_balanced == 0).sum()} campioni ({(y_train_balanced == 0).mean()*100:.2f}%)")
            print(f"    - Classe 1 (Attack): {(y_train_balanced == 1).sum()} campioni ({(y_train_balanced == 1).mean()*100:.2f}%)")
            print(f"    - Campioni sintetici generati: {len(X_train_balanced) - len(X_train)}")
            
            print("=" * 60)
            return X_train_balanced, y_train_balanced
            
        except Exception as e:
            print(f"  - Errore durante l'applicazione di SMOTE: {e}")
            print(f"  - Continuazione con dati originali sbilanciati")
            print("=" * 60)
            return X_train, y_train
    else:
        print(f"  - Squilibrio accettabile, SMOTE non necessario")
        print("=" * 60)
        return X_train, y_train

def apply_pca_reduction(X_train, X_val, X_test, variance_threshold=0.95):
    """
    STEP 5: Applica PCA per riduzione dimensionale DOPO SMOTE.
    Fit della PCA solo sul training set, transform su tutti i set.
    
    Args:
        X_train: Dati di training (dopo SMOTE)
        X_val: Dati di validation (preprocessati)
        X_test: Dati di test (preprocessati)
        variance_threshold: Soglia di varianza cumulativa
    
    Returns:
        Tuple (X_train_pca, X_val_pca, X_test_pca, pca_object, n_components_selected)
    """
    print(f"=== STEP 5: RIDUZIONE DIMENSIONALE CON PCA ===")
    
    original_features = X_train.shape[1]
    print(f"  - Feature originali: {original_features}")
    print(f"  - Soglia varianza cumulativa: {variance_threshold*100:.1f}%")
    
    # Prima esecuzione: PCA completa per analizzare la varianza
    pca_full = PCA()
    pca_full.fit(X_train)  # Fit solo sui dati di training (inclusi quelli generati da SMOTE)
    
    # Calcola varianza cumulativa
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Trova il numero di componenti necessarie per raggiungere la soglia
    n_components_selected = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Assicurati che il numero di componenti sia valido
    n_components_selected = min(n_components_selected, original_features, len(X_train))
    
    print(f"  - Componenti selezionate: {n_components_selected}")
    print(f"  - Varianza spiegata con {n_components_selected} componenti: {cumulative_variance[n_components_selected-1]*100:.2f}%")
    print(f"  - Riduzione dimensionalità: {original_features} → {n_components_selected} ({(1-n_components_selected/original_features)*100:.1f}% riduzione)")
    
    # Mostra varianza spiegata dalle prime 10 componenti
    print(f"  - Varianza spiegata dalle prime 10 componenti:")
    for i in range(min(10, len(pca_full.explained_variance_ratio_))):
        print(f"    - PC{i+1}: {pca_full.explained_variance_ratio_[i]*100:.2f}% (cumulativa: {cumulative_variance[i]*100:.2f}%)")
    
    # Seconda esecuzione: PCA con numero ottimale di componenti
    pca_optimal = PCA(n_components=n_components_selected)
    
    # Fit della PCA solo sui dati di training
    X_train_pca = pca_optimal.fit_transform(X_train)
    
    # Transform su validation e test usando la PCA fitted sul training
    X_val_pca = pca_optimal.transform(X_val)
    X_test_pca = pca_optimal.transform(X_test)
    
    print(f"  - Shape dopo PCA:")
    print(f"    - Training: {X_train_pca.shape}")
    print(f"    - Validation: {X_val_pca.shape}")
    print(f"    - Test: {X_test_pca.shape}")
    
    print("=" * 60)
    
    return X_train_pca, X_val_pca, X_test_pca, pca_optimal, n_components_selected

def create_smartgrid_model(input_shape):
    """
    Crea il modello per la classificazione binaria SmartGrid.
    Architettura identica a quella della versione MNIST centralizzata per consistenza.
    
    Args:
        input_shape: Numero di feature in input (dopo PCA)
    
    Returns:
        Modello Keras compilato
    """
    print("=== CREAZIONE MODELLO SMARTGRID ===")
    
    # Crea il modello (architettura semplice per classificazione binaria)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),    # Layer di input (feature ridotte da PCA)
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
    print(f"Input shape del modello: {input_shape} feature (dopo riduzione PCA)")
    print("=" * 60)
    
    return model

def train_smartgrid_model(model, X_train, y_train, X_val, y_val):
    """
    Addestra il modello SmartGrid sui dati centralizzati.
    Utilizza validation set per monitoraggio durante l'addestramento.
    
    Args:
        model: Modello Keras da addestrare
        X_train, y_train: Dati di training (bilanciati con SMOTE, con PCA applicata)
        X_val, y_val: Dati di validation per monitoraggio (sbilanciati, distribuzione reale, con PCA applicata)
    
    Returns:
        History dell'addestramento
    """
    print("=== ADDESTRAMENTO CENTRALIZZATO SMARTGRID ===")
    
    # Configurazione identica alla versione MNIST
    epochs = 66 # calcolato come (200 round * 5 epoche per round) / 15 client in modo tale da avere un confronto equo con la versione federata
    batch_size = 32  # Stesso batch size di MNIST
    
    print(f"Configurazione addestramento:")
    print(f"  - Epoche: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Campioni training: {len(X_train)}")
    print(f"  - Campioni validation: {len(X_val)}")
    print(f"  - Feature in input (post-PCA): {X_train.shape[1]}")
    print(f"  - Batch per epoca: {len(X_train) // batch_size}")
    
    # Distribuzione delle classi nei set di training e validation
    train_attacks = y_train.sum()
    train_naturals = (y_train == 0).sum()
    val_attacks = y_val.sum()
    val_naturals = (y_val == 0).sum()
    
    print(f"  - Distribuzione training: {train_attacks} attacchi, {train_naturals} naturali")
    print(f"  - Distribuzione validation: {val_attacks} attacchi, {val_naturals} naturali")
    
    print("=" * 60)
    
    # Registra il tempo di inizio
    start_time = time.time()
    
    # Addestra il modello con validation data per monitoraggio
    print("Inizio addestramento...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),  # Validation per monitoraggio
        verbose=1  # Mostra il progresso dettagliato
    )
    
    # Calcola il tempo totale
    training_time = time.time() - start_time
    
    print(f"\nAddestramento completato in {training_time:.2f} secondi")
    print("=" * 60)
    
    return history

def evaluate_smartgrid_model(model, X_test, y_test, set_name="Test"):
    """
    Valuta il modello SmartGrid sui dati specificati.
    
    Args:
        model: Modello addestrato
        X_test, y_test: Dati di test (con PCA applicata)
        set_name: Nome del set per logging (default: "Test")
    
    Returns:
        Tuple con (loss, accuracy)
    """
    print(f"=== VALUTAZIONE FINALE SMARTGRID - {set_name.upper()} SET ===")
    
    # Valutazione finale
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Risultati finali {set_name}:")
    print(f"  - {set_name} Loss: {loss:.4f}")
    print(f"  - {set_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Campioni di {set_name.lower()} utilizzati: {len(X_test)}")
    print(f"  - Feature utilizzate (post-PCA): {X_test.shape[1]}")
    
    # Predizioni per analisi dettagliata
    predictions_prob = model.predict(X_test, verbose=0)
    predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
    
    # Analisi per classe
    print(f"\nAccuracy per classe ({set_name}):")
    
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
    
    # Matrice di confusione
    true_negatives = np.sum((y_test == 0) & (predictions_binary == 0))
    false_positives = np.sum((y_test == 0) & (predictions_binary == 1))
    false_negatives = np.sum((y_test == 1) & (predictions_binary == 0))
    true_positives = np.sum((y_test == 1) & (predictions_binary == 1))
    
    print(f"\nMatrice di confusione ({set_name}):")
    print(f"  - True Negatives (TN): {true_negatives}")
    print(f"  - False Positives (FP): {false_positives}") 
    print(f"  - False Negatives (FN): {false_negatives}")
    print(f"  - True Positives (TP): {true_positives}")
    
    # Metriche aggiuntive per sistemi di sicurezza
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetriche aggiuntive ({set_name}):")
    print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    print("=" * 60)
    
    return loss, accuracy

def print_training_summary(history, final_loss, final_accuracy, dataset_info, pca_components):
    """
    Stampa un riassunto dell'addestramento SmartGrid.
    Formato identico alla versione MNIST per facilità di confronto.
    
    Args:
        history: History dell'addestramento
        final_loss, final_accuracy: Metriche finali del test set
        dataset_info: Informazioni sul dataset
        pca_components: Numero di componenti PCA selezionate
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
    print(f"  - Loss finale (Test): {final_loss:.4f}")
    print(f"  - Accuracy finale (Test): {final_accuracy:.4f}")
    print(f"  - Miglioramento accuracy: {(final_accuracy - train_accuracy[0]):.4f}")
    
    # Informazioni sul dataset utilizzato
    print(f"\nInformazioni dataset:")
    print(f"  - File utilizzati: {dataset_info['total_files']} (data{min(dataset_info['files_loaded'])}.csv - data{max(dataset_info['files_loaded'])}.csv)")
    print(f"  - Campioni totali processati: {dataset_info['total_samples']}")
    print(f"  - Feature originali: {dataset_info['features']}")
    print(f"  - Feature dopo PCA: {pca_components}")
    print(f"  - Riduzione dimensionalità: {(1-pca_components/dataset_info['features'])*100:.1f}%")
    print(f"  - Proporzione attacchi: {dataset_info['attack_ratio']*100:.2f}%")
    print(f"  - Suddivisione: 70% train, 15% validation, 15% test")
    print(f"  - Pipeline preprocessing: Split → Imputazione → Normalizzazione → SMOTE → PCA")
    
    print("\n" + "=" * 70)
    print("ADDESTRAMENTO CENTRALIZZATO SMARTGRID CON PIPELINE COMPLETATO")
    print("Ora puoi confrontare questi risultati con l'approccio federato.")
    print("=" * 70)

def main():
    """
    Funzione principale per l'addestramento centralizzato SmartGrid.
    Implementa la pipeline corretta di preprocessing.
    """
    print("INIZIO ADDESTRAMENTO CENTRALIZZATO SMARTGRID CON PIPELINE CORRETTA")
    print("Questo script addestra un modello di rilevamento intrusioni SmartGrid")
    print("usando un approccio centralizzato con pipeline di preprocessing corretta.")
    print("Pipeline: Split → Imputazione → Normalizzazione → SMOTE → PCA")
    print("Suddivisione: 70% train, 15% validation, 15% test")
    print("=" * 70)
    
    try:
        # 1. Carica i dati grezzi
        X, y, dataset_info = load_centralized_smartgrid_data()
        
        # STEP 1: Suddividi in train/validation/test CON DATI ORIGINALI (no preprocessing)
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_validation_test(
            X, y,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        # STEP 2-3: Crea e applica pipeline di preprocessing (Imputazione + Normalizzazione)
        preprocessing_pipeline = create_preprocessing_pipeline()
        
        print(f"=== STEP 2-3: APPLICAZIONE PIPELINE PREPROCESSING ===")
        print(f"  - Fit della pipeline sui dati di training")
        print(f"  - Transform su training, validation e test")
        
        # Fit della pipeline SOLO sui dati di training
        X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_raw)
        
        # Transform su validation e test con i parametri appresi dal training
        X_val_preprocessed = preprocessing_pipeline.transform(X_val_raw)
        X_test_preprocessed = preprocessing_pipeline.transform(X_test_raw)
        
        print(f"  - Training preprocessed shape: {X_train_preprocessed.shape}")
        print(f"  - Validation preprocessed shape: {X_val_preprocessed.shape}")
        print(f"  - Test preprocessed shape: {X_test_preprocessed.shape}")
        print("=" * 60)
        
        # STEP 4: Applica SMOTE solo sul training set
        X_train_balanced, y_train_balanced = apply_smote_balancing(X_train_preprocessed, y_train)
        
        # STEP 5: Applica PCA (fit sui dati bilanciati, transform su tutti)
        X_train_final, X_val_final, X_test_final, pca_object, n_components = apply_pca_reduction(
            X_train_balanced, X_val_preprocessed, X_test_preprocessed,
            variance_threshold=0.95
        )
        
        # 6. Crea il modello (con input shape delle feature ridotte da PCA)
        model = create_smartgrid_model(n_components)
        
        # 7. Addestra il modello
        history = train_smartgrid_model(model, X_train_final, y_train_balanced, X_val_final, y_val)
        
        # 8. Valuta il modello sul validation set (per completezza)
        print("\n" + "=" * 70)
        evaluate_smartgrid_model(model, X_val_final, y_val, "Validation")
        
        # 9. Valuta il modello sul test set (valutazione finale)
        print("\n" + "=" * 70)
        final_loss, final_accuracy = evaluate_smartgrid_model(model, X_test_final, y_test, "Test")
        
        # 10. Stampa riassunto finale
        print_training_summary(history, final_loss, final_accuracy, dataset_info, n_components)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()