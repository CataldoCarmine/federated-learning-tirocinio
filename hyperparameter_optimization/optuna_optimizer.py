"""
Script di ottimizzazione iperparametri centralizzato con Optuna per SmartGrid.
Trova i migliori parametri da applicare manualmente al sistema federato.

UTILIZZO:
    python hyperparameter_optimization/optuna_optimizer.py

OUTPUT:
    Stampa i migliori iperparametri da copiare manualmente in client.py e server.py
"""

import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, balanced_accuracy_score
import warnings
import os
import json
from datetime import datetime

# Configurazioni fisse (identiche al sistema federato)
PCA_COMPONENTS = 35
PCA_RANDOM_STATE = 42

warnings.filterwarnings('ignore')

class SmartGridOptunaOptimizer:
    """
    Ottimizzatore centralizzato per trovare i migliori iperparametri.
    Usa subset rappresentativo del dataset e modello DNN identico al federato.
    """
    
    def __init__(self, data_dir="data/SmartGrid"):
        """
        Inizializza l'ottimizzatore.
        
        Args:
            data_dir: Directory contenente i dati SmartGrid
        """
        self.data_dir = data_dir
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.best_params = None
        
        print("=== SMARTGRID OPTUNA OPTIMIZER ===")
        print(f"Directory dati: {data_dir}")
        print(f"PCA components: {PCA_COMPONENTS}")
        print("Obiettivo: Ottimizzazione iperparametri per sistema federato")
    
    def load_subset_data(self):
        """
        Carica un subset rappresentativo dei dati per l'ottimizzazione.
        Usa alcuni file client per avere dati diversificati.
        """
        print("\n=== CARICAMENTO SUBSET DATI ===")
        
        # Usa subset di client per rappresentatività
        client_files = ['data1.csv', 'data2.csv', 'data3.csv', 'data5.csv', 'data7.csv']
        df_list = []
        
        for file_name in client_files:
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df_list.append(df)
                    print(f"✅ Caricato {file_name}: {len(df)} campioni")
                except Exception as e:
                    print(f"❌ Errore caricando {file_name}: {e}")
            else:
                print(f"⚠️  File {file_name} non trovato")
        
        if not df_list:
            raise FileNotFoundError(f"Nessun file dati trovato in {self.data_dir}")
        
        # Combina tutti i dataframe
        df_combined = pd.concat(df_list, ignore_index=True)
        
        # Separa features e target
        X = df_combined.drop(columns=["marker"])
        y = (df_combined["marker"] != "Natural").astype(int)
        
        print(f"Dataset combinato: {len(df_combined)} campioni")
        print(f"Features: {X.shape[1]}")
        print(f"Distribuzione: {y.sum()} attacchi ({y.mean()*100:.1f}%), {(y==0).sum()} naturali")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """
        Applica preprocessing identico al sistema federato.
        
        Args:
            X: Features raw
            y: Target
        
        Returns:
            X_train, X_val, y_train, y_val preprocessati
        """
        print("\n=== PREPROCESSING DATI ===")
        
        # Split train/validation (80/20)
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train_raw)} campioni")
        print(f"Validation set: {len(X_val_raw)} campioni")
        
        # Pulizia dati (identica al federato)
        X_train_cleaned = self._clean_data(X_train_raw)
        X_val_cleaned = self._clean_data(X_val_raw)
        
        # Pipeline preprocessing (identica al federato)
        preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_cleaned)
        X_val_preprocessed = preprocessing_pipeline.transform(X_val_cleaned)
        
        # PCA fissa (identica al federato)
        X_train_final = self._apply_pca(X_train_preprocessed)
        X_val_final = self._apply_pca(X_val_preprocessed)
        
        print(f"Dopo preprocessing - Training: {X_train_final.shape}")
        print(f"Dopo preprocessing - Validation: {X_val_final.shape}")
        
        return X_train_final, X_val_final, y_train, y_val
    
    def _clean_data(self, X):
        """Pulizia dati identica al federato."""
        if hasattr(X, 'values'):
            X_array = X.values.copy()
        else:
            X_array = X.copy()
        
        # Rimuovi inf e NaN
        X_array = np.where(np.isinf(X_array), np.nan, X_array)
        X_array = np.where(np.abs(X_array) > 1e8, np.nan, X_array)
        X_array = np.where(np.abs(X_array) < 1e-12, 0, X_array)
        
        return X_array
    
    def _apply_pca(self, X_preprocessed):
        """Applica PCA identica al federato."""
        n_components = min(PCA_COMPONENTS, X_preprocessed.shape[1], len(X_preprocessed))
        
        # Stabilità numerica
        X_stable = np.clip(X_preprocessed, -1e6, 1e6)
        
        # Sostituisci NaN con mediana
        for col in range(X_stable.shape[1]):
            col_data = X_stable[:, col]
            finite_mask = np.isfinite(col_data)
            if np.any(finite_mask):
                median_val = np.median(col_data[finite_mask])
                X_stable[~finite_mask, col] = median_val
            else:
                X_stable[:, col] = 0
        
        # PCA
        try:
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
            X_pca = pca.fit_transform(X_stable)
            
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError("PCA output contiene NaN/inf")
            
            return X_pca
        except:
            # Fallback: usa prime n_components colonne
            return X_stable[:, :n_components]
    
    def create_model(self, trial):
        """
        Crea modello DNN con iperparametri da ottimizzare.
        ARCHITETTURA IDENTICA al sistema federato.
        
        Args:
            trial: Trial Optuna
        
        Returns:
            Modello compilato + dizionario parametri
        """
        # IPERPARAMETRI DA OTTIMIZZARE
        params = {
            # Architettura (neuroni per layer) - mantenendo 4 layer nascosti
            'neurons_layer_1': trial.suggest_int('neurons_layer_1', 32, 128, step=16),
            'neurons_layer_2': trial.suggest_int('neurons_layer_2', 16, 64, step=8),
            'neurons_layer_3': trial.suggest_int('neurons_layer_3', 8, 32, step=4),
            'neurons_layer_4': trial.suggest_int('neurons_layer_4', 4, 16, step=2),
            
            # Regolarizzazione
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1),
            'l2_reg': trial.suggest_float('l2_reg', 0.0001, 0.01, log=True),
            'extended_dropout': trial.suggest_categorical('extended_dropout', [True, False]),
            
            # Attivazione
            'activation_function': trial.suggest_categorical('activation_function', ['relu', 'leaky_relu', 'selu']),
            
            # Ottimizzatore
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
            
            # Training
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 10, 50, step=5)
        }
        
        # Funzione di attivazione
        if params['activation_function'] == 'leaky_relu':
            activation_layer = lambda: layers.LeakyReLU(alpha=0.1)
            initializer = 'he_normal'
        elif params['activation_function'] == 'selu':
            activation_layer = lambda: layers.Activation('selu')
            initializer = 'lecun_normal'
        else:  # relu
            activation_layer = lambda: layers.Activation('relu')
            initializer = 'he_normal'
        
        # Costruzione modello IDENTICO al federato
        model = keras.Sequential([
            # Input
            layers.Input(shape=(PCA_COMPONENTS,)),
            
            # Layer 1
            layers.Dense(params['neurons_layer_1'], 
                        kernel_regularizer=regularizers.l2(params['l2_reg']),
                        kernel_initializer=initializer),
            activation_layer(),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate']),
            
            # Layer 2
            layers.Dense(params['neurons_layer_2'], 
                        kernel_regularizer=regularizers.l2(params['l2_reg']),
                        kernel_initializer=initializer),
            activation_layer(),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate'] if params['extended_dropout'] else 0.0),
            
            # Layer 3
            layers.Dense(params['neurons_layer_3'], 
                        kernel_regularizer=regularizers.l2(params['l2_reg']),
                        kernel_initializer=initializer),
            activation_layer(),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate']),
            
            # Layer 4
            layers.Dense(params['neurons_layer_4'], 
                        kernel_regularizer=regularizers.l2(params['l2_reg']),
                        kernel_initializer=initializer),
            activation_layer(),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate'] * 0.75),
            
            # Output
            layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')
        ])
        
        # Ottimizzatore
        if params['optimizer_type'] == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=params['learning_rate'],
                weight_decay=params['l2_reg'],
                clipnorm=1.0
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params['learning_rate'],
                clipnorm=1.0
            )
        
        # Compilazione
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model, params
    
    def objective(self, trial):
        """
        Funzione obiettivo per Optuna.
        Addestra il modello e restituisce score da massimizzare.
        
        Args:
            trial: Trial Optuna
        
        Returns:
            Score da massimizzare (F1 + Balanced Accuracy) / 2
        """
        # Crea modello con parametri del trial
        model, params = self.create_model(trial)
        
        # Class weights per dataset sbilanciato
        unique_classes = np.unique(self.y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=self.y_train)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Callback per early stopping
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        ]
        
        try:
            # Training
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predizioni su validation set
            y_pred_prob = model.predict(self.X_val, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Metriche bilanciate
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            balanced_acc = balanced_accuracy_score(self.y_val, y_pred)
            
            # Score combinato da massimizzare
            score = (f1 + balanced_acc) / 2
            
            # Log ogni 10 trial
            if trial.number % 10 == 0:
                print(f"Trial {trial.number}: Score = {score:.4f} (F1={f1:.4f}, BalAcc={balanced_acc:.4f})")
            
            return score
            
        except Exception as e:
            print(f"Trial {trial.number} fallito: {e}")
            return 0.0
    
    def optimize(self, n_trials=100):
        """
        Esegue l'ottimizzazione Optuna.
        
        Args:
            n_trials: Numero di trial da eseguire
        
        Returns:
            Migliori parametri trovati
        """
        print(f"\n=== AVVIO OTTIMIZZAZIONE OPTUNA ===")
        print(f"Trial da eseguire: {n_trials}")
        print("Obiettivo: Massimizzare (F1-Score + Balanced Accuracy) / 2")
        
        # Carica e preprocessa dati
        X, y = self.load_subset_data()
        self.X_train, self.X_val, self.y_train, self.y_val = self.preprocess_data(X, y)
        
        # Crea studio Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print("\nInizio ottimizzazione...")
        
        # Esegui ottimizzazione
        study.optimize(self.objective, n_trials=n_trials)
        
        # Risultati
        self.best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n=== OTTIMIZZAZIONE COMPLETATA ===")
        print(f"Migliore score: {best_score:.4f}")
        print(f"Migliori parametri:")
        
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Salva risultati
        self._save_results(study)
        
        return self.best_params
    
    def _save_results(self, study):
        """Salva i risultati in file JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crea directory se non esiste
        os.makedirs("hyperparameter_optimization/results", exist_ok=True)
        
        # Salva risultati
        results = {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'timestamp': timestamp,
            'n_trials': len(study.trials),
            'pca_components': PCA_COMPONENTS
        }
        
        results_file = f"hyperparameter_optimization/results/optuna_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nRisultati salvati in: {results_file}")
    
    def print_manual_instructions(self):
        """
        Stampa le istruzioni per applicare manualmente i parametri ottimizzati.
        """
        if not self.best_params:
            print("❌ Esegui prima l'ottimizzazione!")
            return
        
        print("\n" + "="*80)
        print("📋 ISTRUZIONI PER APPLICAZIONE MANUALE")
        print("="*80)
        
        print("\n🔧 PARAMETRI DA COPIARE NEL CODICE FEDERATO:")
        print("-" * 50)
        
        # Stampa parametri in formato facile da copiare
        for param, value in self.best_params.items():
            if isinstance(value, str):
                print(f"{param.upper()} = '{value}'")
            else:
                print(f"{param.upper()} = {value}")
        
        print(f"\nPCA_COMPONENTS = {PCA_COMPONENTS}  # (non modificare)")
        
        print("\n📝 MODIFICHE DA FARE:")
        print("-" * 30)
        
        print("\n1️⃣ FILE: federated/SmartGrid/client.py")
        print("   TROVA le righe con queste variabili e SOSTITUISCI i valori:")
        print(f"   - ACTIVATION_FUNCTION = '{self.best_params['activation_function']}'")
        print(f"   - USE_ADAMW = {self.best_params['optimizer_type'] == 'adamw'}")
        print(f"   - EXTENDED_DROPOUT = {self.best_params['extended_dropout']}")
        
        print("\n   NELLA FUNZIONE create_smartgrid_dnn_model_static_architecture():")
        print(f"   - dropout_rate = {self.best_params['dropout_rate']}")
        print(f"   - l2_reg = {self.best_params['l2_reg']}")
        print(f"   - learning_rate = {self.best_params['learning_rate']}")
        
        print("   NELL'ARCHITETTURA del modello, SOSTITUISCI i neuroni:")
        print(f"   - Layer 1: {self.best_params['neurons_layer_1']} neuroni")
        print(f"   - Layer 2: {self.best_params['neurons_layer_2']} neuroni")
        print(f"   - Layer 3: {self.best_params['neurons_layer_3']} neuroni")
        print(f"   - Layer 4: {self.best_params['neurons_layer_4']} neuroni")
        
        print("   NELLA FUNZIONE fit() della classe SmartGridDNNClientFixed:")
        print(f"   - local_epochs = {self.best_params['epochs']}")
        print(f"   - batch_size = {self.best_params['batch_size']}")
        
        print("\n2️⃣ FILE: federated/SmartGrid/server.py")
        print("   TROVA le stesse variabili del client.py e applica gli STESSI valori:")
        print(f"   - ACTIVATION_FUNCTION = '{self.best_params['activation_function']}'")
        print(f"   - USE_ADAMW = {self.best_params['optimizer_type'] == 'adamw'}")
        print(f"   - EXTENDED_DROPOUT = {self.best_params['extended_dropout']}")
        print("   - Stessi neuroni per layer")
        print("   - Stesso dropout_rate e l2_reg")
        print("   - Stesso learning_rate")
        
        print("\n✅ DOPO LE MODIFICHE:")
        print("   1. Salva entrambi i file")
        print("   2. Avvia il server: python federated/SmartGrid/server.py")
        print("   3. Avvia i client: python federated/SmartGrid/client.py <client_id>")
        
        print("\n🎯 Il tuo sistema federato avrà ora iperparametri ottimizzati!")
        print("="*80)

def main():
    """Funzione principale per eseguire l'ottimizzazione."""
    # Verifica directory dati
    data_dir = "data/SmartGrid"
    if not os.path.exists(data_dir):
        print(f"❌ Directory {data_dir} non trovata!")
        print("Verifica che esista e contenga i file data1.csv, data2.csv, etc.")
        return
    
    # Crea ottimizzatore
    optimizer = SmartGridOptunaOptimizer(data_dir)
    
    # Esegui ottimizzazione (50 trial per test, aumenta per risultati migliori)
    n_trials = 50  # Modifica questo numero se vuoi più trial
    
    try:
        best_params = optimizer.optimize(n_trials)
        
        # Stampa istruzioni per applicazione manuale
        optimizer.print_manual_instructions()
        
    except Exception as e:
        print(f"❌ Errore durante ottimizzazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()