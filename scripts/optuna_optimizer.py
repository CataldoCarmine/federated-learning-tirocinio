"""
    Ottimizzatore centralizzato per trovare i migliori iperparametri.
    Usa subset rappresentativo del dataset e modello DNN identico al federato.
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
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Configurazioni fisse (identiche al sistema federato)
PCA_COMPONENTS = 21
PCA_RANDOM_STATE = 42

warnings.filterwarnings('ignore')

def clip_outliers_iqr(X, k=5.0):
    """
    Clippa gli outlier per ogni feature usando la regola dei quantili (IQR).
    """
    X_clipped = X.copy()
    for col in range(X_clipped.shape[1]):
        col_data = X_clipped[:, col]
        q1 = np.nanpercentile(col_data, 25)
        q3 = np.nanpercentile(col_data, 75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        X_clipped[:, col] = np.clip(col_data, lower, upper)
    return X_clipped

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
    """
    Rimuove le feature che sono costanti almeno al 99.9% (tutte uguali tranne lo 0.1%).
    """
    keep_mask = []
    n = X.shape[0]
    for col in range(X.shape[1]):
        col_data = X[:, col]
        # Conta la moda (valore più frequente)
        vals, counts = np.unique(col_data, return_counts=True)
        max_count = np.max(counts)
        ratio = max_count / n
        var = np.nanvar(col_data)
        # Tiene solo se NON è costante al 99.9% e varianza > threshold_var
        keep = not (ratio >= threshold_ratio or var < threshold_var)
        keep_mask.append(keep)
    keep_mask = np.array(keep_mask)
    return X[:, keep_mask], keep_mask


class SmartGridOptunaOptimizer:

    def __init__(self, data_dir="data/SmartGrid", random_state=42):
        """
        Inizializza l'ottimizzatore.
        
        Args:
            data_dir: Directory contenente i dati SmartGrid
            random_state: Seed per riproducibilità
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.best_params = None
        self.current_trial_count = 0
        
        print("=== SMARTGRID OPTUNA OPTIMIZER ===")
        print(f"PCA components: {PCA_COMPONENTS}")
        print(f"Random state: {random_state}")
    
    def verify_data_directory(self):
        """
        Verifica che la directory dei dati esista e contenga file CSV validi.
        """
        if not os.path.exists(self.data_dir):
            print(f"❌ ERRORE: Directory {self.data_dir} non trovata!")
            return False
        
        # Verifica che ci siano file CSV nella directory
        csv_files = [f for f in os.listdir(self.data_dir) if f.startswith('data') and f.endswith('.csv')]
        if not csv_files:
            print(f"❌ ERRORE: Nessun file CSV trovato in {self.data_dir}")
            return False
        
        print(f"✅ Directory verificata: {len(csv_files)} file CSV trovati")
        return True
    
    def load_subset_data(self):
        """
        Carica un subset rappresentativo dei dati per l'ottimizzazione.
        Usa alcuni file client per avere dati diversificati.
        """
        print("\n=== CARICAMENTO SUBSET DATI ===")
        
        # Usa subset di client per rappresentatività
        client_files = ['data1.csv', 'data2.csv', 'data3.csv', 'data5.csv', 'data7.csv']
        df_list = []
        successful_loads = []
        
        for file_name in client_files:
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Verifica che il dataframe non sia vuoto e abbia colonne necessarie
                    if len(df) > 0 and 'marker' in df.columns:
                        df_list.append(df)
                        successful_loads.append(file_name)
                        print(f"✅ Caricato {file_name}: {len(df)} campioni")
                    else:
                        print(f"⚠️  {file_name} vuoto o senza colonna 'marker', saltato")
                        
                except Exception as e:
                    print(f"❌ Errore caricando {file_name}: {e}")
            else:
                print(f"⚠️  File {file_name} non trovato")
        
        if not df_list:
            raise FileNotFoundError(f"Nessun file dati valido trovato in {self.data_dir}. "
                                   f"Verifica che esistano file data1.csv, data2.csv, etc. con colonna 'marker'")
        
        print(f"File caricati con successo: {successful_loads}")
        
        # Combina tutti i dataframe
        df_combined = pd.concat(df_list, ignore_index=True)
        
        # Separa features e target
        X = df_combined.drop(columns=["marker"])
        y = (df_combined["marker"] != "Natural").astype(int)
        
        # Statistiche dataset
        attack_samples = y.sum()
        natural_samples = (y == 0).sum()
        attack_ratio = y.mean()
        
        print(f"Dataset combinato: {len(df_combined)} campioni")
        print(f"Features originali: {X.shape[1]}")
        print(f"Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
        
        # Verifica che ci siano campioni di entrambe le classi
        if attack_samples == 0 or natural_samples == 0:
            raise ValueError(f"Dataset sbilanciato estremo: {attack_samples} attacchi, {natural_samples} naturali. "
                           f"Necessari campioni di entrambe le classi per ottimizzazione.")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """
        Applica preprocessing identico al sistema federato aggiornato.
        """
        print("\n=== PREPROCESSING DATI ===")

        # Split train/validation (80/20)
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        print(f"Training set: {len(X_train_raw)} campioni")
        print(f"Validation set: {len(X_val_raw)} campioni")

        # Pulizia inf/NaN
        def clean(X):
            X_array = X.values.copy() if hasattr(X, 'values') else X.copy()
            X_array = np.where(np.isinf(X_array), np.nan, X_array)
            return np.array(X_array, dtype=float)

        X_train_cleaned = clean(X_train_raw)
        X_val_cleaned = clean(X_val_raw)

        # Clipping outlier per quantili calcolati sul train e applicati anche al validation
        q1 = np.nanpercentile(X_train_cleaned, 25, axis=0)
        q3 = np.nanpercentile(X_train_cleaned, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        X_train_clipped = np.clip(X_train_cleaned, lower, upper)
        X_val_clipped = np.clip(X_val_cleaned, lower, upper)

        # Imputazione mediana
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_val_imputed = imputer.transform(X_val_clipped)

        # Rimozione feature quasi-costanti (varianza < 1e-8 sul train)
        X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
        X_val_reduced = X_val_imputed[:, keep_mask]

        # Normalizzazione
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reduced)
        X_val_scaled = scaler.transform(X_val_reduced)

        # PCA fissa
        X_train_final = self._apply_pca(X_train_scaled)
        X_val_final = self._apply_pca(X_val_scaled)
        if X_train_final.shape[1] != PCA_COMPONENTS:
            raise RuntimeError(f"PCA output inconsistente: {X_train_final.shape[1]} vs {PCA_COMPONENTS}")

        return X_train_final, X_val_final, y_train, y_val
    
    def _apply_pca(self, X_preprocessed):
        n_components = min(PCA_COMPONENTS, X_preprocessed.shape[1], len(X_preprocessed))
        try:
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
            X_pca = pca.fit_transform(X_preprocessed)
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError("PCA output contiene NaN/inf")
            return X_pca
        except Exception as e:
            # Fallback: usa prime n_components colonne
            return X_preprocessed[:, :n_components]
    
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
        self.current_trial_count += 1
        
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
            
            # Log progresso ogni 10 trial
            if self.current_trial_count % 10 == 0:
                print(f"Trial {self.current_trial_count}: Score = {score:.4f} (F1={f1:.4f}, BalAcc={balanced_acc:.4f})")
            
            return score
            
        except Exception as e:
            if self.current_trial_count % 10 == 0:
                print(f"Trial {self.current_trial_count} fallito: {e}")
            return 0.0
    
    def optimize(self, n_trials):
        """
        Esegue l'ottimizzazione Optuna con numero specificato di trial.
        
        Args:
            n_trials: Numero di trial da eseguire
        
        Returns:
            Migliori parametri trovati
        """
        
        # Reset contatore
        self.current_trial_count = 0
        
        # Carica e preprocessa dati
        X, y = self.load_subset_data()
        self.X_train, self.X_val, self.y_train, self.y_val = self.preprocess_data(X, y)
        
        print(f"\n📊 Dataset stats: {len(self.X_train)} campioni training, {self.X_val.shape[1]} feature")
        print(f"Distribuzione training: {self.y_train.mean()*100:.1f}% attacchi")
        
        # Crea studio Optuna
        study_name = f'smartgrid_hyperparameter_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                n_startup_trials=max(10, n_trials // 10),  # Warm-up trials
                n_ei_candidates=24,  # Candidati per Expected Improvement
                multivariate=True   # Considera correlazioni tra parametri
            )
        )
        
        print(f"\n🚀 INIZIO OTTIMIZZAZIONE")
        print(f"Studio: {study_name}")
        print(f"Configurazione TPE: startup_trials={max(10, n_trials // 10)}, multivariate=True")
        
        # Esegui ottimizzazione con progress tracking
        try:
            study.optimize(
                self.objective, 
                n_trials=n_trials, 
                show_progress_bar=True,
                timeout=None
            )
        except KeyboardInterrupt:
            print(f"\n⚠️  Ottimizzazione interrotta dall'utente dopo {len(study.trials)} trial")
        except Exception as e:
            print(f"\n❌ Errore durante ottimizzazione: {e}")
            raise
        
        # Risultati
        if len(study.trials) == 0:
            raise RuntimeError("Nessun trial completato con successo")
        
        self.best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n=== OTTIMIZZAZIONE COMPLETATA ===")
        print(f"Trial completati: {len(study.trials)}")
        print(f"Trial riusciti: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Trial falliti: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"Migliore score: {best_score:.4f}")
        print(f"Migliori parametri:")
        
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Analisi convergenza
        scores = [t.value for t in study.trials if t.value is not None]
        if len(scores) > 10:
            recent_scores = scores[-10:]
            early_scores = scores[:10]
            improvement = np.mean(recent_scores) - np.mean(early_scores)
            print(f"Miglioramento negli ultimi 10 trial: {improvement:+.4f}")
        
        # Salva risultati
        self._save_results(study, n_trials)
        
        return self.best_params
    
    def _save_results(self, study, n_trials_requested):
        """Salva i risultati in file JSON strutturato."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crea directory se non esiste
        results_dir = "hyperparameter_optimization/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Salva risultati completi
        results = {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'timestamp': timestamp,
            'n_trials_requested': n_trials_requested,
            'n_trials_completed': len(study.trials),
            'n_trials_successful': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_trials_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'pca_components': PCA_COMPONENTS,
            'random_state': self.random_state,
            'study_name': study.study_name,
            'optimization_info': {
                'algorithm': 'TPE (Tree-structured Parzen Estimator)',
                'objective': 'maximize_(f1_score + balanced_accuracy) / 2',
                'preprocessing': 'identical_to_federated_system'
            }
        }
        
        results_file = os.path.join(results_dir, f"optuna_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Risultati salvati in: {results_file}")

def main():
    """
    Funzione principale autonoma per eseguire l'ottimizzazione iperparametri.
    Richiede il numero di trial come argomento da riga di comando.
    """
    # Verifica argomenti
    if len(sys.argv) != 2:
        print("❌ ERRORE: Numero di argomenti non corretto")
        print("")
        print("UTILIZZO:")
        print("  python scripts/optuna_optimizer.py <numero_trials>")
        print("")
        sys.exit(1)
    
    # Parse numero trial
    try:
        n_trials = int(sys.argv[1])
        if n_trials <= 0:
            raise ValueError("Il numero di trial deve essere positivo")
    except ValueError as e:
        print(f"❌ ERRORE: Numero trial non valido")
        print(f"Dettagli: {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("🎯 OTTIMIZZAZIONE IPERPARAMETRI SMARTGRID")
    print("=" * 80)
    print(f"Trial Optuna: {n_trials}")
    print(f"Directory dati: data/SmartGrid")
    print(f"Random seed: 42")
    print("")
    print("🎛️  Parametri che verranno ottimizzati:")
    print("   • Architettura del modello (neuroni per layer)")
    print("   • Funzioni di attivazione (relu, leaky_relu, selu)")
    print("   • Ottimizzatori (adam, adamw)")
    print("   • Parametri di regolarizzazione (dropout, L2)")
    print("   • Configurazione training (learning rate, batch size, epochs)")
    print("")
    print("Obiettivo: Massimizzazione (F1-Score + Balanced Accuracy) / 2")
    print("Algoritmo: TPE (Tree-structured Parzen Estimator)")
    print("Preprocessing: Identico al sistema federato esistente")
    print("")
    
    try:
        # Crea optimizer
        optimizer = SmartGridOptunaOptimizer("data/SmartGrid", random_state=42)
        
        # Verifica directory dati
        if not optimizer.verify_data_directory():
            sys.exit(1)
        
        # Esegui ottimizzazione
        best_params = optimizer.optimize(n_trials)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Ottimizzazione interrotta dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRORE durante ottimizzazione: {e}")
        print("")
        print("🔍 Informazioni per debug:")
        import traceback
        traceback.print_exc()
        print("")
        print("💡 Possibili soluzioni:")
        print("   • Verifica che i file CSV contengano la colonna 'marker'")
        print("   • Controlla che ci siano campioni di entrambe le classi")
        print("   • Riduci il numero di trial se hai problemi di memoria")
        print("   • Verifica le dipendenze: pip install optuna tensorflow scikit-learn")
        sys.exit(1)

if __name__ == "__main__":
    main()