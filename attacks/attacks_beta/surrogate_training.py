"""
attacks/surrogate_training.py

Modulo per l'addestramento del modello surrogato Random Forest.

STRATEGIA:
1. Usa dati pubblici SmartGrid (client 2-12, NO 1, 13 che sono test)
2. Addestra Random Forest "semplificato" (meno alberi del target federato)
3. Il surrogato MIMICA il comportamento del modello target federato
4. Gli adversarial examples generati sul surrogato vengono trasferiti al target

DIFFERENZE SURROGATO vs TARGET FEDERATO:
- Target: 100 alberi aggregati da 13 client
- Surrogato: 50 alberi addestrati su dati pubblici
- Stesso preprocessing per garantire compatibilità

UTILIZZO:
    from attacks.surrogate_training import train_surrogate_model
    
    surrogate, info = train_surrogate_model(
        client_ids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        n_estimators=50
    )

AUTORE: Carmine Cataldo
DATA: 2025-01-21
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from datetime import datetime

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def set_reproducibility_seeds(seed=42):
    """Imposta semi per riproducibilità."""
    import random
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_surrogate_training_data(client_ids, data_dir=None):
    """
    Carica i dati dai client specificati per addestrare il modello surrogato.
    
    IMPORTANTE: USA SOLO CLIENT PUBBLICI (NO client 1, 13 che sono test)
    
    Args:
        client_ids: Lista di ID client da usare (es. [2, 3, 4, 5, ..., 12])
        data_dir: Directory contenente i file data{id}.csv (default: auto-detect)
        
    Returns:
        X_train: Dati di training RAW (numpy array)
        y_train: Etichette di training (numpy array)
        info: Dictionary con informazioni sul dataset
    """
    print(f"\n[Surrogate] === CARICAMENTO DATI TRAINING SURROGATO ===")
    
    # Auto-detect data directory
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'data', 'SmartGrid')
    
    df_list = []
    
    for client_id in client_ids:
        file_path = os.path.join(data_dir, f"data{client_id}.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"[Surrogate] ✓ Caricato data{client_id}.csv: {len(df)} campioni")
            except Exception as e:
                print(f"[Surrogate] ⚠️ Errore caricamento data{client_id}.csv: {e}")
        else:
            print(f"[Surrogate] ⚠️ File non trovato: {file_path}")
    
    if not df_list:
        raise FileNotFoundError(f"❌ Nessun file trovato per client {client_ids}")
    
    # Combina i dataframe
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Prepara X e y
    X = df_combined.drop(columns=["marker"]).values
    y = (df_combined["marker"] != "Natural").astype(int).values  # 0=Natural, 1=Attack
    
    # Statistiche
    info = {
        'total_samples': len(df_combined),
        'attack_samples': int(y.sum()),
        'natural_samples': int((y == 0).sum()),
        'attack_ratio': float(y.mean()),
        'client_ids': client_ids,
        'original_features': X.shape[1]
    }
    
    print(f"[Surrogate] Dataset training surrogato:")
    print(f"  - Campioni totali: {info['total_samples']}")
    print(f"  - Attacchi: {info['attack_samples']} ({info['attack_ratio']*100:.1f}%)")
    print(f"  - Naturali: {info['natural_samples']}")
    print(f"  - Feature originali: {info['original_features']}")
    
    return X, y, info


def apply_surrogate_preprocessing(X_train, X_val=None):
    """
    Applica preprocessing IDENTICO a quello del modello target federato.
    
    CONFIGURAZIONE (da clientRF.py):
    - Pulizia inf/NaN: True
    - Imputazione mediana: True
    - Scaling standard: True
    - PCA: False
    
    Args:
        X_train: Dati di training
        X_val: Dati di validation (opzionale)
        
    Returns:
        X_train_preprocessed: Dati training preprocessati
        X_val_preprocessed: Dati validation preprocessati (se X_val fornito)
        preprocessing_objects: Dictionary con oggetti preprocessing
    """
    print(f"\n[Surrogate] === PREPROCESSING SURROGATO ===")
    
    # STEP 1: Pulizia inf/NaN
    X_train_clean = np.where(np.isinf(X_train), np.nan, X_train)
    print(f"[Surrogate] ✓ Pulizia inf/NaN")
    
    # STEP 2: Imputazione mediana
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    print(f"[Surrogate] ✓ Imputazione mediana")
    
    # STEP 3: Scaling standard
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    print(f"[Surrogate] ✓ Scaling standard")
    
    # Applica a validation se fornito
    if X_val is not None:
        X_val_clean = np.where(np.isinf(X_val), np.nan, X_val)
        X_val_imputed = imputer.transform(X_val_clean)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        print(f"[Surrogate] ✅ Preprocessing completato:")
        print(f"  - Train: {X_train.shape} → {X_train_scaled.shape}")
        print(f"  - Val: {X_val.shape} → {X_val_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, {'imputer': imputer, 'scaler': scaler}
    else:
        print(f"[Surrogate] ✅ Preprocessing completato:")
        print(f"  - Train: {X_train.shape} → {X_train_scaled.shape}")
        
        return X_train_scaled, None, {'imputer': imputer, 'scaler': scaler}


def train_surrogate_model(
    client_ids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    n_estimators=50,
    max_depth=None,
    random_state=42,
    save_model=True,
    model_dir="attacks/models"
):
    """
    Addestra il modello surrogato Random Forest.
    
    CONFIGURAZIONE SURROGATO:
    - N. alberi: 50 (vs 100 del target federato)
    - Criterio: entropy (identico al target)
    - Max features: sqrt (identico al target)
    - Class weight: balanced (identico al target)
    - Dati: Client 2-12 (NO 1, 13 che sono test)
    
    Args:
        client_ids: Lista di client da usare per training (default: 2-12)
        n_estimators: Numero alberi del surrogato (default: 50)
        max_depth: Profondità massima alberi (default: None)
        random_state: Seed per riproducibilità (default: 42)
        save_model: Se True, salva il modello addestrato (default: True)
        model_dir: Directory dove salvare il modello (default: 'attacks/models')
        
    Returns:
        surrogate_model: Random Forest surrogato addestrato
        training_info: Dictionary con informazioni su training e performance
        
    Example:
        >>> surrogate, info = train_surrogate_model()
        >>> print(f"Accuracy surrogato: {info['test_accuracy']:.4f}")
    """
    print("="*80)
    print("🌳 ADDESTRAMENTO MODELLO SURROGATO RANDOM FOREST")
    print("="*80)
    print(f"Configurazione:")
    print(f"  - N. estimatori: {n_estimators}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Client training: {client_ids}")
    print(f"  - Random state: {random_state}")
    print("="*80)
    
    set_reproducibility_seeds(random_state)
    
    # ========== STEP 1: CARICA DATI ==========
    X_raw, y, data_info = load_surrogate_training_data(client_ids)
    
    # ========== STEP 2: SPLIT TRAIN/VAL ==========
    print(f"\n[Surrogate] === SPLIT TRAIN/VALIDATION ===")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )
    
    print(f"[Surrogate] Train: {len(X_train_raw)} campioni")
    print(f"[Surrogate] Val: {len(X_val_raw)} campioni")
    
    # ========== STEP 3: PREPROCESSING ==========
    X_train_preprocessed, X_val_preprocessed, preproc_objects = apply_surrogate_preprocessing(
        X_train_raw, X_val_raw
    )
    
    # ========== STEP 4: ADDESTRAMENTO SURROGATO ==========
    print(f"\n[Surrogate] === ADDESTRAMENTO RANDOM FOREST SURROGATO ===")
    
    surrogate_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion='entropy',      # Identico al target
        max_features='sqrt',      # Identico al target
        class_weight='balanced',  # Identico al target
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    print(f"[Surrogate] Inizio addestramento su {len(X_train_preprocessed)} campioni...")
    surrogate_model.fit(X_train_preprocessed, y_train)
    print(f"[Surrogate] ✅ Addestramento completato!")
    
    # ========== STEP 5: VALUTAZIONE ==========
    print(f"\n[Surrogate] === VALUTAZIONE MODELLO SURROGATO ===")
    
    # Predizioni training
    train_predictions = surrogate_model.predict(X_train_preprocessed)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    
    # Predizioni validation
    val_predictions = surrogate_model.predict(X_val_preprocessed)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)
    
    # OOB score
    oob_score = surrogate_model.oob_score_ if hasattr(surrogate_model, 'oob_score_') else 0.0
    
    print(f"[Surrogate] Performance Training:")
    print(f"  - Accuracy: {train_accuracy:.4f}")
    print(f"  - F1-Score: {train_f1:.4f}")
    print(f"  - OOB Score: {oob_score:.4f}")
    
    print(f"\n[Surrogate] Performance Validation:")
    print(f"  - Accuracy: {val_accuracy:.4f}")
    print(f"  - F1-Score: {val_f1:.4f}")
    
    print(f"\n[Surrogate] Classification Report (Validation):")
    print(classification_report(y_val, val_predictions, target_names=["Natural", "Attack"]))
    
    # ========== STEP 6: SALVATAGGIO ==========
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"surrogate_rf_{n_estimators}trees_{timestamp}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        # Salva modello + preprocessing
        surrogate_package = {
            'model': surrogate_model,
            'preprocessing': preproc_objects,
            'training_info': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'client_ids': client_ids,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_samples': len(X_train_preprocessed),
                'val_samples': len(X_val_preprocessed)
            }
        }
        
        joblib.dump(surrogate_package, model_path)
        print(f"\n[Surrogate] ✅ Modello surrogato salvato: {model_path}")
    
    # ========== STEP 7: RETURN INFO ==========
    training_info = {
        'surrogate_model': surrogate_model,
        'preprocessing': preproc_objects,
        'train_accuracy': train_accuracy,
        'train_f1': train_f1,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'oob_score': oob_score,
        'n_estimators': n_estimators,
        'train_samples': len(X_train_preprocessed),
        'val_samples': len(X_val_preprocessed),
        'client_ids': client_ids,
        'data_info': data_info,
        'model_path': model_path if save_model else None
    }
    
    print("="*80)
    print("✅ MODELLO SURROGATO RANDOM FOREST COMPLETATO")
    print("="*80)
    
    return surrogate_model, training_info