"""
attacks/utils.py

Funzioni di utilità condivise per gli attacchi adversarial sul modello Random Forest federato.

Questo modulo fornisce:
- Caricamento del modello Random Forest federato salvato
- Caricamento e preprocessing dei dati di test (identico al federato)
- Applicazione vincoli fisici SmartGrid
- Gestione riproducibilità

DIPENDENZE:
- joblib: Per caricamento modello .pkl
- scikit-learn: Per preprocessing (StandardScaler, SimpleImputer)
- numpy, pandas: Per manipolazione dati

UTILIZZO:
    from attacks.utils import load_federated_model, load_test_data_from_clients
    
    model = load_federated_model('models/federated_rf_global_20251121_024044.pkl')
    X_test, y_test, info = load_test_data_from_clients([1, 13])
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys


def set_reproducibility_seeds(seed=42):
    """
    Imposta tutti i semi per garantire riproducibilità degli attacchi.
    
    Args:
        seed: Valore del seed (default: 42)
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_federated_model(model_path):
    """
    Carica il modello Random Forest federato salvato.
    
    Args:
        model_path: Path al file .pkl del modello (es. 'models/federated_rf_global_20251121_024044.pkl')
        
    Returns:
        RandomForestClassifier: Modello caricato
        
    Raises:
        FileNotFoundError: Se il file del modello non esiste
        Exception: Se il caricamento fallisce
        
    Example:
        >>> model = load_federated_model('models/federated_rf_global_20251121_024044.pkl')
        >>> print(f"Modello con {len(model.estimators_)} alberi caricato")
    """
    print(f"[Attack Utils] === CARICAMENTO MODELLO FEDERATO ===")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modello non trovato: {model_path}")
    
    try:
        # Carica il modello con joblib (formato standard per scikit-learn)
        model = joblib.load(model_path)
        
        print(f"[Attack Utils] ✅ Modello caricato: {model_path}")
        print(f"[Attack Utils]    Tipo: {type(model).__name__}")
        print(f"[Attack Utils]    N. alberi: {len(model.estimators_)}")
        print(f"[Attack Utils]    N. feature: {model.n_features_in_}")
        print(f"[Attack Utils]    Classi: {model.classes_}")
        
        return model
        
    except Exception as e:
        print(f"[Attack Utils] ❌ Errore caricamento modello: {e}")
        raise


def load_test_data_from_clients(client_ids=[1, 13], data_dir=None):
    """
    Carica i dati dai client specificati (tipicamente test set: client 1 e 13).
    
    NOTA: Questa funzione carica i dati RAW (prima del preprocessing).
    Il preprocessing viene applicato separatamente con apply_preprocessing_pipeline().
    
    Args:
        client_ids: Lista di ID client da caricare (default: [1, 13])
        data_dir: Directory contenente i file data{id}.csv (default: auto-detect)
        
    Returns:
        X_test: Dati di test RAW (numpy array)
        y_test: Etichette di test (numpy array, 0=Natural, 1=Attack)
        info_dict: Dizionario con informazioni sul dataset
        
    Example:
        >>> X_raw, y, info = load_test_data_from_clients([1, 13])
        >>> print(f"Caricati {info['total_samples']} campioni")
        >>> print(f"Distribuzione: {info['attack_ratio']*100:.1f}% attacchi")
    """
    print(f"[Attack Utils] === CARICAMENTO DATI TEST ===")
    
    # Auto-detect data directory se non specificato
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
                print(f"[Attack Utils] ✅ Caricato data{client_id}.csv: {len(df)} campioni")
            except Exception as e:
                print(f"[Attack Utils] ⚠️ Errore caricamento data{client_id}.csv: {e}")
        else:
            print(f"[Attack Utils] ⚠️ File non trovato: {file_path}")
    
    if not df_list:
        raise FileNotFoundError(f"❌ Nessun file di test trovato per client {client_ids}")
    
    # Combina i dataframe
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Prepara X e y
    X = df_combined.drop(columns=["marker"]).values  # Converti a numpy array
    y = (df_combined["marker"] != "Natural").astype(int).values  # 0=Natural, 1=Attack
    
    # Statistiche dataset
    info = {
        'total_samples': len(df_combined),
        'attack_samples': int(y.sum()),
        'natural_samples': int((y == 0).sum()),
        'attack_ratio': float(y.mean()),
        'client_ids': client_ids,
        'original_features': X.shape[1]
    }
    
    print(f"[Attack Utils] Dataset test combinato:")
    print(f"  - Campioni totali: {info['total_samples']}")
    print(f"  - Attacchi: {info['attack_samples']} ({info['attack_ratio']*100:.1f}%)")
    print(f"  - Naturali: {info['natural_samples']}")
    print(f"  - Feature originali: {info['original_features']}")
    
    return X, y, info


def apply_preprocessing_pipeline(X, fit_on_data=None):
    """
    Applica la stessa pipeline di preprocessing del federato.
    DEVE essere IDENTICA a quella usata in clientRF.py/serverRF.py.
    
    CONFIGURAZIONE (da clientRF.py):
    - ENABLE_CLEAN_INF_NAN = True
    - ENABLE_IMPUTATION = True (imputazione mediana)
    - ENABLE_SCALING = True (StandardScaler)
    - ENABLE_PCA = False
    
    Args:
        X: Dati da preprocessare (numpy array)
        fit_on_data: Se None, fit+transform su X. Altrimenti, solo transform usando fit_on_data.
        
    Returns:
        X_preprocessed: Dati preprocessati (numpy array)
        preprocessing_objects: Dictionary con oggetti preprocessing (imputer, scaler)
        
    Example:
        >>> X_test_preprocessed, _ = apply_preprocessing_pipeline(X_test_raw)
        >>> print(f"Shape dopo preprocessing: {X_test_preprocessed.shape}")
    """
    print(f"[Attack Utils] === APPLICAZIONE PREPROCESSING ===")
    
    # ========== CONFIGURAZIONE (IDENTICA AL FEDERATO) ==========
    ENABLE_CLEAN_INF_NAN = True
    ENABLE_IMPUTATION = True
    ENABLE_SCALING = True
    
    print(f"[Attack Utils] Pipeline configurata:")
    print(f"  - Pulizia inf/NaN: {'✓' if ENABLE_CLEAN_INF_NAN else '✗'}")
    print(f"  - Imputazione mediana: {'✓' if ENABLE_IMPUTATION else '✗'}")
    print(f"  - Scaling standard: {'✓' if ENABLE_SCALING else '✗'}")
    
    # STEP 1: Pulizia inf/NaN
    if ENABLE_CLEAN_INF_NAN:
        X_clean = np.where(np.isinf(X), np.nan, X)
        print(f"[Attack Utils] ✓ Pulizia inf/NaN applicata")
    else:
        X_clean = X
    
    # STEP 2: Imputazione mediana
    if ENABLE_IMPUTATION:
        if fit_on_data is not None:
            # Fit su altri dati, transform su X
            fit_on_data_clean = np.where(np.isinf(fit_on_data), np.nan, fit_on_data)
            imputer = SimpleImputer(strategy='median')
            imputer.fit(fit_on_data_clean)
            X_imputed = imputer.transform(X_clean)
        else:
            # Fit+transform su X
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X_clean)
        print(f"[Attack Utils] ✓ Imputazione mediana applicata")
    else:
        X_imputed = X_clean
        imputer = None
    
    # STEP 3: Scaling standard
    if ENABLE_SCALING:
        if fit_on_data is not None:
            # Fit su altri dati, transform su X
            fit_on_data_imputed = SimpleImputer(strategy='median').fit_transform(
                np.where(np.isinf(fit_on_data), np.nan, fit_on_data)
            )
            scaler = StandardScaler()
            scaler.fit(fit_on_data_imputed)
            X_scaled = scaler.transform(X_imputed)
        else:
            # Fit+transform su X
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
        print(f"[Attack Utils] ✓ Scaling standard applicato")
    else:
        X_scaled = X_imputed
        scaler = None
    
    print(f"[Attack Utils] ✅ Preprocessing completato:")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Output shape: {X_scaled.shape}")
    
    return X_scaled, {'imputer': imputer, 'scaler': scaler}


def get_smartgrid_physical_constraints(X_sample):
    """
    Definisce i vincoli fisici per SmartGrid basati sui dati osservati.
    
    SPIEGAZIONE:
    Nel contesto SmartGrid, le feature rappresentano misurazioni fisiche
    (es. voltaggio, corrente, frequenza). Per garantire che gli esempi
    adversarial siano REALISTICI, devono rispettare i range fisici plausibili.
    
    Strategia:
    - Min: Percentile 0.1% (per evitare outlier estremi)
    - Max: Percentile 99.9%
    
    Args:
        X_sample: Campione di dati SmartGrid preprocessati (per calcolare range)
        
    Returns:
        Dictionary con vincoli fisici:
            - feature_min: Array con valori minimi per ogni feature
            - feature_max: Array con valori massimi per ogni feature
            - description: Descrizione del metodo usato
            
    Example:
        >>> constraints = get_smartgrid_physical_constraints(X_test)
        >>> print(f"Range feature 0: [{constraints['feature_min'][0]:.3f}, {constraints['feature_max'][0]:.3f}]")
    """
    print(f"[Attack Utils] === CALCOLO VINCOLI FISICI SMARTGRID ===")
    
    # Calcola range basato sui dati osservati (percentili robusti)
    feature_min = np.percentile(X_sample, 0.1, axis=0)
    feature_max = np.percentile(X_sample, 99.9, axis=0)
    
    print(f"[Attack Utils] Vincoli calcolati su {len(X_sample)} campioni")
    print(f"[Attack Utils] Range globale: [{feature_min.min():.3f}, {feature_max.max():.3f}]")
    
    return {
        'feature_min': feature_min,
        'feature_max': feature_max,
        'description': 'Vincoli basati su percentili 0.1-99.9 del dataset'
    }


def apply_physical_constraints(X_adv, X_original, constraints, max_perturbation_linf=None):
    """
    Applica vincoli fisici agli esempi adversarial per garantire realismo.
    
    SPIEGAZIONE:
    Gli esempi adversarial devono essere:
    1. FISICAMENTE PLAUSIBILI: Rispettare i range delle misurazioni SmartGrid
    2. MINIMAMENTE PERTURBATI: La perturbazione non deve essere troppo grande
    
    Vincoli applicati (in ordine):
    1. Range fisico: Ogni feature deve essere tra feature_min e feature_max
    2. Perturbazione L-inf: |x_adv - x_orig| ≤ max_perturbation_linf
    3. Ri-applicazione range fisico (dopo clip perturbazione)
    
    Args:
        X_adv: Esempi adversarial generati (numpy array)
        X_original: Esempi originali (numpy array)
        constraints: Dictionary con feature_min/feature_max
        max_perturbation_linf: Perturbazione massima L-inf consentita (opzionale)
        
    Returns:
        X_adv_constrained: Esempi adversarial con vincoli applicati
        
    Example:
        >>> X_adv_safe = apply_physical_constraints(X_adv, X_orig, constraints, max_perturbation_linf=0.1)
    """
    print(f"[Attack Utils] === APPLICAZIONE VINCOLI FISICI ===")
    
    X_constrained = X_adv.copy()
    
    # Vincolo 1: Range fisico delle feature
    X_constrained = np.clip(
        X_constrained,
        constraints['feature_min'],
        constraints['feature_max']
    )
    print(f"[Attack Utils] ✓ Vincolo range fisico applicato")
    
    # Vincolo 2: Perturbazione massima L-inf (se specificato)
    if max_perturbation_linf is not None:
        perturbation = X_constrained - X_original
        perturbation_clipped = np.clip(perturbation, -max_perturbation_linf, max_perturbation_linf)
        X_constrained = X_original + perturbation_clipped
        print(f"[Attack Utils] ✓ Vincolo perturbazione L-inf ≤ {max_perturbation_linf} applicato")
    
    # Vincolo 3: Ri-applica range fisico (dopo clip perturbazione)
    X_constrained = np.clip(
        X_constrained,
        constraints['feature_min'],
        constraints['feature_max']
    )
    
    # Calcola statistiche finali
    final_perturbation = X_constrained - X_original
    l2_norm_mean = np.mean(np.linalg.norm(final_perturbation, axis=1))
    linf_norm_mean = np.mean(np.max(np.abs(final_perturbation), axis=1))
    
    print(f"[Attack Utils] ✅ Vincoli applicati:")
    print(f"  - Perturbazione L2 media: {l2_norm_mean:.6f}")
    print(f"  - Perturbazione L-inf media: {linf_norm_mean:.6f}")
    
    return X_constrained


def select_attack_samples(X, y, target_class=1):
    """
    Seleziona solo i campioni di una specifica classe da perturbare.
    
    SPIEGAZIONE:
    Per l'attacco di EVASION, vogliamo:
    - Prendere campioni di ATTACCO (classe 1)
    - Perturbarli per farli classificare come NATURAL (classe 0)
    
    Questo simula un attaccante che cerca di evadere il sistema di intrusion detection.
    
    Args:
        X: Tutti i dati (numpy array)
        y: Tutte le etichette (numpy array)
        target_class: Classe dei campioni da selezionare (default: 1 = Attack)
        
    Returns:
        X_selected: Campioni della classe target
        y_selected: Etichette corrispondenti (tutte = target_class)
        indices: Indici originali dei campioni selezionati
        
    Example:
        >>> X_attacks, y_attacks, indices = select_attack_samples(X_test, y_test, target_class=1)
        >>> print(f"Selezionati {len(X_attacks)} campioni di attacco")
    """
    print(f"[Attack Utils] === SELEZIONE CAMPIONI PER ATTACCO ===")
    
    # Maschera per selezionare solo la classe target
    mask = (y == target_class)
    
    X_selected = X[mask]
    y_selected = y[mask]
    indices = np.where(mask)[0]
    
    print(f"[Attack Utils] Selezionati {len(X_selected)} campioni di classe {target_class}")
    print(f"  - Totale campioni disponibili: {len(X)}")
    print(f"  - Percentuale selezionata: {len(X_selected)/len(X)*100:.1f}%")
    
    return X_selected, y_selected, indices