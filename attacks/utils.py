"""
Utility per attacchi evasion: vincoli fisici, preprocessing, caricamento dati.

Questo modulo fornisce funzioni di supporto per:
- Applicare vincoli fisici specifici del dominio SmartGrid
- Preprocessing identico a quello usato nel training (per compatibilità)
- Caricamento dati di test
- Gestione range feature

Le funzioni in questo modulo garantiscono che gli adversarial examples
generati siano realistici e compatibili con il modello target.
"""

import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============== CONFIGURAZIONE PREPROCESSING (DEVE MATCHARE clientRF.py) ==============
# Questa configurazione DEVE essere identica a quella usata nel training del Random Forest
# per garantire compatibilità degli adversarial examples

ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = False      # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = True                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False                    # PCA per riduzione dimensionalità

RANDOM_SEED = 42


def set_reproducibility_seeds(seed=RANDOM_SEED):
    """
    Imposta i seed per riproducibilità degli attacchi.
    
    Args:
        seed: Seed per random number generators
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def clean_data_for_preprocessing(X):
    """
    Pulizia robusta dei dati: sostituisce inf/-inf con NaN.
    Identica alla funzione usata nel training.
    
    Args:
        X: Dati da pulire (numpy array o pandas DataFrame)
        
    Returns:
        X_cleaned: Dati puliti (numpy array)
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    
    return X_array


def apply_preprocessing_pipeline(X, fit_on_data=None):
    """
    Applica la pipeline di preprocessing IDENTICA a quella del Random Forest federato.
    
    IMPORTANTE: Questa funzione deve replicare esattamente il preprocessing
    usato nel training per garantire che gli adversarial examples siano
    compatibili con il modello.
    
    Pipeline:
    1. Pulizia inf/NaN
    2. Imputazione mediana
    3. Scaling standard (mean=0, std=1)
    
    Args:
        X: Dati da preprocessare (numpy array o pandas DataFrame)
        fit_on_data: Se fornito, usa questi dati per fit di imputer/scaler
                     (per train set). Se None, assume preprocessing già fittato.
        
    Returns:
        X_preprocessed: Dati preprocessati
        preprocessing_objects: Dict con imputer e scaler (se fit_on_data fornito)
    """
    print(f"[Preprocessing] Inizio preprocessing pipeline...")
    print(f"  - Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"  - Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"  - Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    
    X_processed = X.copy()
    preprocessing_objects = {}
    
    # STEP 1: Pulizia inf/NaN
    if ENABLE_CLEAN_INF_NAN:
        X_processed = clean_data_for_preprocessing(X_processed)
        print(f"[Preprocessing] Pulizia inf/NaN completata")
    
    # STEP 2: Imputazione mediana
    if ENABLE_IMPUTATION:
        if fit_on_data is not None:
            # Fit imputer sul training set
            imputer = SimpleImputer(strategy='median')
            X_processed = imputer.fit_transform(X_processed)
            preprocessing_objects['imputer'] = imputer
            print(f"[Preprocessing] Imputer fittato e applicato")
        else:
            # Assume imputer già fittato (non dovrebbe accadere in questo caso)
            print(f"[Preprocessing] ⚠️ Imputazione richiesta ma nessun fit_on_data fornito")
    
    # STEP 3: Scaling standard
    if ENABLE_SCALING:
        if fit_on_data is not None:
            # Fit scaler sul training set
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            preprocessing_objects['scaler'] = scaler
            print(f"[Preprocessing] Scaler fittato e applicato")
        else:
            print(f"[Preprocessing] ⚠️ Scaling richiesto ma nessun fit_on_data fornito")
    
    print(f"[Preprocessing] Pipeline completata: shape={X_processed.shape}")
    
    return X_processed, preprocessing_objects


def get_smartgrid_physical_constraints(X_data):
    """
    Definisce i vincoli fisici per le feature del dataset SmartGrid.
    
    IMPORTANTE: Questi vincoli garantiscono che le perturbazioni siano realistiche
    e rispettino i limiti fisici dei sensori nella Smart Grid.
    
    I vincoli sono derivati dai range osservati nel dataset + margine di sicurezza.
    
    Args:
        X_data: Dati del dataset (per calcolare range)
        
    Returns:
        constraints: Dict con {feature_idx: (min_val, max_val)}
    """
    print(f"[Vincoli Fisici] Calcolo vincoli fisici SmartGrid...")
    
    constraints = {}
    
    # Calcola range per ogni feature basandosi sui dati osservati
    # Usiamo percentili per evitare outlier estremi
    for col_idx in range(X_data.shape[1]):
        col_data = X_data[:, col_idx]
        
        # Rimuovi NaN per calcolo percentili
        col_clean = col_data[~np.isnan(col_data)]
        
        if len(col_clean) > 0:
            # Usa percentili 0.1% e 99.9% per margine di sicurezza
            min_val = np.percentile(col_clean, 0.1)
            max_val = np.percentile(col_clean, 99.9)
            
            # Aggiungi margine del 5% per flessibilità
            range_val = max_val - min_val
            margin = range_val * 0.05
            
            constraints[col_idx] = (min_val - margin, max_val + margin)
        else:
            # Fallback se colonna vuota
            constraints[col_idx] = (-1e10, 1e10)
    
    print(f"[Vincoli Fisici] Vincoli calcolati per {len(constraints)} feature")
    print(f"[Vincoli Fisici] Esempio feature 0: range={constraints.get(0, 'N/A')}")
    
    return constraints


def apply_physical_constraints(X_adv, X_original, constraints, max_perturbation_linf=None):
    """
    Applica vincoli fisici agli adversarial examples generati.
    
    Questa funzione garantisce che:
    1. Le perturbazioni non superino un limite massimo (L-inf)
    2. I valori finali rispettino i range fisici del dominio
    
    Args:
        X_adv: Adversarial examples generati
        X_original: Dati originali (riferimento)
        constraints: Vincoli fisici per feature (da get_smartgrid_physical_constraints)
        max_perturbation_linf: Epsilon massimo per L-inf (opzionale)
        
    Returns:
        X_adv_constrained: Adversarial examples con vincoli applicati
    """
    print(f"[Vincoli Fisici] Applicazione vincoli fisici...")
    
    X_adv_constrained = X_adv.copy()
    
    # 1. Limita perturbazione massima (L-inf) se specificato
    if max_perturbation_linf is not None:
        perturbation = X_adv - X_original
        perturbation = np.clip(perturbation, -max_perturbation_linf, max_perturbation_linf)
        X_adv_constrained = X_original + perturbation
        print(f"[Vincoli Fisici] Perturbazione limitata a L-inf={max_perturbation_linf}")
    
    # 2. Clippa ogni feature entro i suoi range fisici
    for feature_idx, (min_val, max_val) in constraints.items():
        X_adv_constrained[:, feature_idx] = np.clip(
            X_adv_constrained[:, feature_idx], 
            min_val, 
            max_val
        )
    
    # 3. Verifica finale: nessun inf/NaN
    X_adv_constrained = np.nan_to_num(
        X_adv_constrained, 
        nan=0.0, 
        posinf=1e10, 
        neginf=-1e10
    )
    
    # Calcola statistiche perturbazione finale
    final_perturbation = X_adv_constrained - X_original
    l_inf_final = np.abs(final_perturbation).max()
    l2_final = np.sqrt((final_perturbation ** 2).sum(axis=1)).mean()
    
    print(f"[Vincoli Fisici] Vincoli applicati con successo")
    print(f"[Vincoli Fisici] Perturbazione finale: L-inf={l_inf_final:.6f}, L2={l2_final:.6f}")
    
    return X_adv_constrained


def load_test_data_from_clients(client_ids, data_dir="data/SmartGrid"):
    """
    Carica dati di test dai client specificati (client 1 e 13 per RF federato).
    
    IMPORTANTE: Questa funzione carica i dati RAW (senza preprocessing)
    perché il preprocessing verrà applicato successivamente in modo
    identico a quello del training.
    
    Args:
        client_ids: Lista di ID client da cui caricare dati
        data_dir: Directory contenente i file data{id}.csv
        
    Returns:
        X_test: Feature del test set (senza preprocessing)
        y_test: Etichette del test set
        dataset_info: Informazioni sul dataset
    """
    print(f"[Caricamento Dati] Caricamento test set da client {client_ids}...")
    
    # Trova il percorso assoluto della directory dati
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, data_dir)
    
    df_list = []
    
    for client_id in client_ids:
        file_path = os.path.join(data_path, f"data{client_id}.csv")
        
        if not os.path.exists(file_path):
            print(f"[Caricamento Dati] ⚠️ File non trovato: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"[Caricamento Dati] Caricato data{client_id}.csv: {len(df)} campioni")
        except Exception as e:
            print(f"[Caricamento Dati] ⚠️ Errore caricamento data{client_id}.csv: {e}")
            continue
    
    if not df_list:
        raise FileNotFoundError(
            f"Nessun file di dati trovato per client {client_ids} in {data_path}"
        )
    
    # Combina i dataframe
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Separa features e label
    X = df_combined.drop(columns=["marker"])
    y = (df_combined["marker"] != "Natural").astype(int)
    
    # Statistiche dataset
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    dataset_info = {
        'total_samples': len(df_combined),
        'attack_samples': int(attack_samples),
        'natural_samples': int(natural_samples),
        'attack_ratio': float(attack_ratio),
        'n_features': X.shape[1],
        'client_ids': client_ids
    }
    
    print(f"[Caricamento Dati] Dataset combinato:")
    print(f"  - Totale campioni: {dataset_info['total_samples']}")
    print(f"  - Attacchi: {dataset_info['attack_samples']} ({attack_ratio*100:.1f}%)")
    print(f"  - Naturali: {dataset_info['natural_samples']}")
    print(f"  - Feature: {dataset_info['n_features']}")
    
    return X.values, y.values, dataset_info