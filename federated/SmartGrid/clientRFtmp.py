import flwr as fl
import pandas as pd
import numpy as np
import sys
import os
import warnings
import pickle
import joblib
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from scipy import stats
warnings.filterwarnings('ignore')

# CONFIGURAZIONE SEMI PER RIPRODUCIBILITÀ
RANDOM_SEED = 42

# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==============
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = False       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = False                 # StandardScaler (mean=0, std=1) - ABILITATO
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti - ABILITATO
ENABLE_PCA = False  # PCA per riduzione dimensionalità
ENABLE_FEATURE_ENGINEERING = True    # NUOVA: Feature engineering per SmartGrid

if ENABLE_PCA:
    ENABLE_IMPUTATION = True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 74  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 42  # Seme specifico per PCA
  
# Quando PCA disabilitata, disabilita rimozione feature quasi-costanti per compatibilità dei modelli
if ENABLE_PCA == False and not ENABLE_FEATURE_ENGINEERING:
    ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False
    PCA_COMPONENTS = None

# CONFIGURAZIONE MODELLO RANDOM FOREST - OTTIMIZZATA
# Basato sui risultati del paper: hyperparameter tuning per ottimizzare performance
RF_N_ESTIMATORS = 100      # AUMENTATO da 65 per maggiore diversità
RF_MAX_DEPTH = 15         # LIMITATO da None per ridurre overfitting
RF_MIN_SAMPLES_SPLIT = 5  # AUMENTATO da 2 per ridurre overfitting
RF_MIN_SAMPLES_LEAF = 2   # AUMENTATO da 1 per ridurre overfitting
RF_MAX_FEATURES = 'sqrt'  # Feature da considerare per ogni split ('sqrt' dal paper)
RF_BOOTSTRAP = True       # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced_subsample'  # CAMBIATO per migliore gestione sbilanciamento locale
RF_CRITERION = 'entropy'  # Criterio di splitting (dal paper: entropy migliore di gini per molti dataset)

# CONFIGURAZIONE ENSEMBLE PER FEDERATED RANDOM FOREST
# Basato sulla metodologia del paper per aggregazione degli alberi
ENSEMBLE_METHOD = 'weighted_voting'  # 'simple_voting' o 'weighted_voting'
TREE_SELECTION_METHOD = 'diversity_weighted'  # NUOVO: accuracy + diversity

def set_reproducibility_seeds():
    """
    Imposta tutti i semi per garantire riproducibilità.
    Da chiamare all'inizio di ogni funzione critica.
    """
    # Seed per NumPy
    np.random.seed(RANDOM_SEED)
    
    # Seed per Python random (usato da scikit-learn)
    import random
    random.seed(RANDOM_SEED)
    
    # Configurazioni per determinismo
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

def create_smartgrid_features(df, client_id):
    """
    Crea feature ingegnerizzate specifiche per il dataset SmartGrid.
    Basato sui patterns identificati nel paper MSU/ORNL per cyber-physical attacks.
    AGGIORNATO: Versione migliorata con logging dettagliato per debug.
    
    Args:
        df: DataFrame originale con feature SmartGrid
        client_id: ID del client per debug
        
    Returns:
        DataFrame con feature aggiuntive per detection attacchi
    """
    if not ENABLE_FEATURE_ENGINEERING:
        return df
        
    print(f"[{client_id}] === FEATURE ENGINEERING SMARTGRID CLIENT ===")
    
    # Copia il dataframe
    df_enhanced = df.copy()
    
    # Gestisci sia il caso con che senza colonna marker
    if 'marker' in df_enhanced.columns:
        original_features = len(df_enhanced.columns) - 1  # -1 per 'marker'
        feature_cols = [col for col in df_enhanced.columns if col != 'marker']
    else:
        original_features = len(df_enhanced.columns)
        feature_cols = list(df_enhanced.columns)
    
    print(f"[{client_id}] 🔍 DEBUG FEATURE ENGINEERING:")
    print(f"[{client_id}]   DataFrame shape iniziale: {df_enhanced.shape}")
    print(f"[{client_id}]   Feature originali: {original_features}")
    print(f"[{client_id}]   Colonne totali: {list(df_enhanced.columns)[:10]}... ({len(df_enhanced.columns)} totali)")
    
    # Seleziona solo colonne numeriche
    numeric_cols = df_enhanced[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print(f"[{client_id}] ⚠️ Nessuna colonna numerica trovata per feature engineering")
        return df_enhanced
    
    print(f"[{client_id}]   Colonne numeriche trovate: {len(numeric_cols)}")
    print(f"[{client_id}]   Prime 10 colonne numeriche: {numeric_cols[:10]}")
    
    # Converti in numpy per efficienza
    X = df_enhanced[numeric_cols].values
    
    print(f"[{client_id}] Processing {len(numeric_cols)} numeric features...")
    
    # Contatori per tracciare feature aggiunte
    features_added = 0
    
    # 1. STATISTICAL FEATURES (papers SmartGrid indicano l'importanza di statistiche aggregate)
    try:
        window_size = min(10, len(numeric_cols))
        stat_features_added = 0
        for i in range(0, len(numeric_cols), window_size):
            end_idx = min(i + window_size, len(numeric_cols))
            window_data = X[:, i:end_idx]
            
            # Statistiche finestra
            df_enhanced[f'window_{i}_mean'] = np.mean(window_data, axis=1)
            df_enhanced[f'window_{i}_std'] = np.std(window_data, axis=1)
            df_enhanced[f'window_{i}_range'] = np.ptp(window_data, axis=1)  # max - min
            df_enhanced[f'window_{i}_skew'] = stats.skew(window_data, axis=1)
            stat_features_added += 4
        
        features_added += stat_features_added
        print(f"[{client_id}] ✅ Aggiunte {stat_features_added} statistical features")
        print(f"[{client_id}]   Shape dopo statistical: {df_enhanced.shape}")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore statistical features: {e}")
    
    # 2. RATIO FEATURES (critiche per power systems)
    try:
        n_ratios = 0
        for i in range(0, min(20, len(numeric_cols))):
            for j in range(i+1, min(i+5, len(numeric_cols))):
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                denominator = df_enhanced[col_j].replace(0, np.nan)
                
                if not denominator.isna().all():
                    df_enhanced[f'ratio_{i}_{j}'] = df_enhanced[col_i] / denominator
                    n_ratios += 1
                
                if n_ratios >= 50:
                    break
            if n_ratios >= 50:
                break
        
        features_added += n_ratios
        print(f"[{client_id}] ✅ Aggiunti {n_ratios} ratio features")
        print(f"[{client_id}]   Shape dopo ratios: {df_enhanced.shape}")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore ratio features: {e}")
    
    # 3. ANOMALY INDICATORS 
    try:
        n_zscore = 0
        for col in numeric_cols[:15]:
            col_mean = df_enhanced[col].mean()
            col_std = df_enhanced[col].std()
            if col_std > 0:
                df_enhanced[f'zscore_{col}'] = np.abs((df_enhanced[col] - col_mean) / col_std)
                n_zscore += 1
        
        features_added += n_zscore
        print(f"[{client_id}] ✅ Aggiunti {n_zscore} anomaly indicators")
        print(f"[{client_id}]   Shape dopo anomaly: {df_enhanced.shape}")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore anomaly features: {e}")
    
    # 4. INTERACTION FEATURES (importanti per cyber-attacks che influenzano multiple variabili)
    try:
        n_interactions = 0
        # Prodotti tra feature correlate (simula interazioni fisiche)
        for i in range(0, min(10, len(numeric_cols))):
            for j in range(i+1, min(i+3, len(numeric_cols))):
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                df_enhanced[f'interact_{i}_{j}'] = df_enhanced[col_i] * df_enhanced[col_j]
                n_interactions += 1
                
                if n_interactions >= 20:  # Limita interazioni
                    break
            if n_interactions >= 20:
                break
        
        features_added += n_interactions
        print(f"[{client_id}] ✅ Aggiunte {n_interactions} interaction features")
        print(f"[{client_id}]   Shape dopo interactions: {df_enhanced.shape}")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore interaction features: {e}")
    
    new_features = len(df_enhanced.columns) - original_features
    if 'marker' in df_enhanced.columns:
        new_features -= 1  # Non contare la colonna marker
        
    print(f"[{client_id}] 🎯 Feature engineering completato:")
    print(f"[{client_id}]   Features originali: {original_features}")
    print(f"[{client_id}]   Features aggiunte: {new_features}")
    print(f"[{client_id}]   Features totali: {original_features + new_features}")
    print(f"[{client_id}]   Shape finale: {df_enhanced.shape}")
    
    return df_enhanced

def fit_clip_outliers_iqr(X, k=5.0):
    """
    Calcola i limiti inferiori e superiori per ogni feature
    usando la regola dei quantili (IQR) sul dataset fornito (tipicamente il training).
    Ritorna due array: lower e upper.
    """
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def transform_clip_outliers_iqr(X, lower, upper):
    """
    Applica il clipping ai dati X usando i limiti forniti.
    """
    return np.clip(X, lower, upper)

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

def clean_data_for_pca(X):
    """
    Pulizia robusta dei dati per prevenire problemi numerici in PCA:
    - Sostituisce inf/-inf con NaN
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X_preprocessed, client_id=None):
    """
    Applica PCA con numero FISSO di componenti.
    """
    print(f"[Client {client_id}] === APPLICAZIONE PCA ===")

    original_features = X_preprocessed.shape[1]
    n_samples = len(X_preprocessed)
    n_components = min(PCA_COMPONENTS, original_features, n_samples)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pca = PCA(n_components=n_components, random_state=PCA_RANDOM_SEED)
            X_pca = pca.fit_transform(X_preprocessed)

            # VERIFICA: Output senza NaN/inf e dimensioni corrette
            if np.any(np.isnan(X_pca)) or np.any(np.isinf(X_pca)):
                raise ValueError(f"❌PCA client {client_id} ha prodotto output con NaN o inf")
            if X_pca.shape[1] != n_components:
                raise ValueError(f"❌ PCA output shape inconsistente: {X_pca.shape[1]} vs {n_components}")
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            print(f"[Client {client_id}] ✅ PCA fissa applicata: {X_pca.shape}")
            print(f"[Client {client_id}] Varianza spiegata: {variance_explained*100:.2f}%")
            return X_pca
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ ERRORE PCA: {e}")
        print(f"[Client {client_id}] Attivazione fallback semplificato...")
        n_fallback = min(n_components, original_features)
        X_fallback = X_preprocessed[:, :n_fallback]
        print(f"[Client {client_id}] ⚠️ Fallback: {X_fallback.shape}")
        return X_fallback

def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico.
    Applica preprocessing completo per gestire valori infiniti e NaN.
    AGGIORNATO: Include feature engineering per SmartGrid.
    """
    # Imposta semi per riproducibilità del preprocessing
    set_reproducibility_seeds()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")
    
    df = pd.read_csv(file_path)
    
    print(f"=== PREPROCESSING FEDERATO RANDOM FOREST ===")
    print(f"Feature engineering: {'ABILITATA' if ENABLE_FEATURE_ENGINEERING else 'DISABILITATA'}")
    print(f"Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")

    # NUOVO: Feature Engineering per SmartGrid
    if ENABLE_FEATURE_ENGINEERING:
        df = create_smartgrid_features(df, client_id)
    
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"[Client {client_id}] Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
    
    # STEP 1: Pulizia inf/NaN 
    print(f"[Client {client_id}] Pulizia valori infiniti e NaN...")
    X_cleaned = clean_data_for_pca(X)
    
    # Converti a numpy e sostituisci inf con valori finiti
    X_array = np.array(X_cleaned, dtype=float)
    
    # Gestisci infiniti: sostituisci con valori estremi ma finiti
    inf_mask = np.isinf(X_array)
    if np.any(inf_mask):
        print(f"[Client {client_id}] Trovati {np.sum(inf_mask)} valori infiniti, li sostituisco...")
        # Sostituisci +inf con il 99.9° percentile della colonna
        # Sostituisci -inf con il 0.1° percentile della colonna
        for col in range(X_array.shape[1]):
            col_data = X_array[:, col]
            finite_mask = np.isfinite(col_data)
            if np.any(finite_mask):
                percentile_99 = np.percentile(col_data[finite_mask], 99.9)
                percentile_01 = np.percentile(col_data[finite_mask], 0.1)
                X_array[np.isposinf(col_data), col] = percentile_99
                X_array[np.isneginf(col_data), col] = percentile_01
            else:
                # Se tutta la colonna è infinita, usa 0
                X_array[:, col] = 0.0

    # Suddivisione train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_array, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"[Client {client_id}] Suddivisione: {len(X_train_raw)} training, {len(X_val_raw)} validation")

    # STEP 2: Clipping outlier per quantili
    if ENABLE_CLIPPING_OUTLIERS:
        lower, upper = fit_clip_outliers_iqr(X_train_raw, k=5.0)
        X_train_clipped = transform_clip_outliers_iqr(X_train_raw, lower, upper)
        X_val_clipped = transform_clip_outliers_iqr(X_val_raw, lower, upper)
    else:
        X_train_clipped = X_train_raw
        X_val_clipped = X_val_raw

    # STEP 3: Imputazione mediana
    print(f"[Client {client_id}] Applicazione imputazione mediana...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clipped)
    X_val_imputed = imputer.transform(X_val_clipped)

    # STEP 4: Rimozione feature quasi-costanti
    if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES:
        X_train_reduced, keep_mask = remove_near_constant_features(X_train_imputed, threshold_var=1e-12, threshold_ratio=0.999)
        X_val_reduced = X_val_imputed[:, keep_mask]
        print(f"[Client {client_id}] Feature dopo rimozione quasi-costanti: {X_train_reduced.shape[1]} (da {X_train_imputed.shape[1]})")
    else:
        X_train_reduced = X_train_imputed
        X_val_reduced = X_val_imputed
        print(f"[Client {client_id}] Rimozione feature quasi-costanti DISABILITATA - mantenute {X_train_reduced.shape[1]} feature")

    # STEP 5: Scaling standard
    if ENABLE_SCALING:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reduced)
        X_val_scaled = scaler.transform(X_val_reduced)
        print(f"[Client {client_id}] Scaling applicato")
    else:
        X_train_scaled = X_train_reduced
        X_val_scaled = X_val_reduced
        print(f"[Client {client_id}] Scaling DISABILITATO")

    # STEP 6: PCA
    if ENABLE_PCA:
        X_train_final = apply_pca(X_train_scaled, client_id=client_id)
        X_val_final = apply_pca(X_val_scaled, client_id=client_id)
        expected_features = PCA_COMPONENTS
        if X_train_final.shape[1] != expected_features:
            raise RuntimeError(f"Client {client_id}: ⚠️ PCA output shape inconsistente: {X_train_final.shape} vs {expected_features}")
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        print(f"[Client {client_id}] PCA DISABILITATA - usando dati preprocessati: {X_train_final.shape}")
    
    # VERIFICA FINALE: nessun valore infinito o NaN
    if np.any(np.isinf(X_train_final)) or np.any(np.isnan(X_train_final)):
        print(f"[Client {client_id}] ❌ ERRORE: Dati finali contengono ancora inf/NaN")
        # Pulizia di emergenza
        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val_final = np.nan_to_num(X_val_final, nan=0.0, posinf=1e10, neginf=-1e10)
        print(f"[Client {client_id}] ⚠️ Pulizia di emergenza applicata")
    
    print(f"[Client {client_id}] ✅ Preprocessing completato: {X_train_final.shape}, {X_val_final.shape}")
        
    # Info dataset
    dataset_info = {
        'client_id': client_id,
        'total_samples': len(df),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio,
        'train_attack_ratio': y_train.mean(),
        'val_attack_ratio': y_val.mean(),
        'original_features': X.shape[1],
        'final_features': X_train_final.shape[1],
        'pca_enabled': ENABLE_PCA,
        'feature_engineering_enabled': ENABLE_FEATURE_ENGINEERING,
        'remove_near_constant_enabled': ENABLE_REMOVE_NEAR_CONSTANT_FEATURES,
        'pca_components_fixed': PCA_COMPONENTS if ENABLE_PCA else None,
        'preprocessing_method': f"enhanced_smartgrid{'_pca' if ENABLE_PCA else ''}",
        'compatibility_guaranteed': True
    }
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO ===")
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_random_forest_model():
    """
    Crea il modello Random Forest per SmartGrid.
    Implementa la configurazione ottimizzata basata su ricerca recente.
    
    Returns:
        Modello RandomForestClassifier configurato secondo ricerca ottimizzata
    """
    # Imposta semi per riproducibilità del modello
    set_reproducibility_seeds()

    print(f"[Client {client_id}] === CREAZIONE RANDOM FOREST OTTIMIZZATO ===")
    print(f"[Client {client_id}] Modello: Random Forest con {RF_N_ESTIMATORS} alberi (diversificato)")
    print(f"[Client {client_id}] Criterio: {RF_CRITERION} (entropia per cyber-attacks)")
    print(f"[Client {client_id}] Max features: {RF_MAX_FEATURES} (feature selection automatica)")
    print(f"[Client {client_id}] Class weight: {RF_CLASS_WEIGHT} (gestione sbilanciamento locale)")
    print(f"[Client {client_id}] Max depth: {RF_MAX_DEPTH} (controllo overfitting)")
    
    # PARAMETRI OTTIMIZZATI PER FEDERATED LEARNING
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,           # Aumentato per diversità
        criterion=RF_CRITERION,                 # Entropy per cybersecurity
        max_depth=RF_MAX_DEPTH,                 # Limitato contro overfitting
        min_samples_split=RF_MIN_SAMPLES_SPLIT, # Aumentato contro overfitting
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,   # Aumentato contro overfitting
        max_features=RF_MAX_FEATURES,           # Feature selection sqrt
        bootstrap=RF_BOOTSTRAP,                 # Bootstrap sampling
        random_state=RANDOM_SEED + client_id,   # DIVERSIFICATO per client
        n_jobs=-1,                              # Parallelismo
        class_weight=RF_CLASS_WEIGHT,           # Balanced subsample per FL
        oob_score=True                          # Out-of-bag validation
    )
    
    print(f"[Client {client_id}] Parametri Random Forest Ottimizzati:")
    print(f"  - N. estimatori: {RF_N_ESTIMATORS} (diversificato)")
    print(f"  - Criterio: {RF_CRITERION}")
    print(f"  - Max depth: {RF_MAX_DEPTH} (contro overfitting)")
    print(f"  - Min samples split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  - Min samples leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  - Max features: {RF_MAX_FEATURES}")
    print(f"  - Bootstrap: {RF_BOOTSTRAP}")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED + client_id} (diversificato)")
    print(f"  - OOB Score: True")
    
    return model

def calculate_tree_diversity(tree1, tree2, X_sample):
    """
    Calcola la diversità tra due alberi basata sui patterns di predizione.
    Maggiore diversità = migliore per ensemble federato.
    
    Args:
        tree1, tree2: Due decision trees
        X_sample: Campione di dati per calcolare diversità
        
    Returns:
        float: Punteggio diversità [0,1] (1 = massima diversità)
    """
    try:
        pred1 = tree1.predict(X_sample)
        pred2 = tree2.predict(X_sample)
        
        # Calcola disagreement rate (diversità)
        disagreement = np.mean(pred1 != pred2)
        return float(disagreement)
    except:
        return 0.0

def extract_trees_from_forest(model, X_val, y_val):
    """
    Estrae gli alberi dal Random Forest e calcola le loro performance individuali REALI + DIVERSITÀ.
    Implementa la metodologia del paper per la selezione degli alberi migliori con diversificazione.
    
    Args:
        model: Random Forest addestrato
        X_val: Dati di validazione
        y_val: Etichette di validazione
        
    Returns:
        Lista di tuple (tree, accuracy_reale, weighted_accuracy_reale, diversity_score) per ogni albero
    """
    print(f"[Client {client_id}] === ESTRAZIONE ALBERI CON ACCURACY + DIVERSITÀ REALI ===")

    print(f"[Client {client_id}] 🔍 DEBUG extract_trees_from_forest: INIZIO")
    print(f"[Client {client_id}] 🔍 DEBUG: model type = {type(model)}")
    print(f"[Client {client_id}] 🔍 DEBUG: X_val shape = {X_val.shape}")
    print(f"[Client {client_id}] 🔍 DEBUG: y_val shape = {y_val.shape}")

    # CONTROLLO: Verifica se il modello è addestrato
    if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        print(f"[Client {client_id}] ⚠️ Modello non ancora addestrato, nessun albero disponibile")
        return []  # Restituisce lista vuota
    
    print(f"[Client {client_id}] 🔍 DEBUG: Modello ha {len(model.estimators_)} alberi")
    print(f"[Client {client_id}] === CALCOLO ACCURACY + DIVERSITÀ REALI PER {len(model.estimators_)} ALBERI ===")
    
    trees_performance = []
    
    # Campione per calcolo diversità
    sample_size = min(500, len(X_val))
    X_sample = X_val[:sample_size]
    
    for i, tree in enumerate(model.estimators_):
        print(f"[Client {client_id}] 🔍 DEBUG: Calcolo metrics per albero {i+1}/{len(model.estimators_)}")

        # Predizioni dell'albero singolo
        tree_predictions = tree.predict(X_val)
        
        # Calcola accuracy standard REALE
        accuracy_real = accuracy_score(y_val, tree_predictions)
        
        # Calcola weighted accuracy REALE
        # Weighted accuracy considera la distribuzione delle classi
        class_counts = np.bincount(y_val)
        weights = 1.0 / class_counts  # Peso inversamente proporzionale alla frequenza
        class_weights_norm = weights / weights.sum()  # Normalizza i pesi
        
        # Calcola accuracy pesata per classe REALE
        weighted_acc_real = 0.0
        for class_label in np.unique(y_val):
            class_mask = (y_val == class_label)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_val[class_mask], tree_predictions[class_mask])
                weighted_acc_real += class_accuracy * class_weights_norm[class_label]
        
        # NUOVO: Calcola diversity score medio rispetto agli altri alberi
        diversity_score = 0.0
        n_comparisons = 0
        
        # Confronta con al massimo 10 altri alberi per efficienza
        comparison_indices = np.random.choice(
            [idx for idx in range(len(model.estimators_)) if idx != i],
            size=min(10, len(model.estimators_) - 1),
            replace=False
        ) if len(model.estimators_) > 1 else []
        
        for j in comparison_indices:
            other_tree = model.estimators_[j]
            div_score = calculate_tree_diversity(tree, other_tree, X_sample)
            diversity_score += div_score
            n_comparisons += 1
        
        diversity_score = diversity_score / n_comparisons if n_comparisons > 0 else 0.0
        
        trees_performance.append((tree, accuracy_real, weighted_acc_real, diversity_score))
        
        if i < 5:  # Stampa info per i primi 5 alberi
            print(f"[Client {client_id}] Albero {i+1}: Accuracy={accuracy_real:.4f}, W_Accuracy={weighted_acc_real:.4f}, Diversity={diversity_score:.4f}")
    
    print(f"[Client {client_id}] 🔍 DEBUG extract_trees_from_forest: COMPLETATO con {len(trees_performance)} alberi CON ACCURACY + DIVERSITÀ REALI")

    # Ordina gli alberi per performance combinata (accuracy + diversity)
    def combined_score(tree_perf):
        _, acc, w_acc, div = tree_perf
        # Combina weighted accuracy (70%) + diversity (30%) per federated learning
        return 0.7 * w_acc + 0.3 * div
    
    trees_performance.sort(key=combined_score, reverse=True)
    
    best_tree = trees_performance[0]
    worst_tree = trees_performance[-1]
    print(f"[Client {client_id}] Migliore albero (COMBINATO): Acc={best_tree[1]:.4f}, W_Acc={best_tree[2]:.4f}, Div={best_tree[3]:.4f}")
    print(f"[Client {client_id}] Peggiore albero (COMBINATO): Acc={worst_tree[1]:.4f}, W_Acc={worst_tree[2]:.4f}, Div={worst_tree[3]:.4f}")
    
    return trees_performance

def serialize_trees_for_aggregation(trees_performance, max_trees=None):
    """
    Serializza gli alberi con le loro accuracy reali + diversità per l'invio al server.
    AGGIORNATO: Include diversity score per selezione server-side.
    """
    print(f"[Client {client_id}] === SERIALIZZAZIONE ALBERI CON ACCURACY + DIVERSITÀ REALI ===")
    
    if max_trees is not None:
        selected_trees = trees_performance[:max_trees]
        print(f"[Client {client_id}] Selezionati {len(selected_trees)} migliori alberi su {len(trees_performance)}")
    else:
        selected_trees = trees_performance
        print(f"[Client {client_id}] Invio tutti i {len(selected_trees)} alberi")
    
    serialized_data = []
    
    for i, (tree, accuracy_real, weighted_accuracy_real, diversity_score) in enumerate(selected_trees):
        try:
            # AGGIORNATO: Include diversity score
            tree_data = {
                'tree': tree,
                'accuracy': accuracy_real,
                'weighted_accuracy': weighted_accuracy_real,
                'diversity_score': diversity_score,  # NUOVO
                'tree_index': i,
                'client_id': client_id,  # NUOVO: identifica origine
                'accuracy_type': 'REAL_ENHANCED'  # Flag upgraded
            }
            
            # Serializza l'intero dizionario con pickle
            tree_bytes = pickle.dumps(tree_data, protocol=pickle.HIGHEST_PROTOCOL)

            # Converti in array uint8 (formato sicuro per Flower)
            tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
            serialized_data.append(tree_array)

            print(f"[Client {client_id}] ✅ Albero {i+1} serializzato ENHANCED ({len(tree_bytes)} bytes)")
            print(f"[Client {client_id}]    Acc={accuracy_real:.4f}, W_Acc={weighted_accuracy_real:.4f}, Div={diversity_score:.4f}")

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore serializzazione albero {i+1}: {e}")
            import traceback; traceback.print_exc()
            continue
    
    print(f"[Client {client_id}] Serializzati {len(serialized_data)} alberi con ACCURACY + DIVERSITÀ REALI")

    # ===== DEBUG FLOWER FORMAT =====
    if serialized_data:
        first = serialized_data[0]
        print(f"[Client {client_id}] DEBUG Primo albero serializzato ENHANCED:")
        print(f"  Tipo: {type(first)}, dtype: {first.dtype}, shape: {first.shape}")
        print(f"  Prime 10 byte: {first[:10].tolist()}")
    else:
        print(f"[Client {client_id}] ⚠️ Nessun albero serializzato!")

    return serialized_data

class SmartGridRandomForestClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con Random Forest.
    Implementa la metodologia ottimizzata per l'aggregazione federata di Random Forest.
    """
    
    def get_parameters(self, config):
        """
        Restituisce gli alberi serializzati del Random Forest locale.
        Gli alberi sono serializzati come numpy arrays (uint8) per compatibilità con Flower.
        """
        global model, X_val, y_val

        if model is None:
            print(f"[Client {client_id}] ⚠️ Modello non ancora addestrato, restituisco parametri vuoti")
            return []
        
        # CONTROLLO: Verifica se il modello è addestrato
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
            print(f"[Client {client_id}] ⚠️ Modello non ancora addestrato, restituisco parametri vuoti")
            return []
        
        print(f"[Client {client_id}] 🔍 DEBUG PRE-GET_PARAMETERS:")
        print(f"  - model type: {type(model)}")
        print(f"  - has estimators_: {hasattr(model, 'estimators_')}")
        if hasattr(model, 'estimators_'):
            print(f"  - n_estimators: {len(model.estimators_)}")

        print(f"[Client {client_id}] 🔍 DEBUG: Modello è addestrato con {len(model.estimators_)} alberi")

        try:
            print(f"[Client {client_id}] 🔍 DEBUG: Chiamo extract_trees_from_forest ENHANCED...")
            # Estrai e valuta le performance degli alberi CON DIVERSITÀ
            trees_performance = extract_trees_from_forest(model, X_val, y_val)
            print(f"[Client {client_id}] 🔍 DEBUG: extract_trees_from_forest completata, {len(trees_performance)} alberi ENHANCED")

            print(f"[Client {client_id}] 🔍 DEBUG: Chiamo serialize_trees_for_aggregation ENHANCED...")
            # Serializza gli alberi con verifica
            serialized_trees = serialize_trees_for_aggregation(trees_performance)
            print(f"[Client {client_id}] 🔍 DEBUG: serialize_trees_for_aggregation completata, {len(serialized_trees)} alberi ENHANCED")

            # Debug se non ci sono alberi serializzati
            if len(serialized_trees) == 0:
                print(f"[Client {client_id}] ⚠️ Nessun albero serializzato — invio parametri vuoti")
                return []
        
            print(f"[Client {client_id}] 🔍 DEBUG: Invio {len(serialized_trees)} alberi ENHANCED al server")
            # Gli alberi sono già numpy arrays (uint8) pronti per Flower
            print(f"[Client {client_id}] Invio {len(serialized_trees)} alberi ENHANCED al server")
            print(f"[Client {client_id}] Primo albero: shape={serialized_trees[0].shape}, dtype={serialized_trees[0].dtype}")
            return serialized_trees
            
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore nell'estrazione parametri: {e}")
            import traceback
            traceback.print_exc()
            return []

    def set_parameters(self, parameters):
        """
        Riceve e deserializza il modello aggregato dal server.
        Il modello è ricevuto come numpy array (uint8) serializzato con pickle.
        """
        global model

        if not parameters or len(parameters) == 0:
            print(f"[Client {client_id}] ❌ Nessun parametro ricevuto dal server")
            return

        try:
            if len(parameters) > 0:
                # Il server invia un singolo modello Random Forest aggregato
                model_array = parameters[0]

                # Debug del tipo di parametro ricevuto
                print(f"[Client {client_id}] Tipo parametro ricevuto: {type(model_array)}")
                
                # Converte numpy array in bytes
                if isinstance(model_array, np.ndarray):
                    model_bytes = model_array.tobytes()
                    print(f"[Client {client_id}] Convertito numpy array in bytes: {len(model_bytes)} bytes")
                elif isinstance(model_array, bytes):
                    model_bytes = model_array
                    print(f"[Client {client_id}] Ricevuto bytes direttamente: {len(model_bytes)} bytes")
                else:
                    print(f"[Client {client_id}] ⚠️ Tipo parametro non supportato: {type(model_array)}")
                    return
                
                # Deserializza il modello Random Forest
                model = pickle.loads(model_bytes)
                print(f"[Client {client_id}] ✅ Modello aggregato ENHANCED ricevuto dal server")
                print(f"[Client {client_id}] Nuovo modello ha {model.n_estimators} alberi diversificati")
                    
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore nell'impostazione parametri: {e}")
            import traceback
            traceback.print_exc()
            # Mantieni il modello corrente in caso di errore
            pass

    def fit(self, parameters, config):
        """
        Addestra il modello Random Forest locale ottimizzato.
        """
        global model, X_train, y_train, dataset_info

        # Imposta semi per riproducibilità dell'addestramento
        set_reproducibility_seeds()
    
        print(f"[Client {client_id}] Round di addestramento Random Forest OTTIMIZZATO...")
    
        # Imposta parametri se ricevuti dal server
        if parameters:
            self.set_parameters(parameters)
    
        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training!")
            return [], 0, {}
    
        try:
            # Verifica che i dati siano puliti
            if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
                print(f"[Client {client_id}] ⚠️ Dati contengono inf/NaN, applico pulizia...")
                X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
            else:
                X_train_clean = X_train
        
            # Addestra il Random Forest locale
            print(f"[Client {client_id}] Addestramento Random Forest OTTIMIZZATO su {len(X_train_clean)} campioni...")
            model.fit(X_train_clean, y_train)
        
            # Verifica che il modello sia stato addestrato
            if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
                raise RuntimeError("Random Forest non addestrato correttamente - nessun albero trovato")
        
            print(f"[Client {client_id}] ✅ Random Forest OTTIMIZZATO addestrato con {len(model.estimators_)} alberi")

            # DOPO l'addestramento, aggiungi:
            print(f"[Client {client_id}] 🔍 DEBUG POST-FIT OTTIMIZZATO:")
            print(f"  - model type: {type(model)}")
            print(f"  - has estimators_: {hasattr(model, 'estimators_')}")
            if hasattr(model, 'estimators_'):
                print(f"  - n_estimators: {len(model.estimators_)}")
                print(f"  - first tree type: {type(model.estimators_[0]) if len(model.estimators_) > 0 else 'N/A'}")
                print(f"  - random_state diversificato: {model.random_state}")
        
            # Calcola metriche di training
            train_predictions = model.predict(X_train_clean)
            train_prob = model.predict_proba(X_train_clean)[:, 1]  # Probabilità classe positiva
        
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_precision = precision_score(y_train, train_predictions, zero_division=0)
            train_recall = recall_score(y_train, train_predictions, zero_division=0)
            train_f1 = f1_score(y_train, train_predictions, zero_division=0)
            train_balanced_acc = balanced_accuracy_score(y_train, train_predictions)
        
            # AUC se abbiamo probabilità
            try:
                train_auc = roc_auc_score(y_train, train_prob)
            except:
                train_auc = 0.0
        
            # Out-of-bag score se disponibile
            oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else 0.0
        
            print(f"[Client {client_id}] Training OTTIMIZZATO completato!")
            print(f"[Client {client_id}] Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
            print(f"[Client {client_id}] Balanced Acc: {train_balanced_acc:.4f}, OOB Score: {oob_score:.4f}")
        
        except Exception as e:
            print(f"[Client {client_id}] Errore durante addestramento: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, {'error': f'training_failed: {str(e)}'}
    
        # Metriche da inviare al server
        metrics = {
            # Metriche base
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            'train_balanced_accuracy': float(train_balanced_acc),
            'train_auc': float(train_auc),
            'oob_score': float(oob_score),
        
            # Info modello
            'n_estimators': int(len(model.estimators_)),
            'n_features': int(model.n_features_in_),
        
            # Dataset info
            'client_id': int(dataset_info['client_id']),
            'train_samples': int(dataset_info['train_samples']),
            
            # NUOVI: Info ottimizzazioni
            'feature_engineering_enabled': bool(dataset_info.get('feature_engineering_enabled', False)),
            'enhanced_preprocessing': bool(ENABLE_SCALING and ENABLE_REMOVE_NEAR_CONSTANT_FEATURES),
            'random_state_diversified': int(model.random_state),
        }
    
        # Restituisce gli alberi del modello addestrato
        try:
            # Calcola accuracy reali + diversità per ogni albero usando validation set
            trees_perf_real = extract_trees_from_forest(model, X_val, y_val)
            serialized_trees = serialize_trees_for_aggregation(trees_perf_real)
            
            print(f"[Client {client_id}] Invio {len(serialized_trees)} alberi CON ACCURACY + DIVERSITÀ REALI al server...")
            return serialized_trees, len(X_train), metrics

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore serializzazione finale: {e}")
            import traceback; traceback.print_exc()
            return [], 0, {'error': f'serialization_failed: {str(e)}'}

    def evaluate(self, parameters, config):
        """
        Valuta il modello Random Forest ottimizzato.
        """
        global model, X_val, y_val

        # Imposta semi per riproducibilità della valutazione
        set_reproducibility_seeds()
        
        # Imposta parametri se ricevuti dal server
        if parameters:
            self.set_parameters(parameters)
        
        if model is None:
            print(f"[Client {client_id}] Modello non disponibile per valutazione")
            return 1.0, 0, {"accuracy": 0.0}
        
        if len(X_val) == 0:
            return 0.0, 0, {"accuracy": 0.0}
        
        # Verifica che il modello sia addestrato
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
            print(f"[Client {client_id}] ⚠️ Modello Random Forest non addestrato, uso accuracy 0")
            return 1.0, len(X_val), {"accuracy": 0.0, "error": "model_not_fitted"}
        
        try:
            # Verifica che i dati siano puliti per la valutazione
            if np.any(np.isinf(X_val)) or np.any(np.isnan(X_val)):
                print(f"[Client {client_id}] ⚠️ Dati val contengono inf/NaN, applico pulizia...")
                X_val_clean = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)
            else:
                X_val_clean = X_val
            
            # Valutazione Random Forest
            val_predictions = model.predict(X_val_clean)
            val_prob = model.predict_proba(X_val_clean)[:, 1]  # Probabilità classe positiva
            
            # Calcola metriche
            accuracy = accuracy_score(y_val, val_predictions)
            precision = precision_score(y_val, val_predictions, zero_division=0)
            recall = recall_score(y_val, val_predictions, zero_division=0)
            f1_score_val = f1_score(y_val, val_predictions, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_val, val_predictions)
            
            # AUC
            try:
                auc = roc_auc_score(y_val, val_prob)
            except:
                auc = 0.0
            
            # Metriche per classe
            report = classification_report(y_val, val_predictions, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_val, val_predictions)

            print(f"[Client {client_id}] Val Accuracy OTTIMIZZATO: {accuracy:.4f}, Val F1: {f1_score_val:.4f}")
            print(f"[Client {client_id}] Val Balanced Acc: {balanced_acc:.4f}, Val AUC: {auc:.4f}")
            print(f"[Client {client_id}] Classification report (per classe):")
            print(classification_report(y_val, val_predictions, target_names=["natural", "attack"], zero_division=0))
            print(f"[Client {client_id}] Confusion matrix:")
            print(f"tn: {conf_matrix[0, 0]}, fp: {conf_matrix[0, 1]}, fn: {conf_matrix[1, 0]}, tp: {conf_matrix[1, 1]}")
            
            # Simula loss per compatibilità (Random Forest non ha loss)
            loss = 1 - accuracy  # Loss simulata
            
            # Metriche
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "f1_score": f1_score_val,
                "balanced_accuracy": balanced_acc,
                "val_samples": len(X_val),
                "precision_natural": report["natural"]["precision"],
                "recall_natural": report["natural"]["recall"],
                "f1_natural": report["natural"]["f1-score"],
                "precision_attack": report["attack"]["precision"],
                "recall_attack": report["attack"]["recall"],
                "f1_attack": report["attack"]["f1-score"],
                "support_natural": report["natural"]["support"],
                "support_attack": report["attack"]["support"],
                # Confusion matrix
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1]),
                
                # NUOVI: Info modello ottimizzato
                "model_n_estimators": int(model.n_estimators),
                "model_max_depth": model.max_depth,
                "enhanced_features": bool(ENABLE_FEATURE_ENGINEERING),
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante valutazione: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

def main():
    """
    Funzione principale per avviare il client SmartGrid Random Forest ottimizzato.
    """

    global client_id, model, X_train, y_train, X_val, y_val, dataset_info

    # Imposta semi all'avvio del client
    set_reproducibility_seeds()
    
    if len(sys.argv) != 2:
        print("Uso: python clientRF.py <client_id>")
        print("Esempio: python clientRF.py 1")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("⚠️ Client ID deve essere tra 1 e 13")
    except ValueError as e:
        print(f"❌ Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"=== AVVIO CLIENT RANDOM FOREST OTTIMIZZATO {client_id} ===")
    
    try:
        # Carica i dati con preprocessing ottimizzato per SmartGrid
        print(f"[Client {client_id}] Caricamento dati per Random Forest OTTIMIZZATO...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
        
        # Imposta semi all'avvio del client
        set_reproducibility_seeds()

        # Crea il modello Random Forest ottimizzato
        model = create_random_forest_model()

        print(f"[Client {client_id}] === RIASSUNTO CLIENT RANDOM FOREST OTTIMIZZATO ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['final_features']}")
        print(f"[Client {client_id}] Feature Engineering: {'ABILITATA' if dataset_info.get('feature_engineering_enabled') else 'DISABILITATA'}")
        print(f"[Client {client_id}] Modello: Random Forest OTTIMIZZATO con {model.n_estimators} alberi diversificati")
        print(f"[Client {client_id}] Criterio: {model.criterion}, Max features: {model.max_features}")
        print(f"[Client {client_id}] Max depth: {model.max_depth} (controllo overfitting)")
        print(f"[Client {client_id}] Random state: {model.random_state} (diversificato)")
        print(f"[Client {client_id}] Connessione al server su localhost:8080...")
        
        # Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridRandomForestClient()
        )
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()