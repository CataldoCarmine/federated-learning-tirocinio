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
ENABLE_IMPUTATION = False              # Imputazione mediana
ENABLE_SCALING = False                 # StandardScaler (mean=0, std=1) - ABILITATO
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti - ABILITATO
ENABLE_PCA = False  # PCA per riduzione dimensionalità
ENABLE_FEATURE_ENGINEERING = False    # NUOVA: Feature engineering per SmartGrid

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
RF_MAX_DEPTH = None         # LIMITATO da None per ridurre overfitting
RF_MIN_SAMPLES_SPLIT = 2  # AUMENTATO da 2 per ridurre overfitting
RF_MIN_SAMPLES_LEAF = 1   # AUMENTATO da 1 per ridurre overfitting
RF_MAX_FEATURES = 'sqrt'  # Feature da considerare per ogni split ('sqrt' dal paper)
RF_BOOTSTRAP = True       # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced_subsample'  # CAMBIATO per migliore gestione sbilanciamento locale
RF_CRITERION = 'entropy'  # Criterio di splitting (dal paper: entropy migliore di gini per molti dataset)

# CONFIGURAZIONE ENSEMBLE PER FEDERATED RANDOM FOREST
# Basato sulla metodologia del paper per aggregazione degli alberi
ENSEMBLE_METHOD = 'weighted_voting'  # 'simple_voting' o 'weighted_voting'
TREE_SELECTION_METHOD = 'weighted_accuracy'  # NUOVO: accuracy + diversity

def set_reproducibility_seeds(preserve_client_diversity=False):
    """
    Imposta tutti i semi per garantire riproducibilità.
    
    Args:
        preserve_client_diversity: Se True, non sovrascrive i semi già impostati
                                  per mantenere la diversità tra client
    """
    if not preserve_client_diversity:
        # Seed globali per operazioni deterministiche
        np.random.seed(RANDOM_SEED)
        import random
        random.seed(RANDOM_SEED)
        os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    else:
        # Solo PYTHONHASHSEED per evitare problemi di hash
        os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

def create_smartgrid_features(df, client_id):
    """
    Crea feature ingegnerizzate specifiche per il dataset SmartGrid.
    VERSIONE CORRETTA: Previene NaN/Inf con validazione robusta E garantisce numero feature fisso.
    """
    if not ENABLE_FEATURE_ENGINEERING:
        return df
        
    # ✅ SEED FISSO per operazioni deterministiche
    np.random.seed(RANDOM_SEED)
    
    print(f"[{client_id}] === FEATURE ENGINEERING SMARTGRID CLIENT ROBUSTO ===")
    
    # Copia il dataframe
    df_enhanced = df.copy()
    
    # Gestisci sia il caso con che senza colonna marker
    if 'marker' in df_enhanced.columns:
        original_features = len(df_enhanced.columns) - 1
        feature_cols = [col for col in df_enhanced.columns if col != 'marker']
    else:
        original_features = len(df_enhanced.columns)
        feature_cols = list(df_enhanced.columns)
    
    print(f"[{client_id}] 🔍 DEBUG FEATURE ENGINEERING DETERMINISTICO:")
    print(f"[{client_id}]   DataFrame shape iniziale: {df_enhanced.shape}")
    print(f"[{client_id}]   Feature originali: {original_features}")
    
    # ✅ Seleziona solo colonne numeriche e ORDINA per determinismo
    numeric_cols = df_enhanced[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = sorted(numeric_cols)
    
    if len(numeric_cols) == 0:
        print(f"[{client_id}] ⚠️ Nessuna colonna numerica trovata per feature engineering")
        return df_enhanced
    
    print(f"[{client_id}]   Colonne numeriche ordinate: {len(numeric_cols)}")
    
    # Converti in numpy per efficienza
    X = df_enhanced[numeric_cols].values
    
    # ✅ PARAMETRI FISSI per garantire stesso numero di feature
    FIXED_WINDOW_SIZE = 10
    FIXED_MAX_RATIOS = 50
    FIXED_MAX_ANOMALY = 15
    FIXED_MAX_INTERACTIONS = 20
    
    features_added = 0
    nan_created = 0
    inf_created = 0
    
    # 1. STATISTICAL FEATURES con parametri fissi (INVARIATO)
    try:
        window_size = min(FIXED_WINDOW_SIZE, len(numeric_cols))
        stat_features_added = 0
        
        for i in range(0, len(numeric_cols), window_size):
            end_idx = min(i + window_size, len(numeric_cols))
            window_data = X[:, i:end_idx]
            
            # ✅ Gestione robusta con controllo NaN/Inf
            mean_val = np.mean(window_data, axis=1)
            std_val = np.std(window_data, axis=1)
            range_val = np.ptp(window_data, axis=1)
            
            # Calcola skew con gestione errori
            try:
                skew_val = stats.skew(window_data, axis=1, nan_policy='propagate')
                # Sostituisci NaN/Inf con valori sicuri
                skew_val = np.nan_to_num(skew_val, nan=0.0, posinf=3.0, neginf=-3.0)
            except:
                skew_val = np.zeros(len(window_data))
            
            df_enhanced[f'window_{i}_mean'] = mean_val
            df_enhanced[f'window_{i}_std'] = std_val
            df_enhanced[f'window_{i}_range'] = range_val
            df_enhanced[f'window_{i}_skew'] = skew_val
            stat_features_added += 4
        
        features_added += stat_features_added
        print(f"[{client_id}] ✅ Aggiunte {stat_features_added} statistical features ROBUSTE")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore statistical features: {e}")
    
    # 2. RATIO FEATURES con CLIP PREVENTIVO
    try:
        n_ratios = 0
        for i in range(0, min(20, len(numeric_cols))):
            for j in range(i+1, min(i+5, len(numeric_cols))):
                if n_ratios >= FIXED_MAX_RATIOS:
                    break
                    
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                
                # ✅ VALIDAZIONE ROBUSTA: Verifica denominatore
                numerator = df_enhanced[col_i].values
                denominator = df_enhanced[col_j].values
                
                # ✅ CLIP VALORI ESTREMI **PRIMA** DELLA DIVISIONE
                numerator_clipped = np.clip(numerator, -1e6, 1e6)
                denominator_clipped = np.clip(denominator, -1e6, 1e6)
                
                # Maschera per valori validi (denominatore != 0)
                valid_mask = np.abs(denominator_clipped) > 1e-10
                
                ratio_val = np.zeros(len(numerator_clipped))
                
                if np.any(valid_mask):
                    # Calcola ratio con valori già clippati
                    ratio_val[valid_mask] = numerator_clipped[valid_mask] / denominator_clipped[valid_mask]
                    
                    # ✅ SECONDO CLIP per sicurezza (raramente necessario ora)
                    ratio_val = np.clip(ratio_val, -1e6, 1e6)
                    
                    # Imputa valori non validi con mediana
                    median_ratio = np.median(ratio_val[valid_mask]) if np.sum(valid_mask) > 0 else 0.0
                    ratio_val[~valid_mask] = median_ratio
                else:
                    # ✅ Se NESSUN valore valido, usa 0 per tutti
                    ratio_val[:] = 0.0
                
                # ✅ SEMPRE aggiungi la feature (deterministico)
                df_enhanced[f'ratio_{i}_{j}'] = ratio_val
                n_ratios += 1
            
            if n_ratios >= FIXED_MAX_RATIOS:
                break
        
        features_added += n_ratios
        print(f"[{client_id}] ✅ Aggiunti {n_ratios} ratio features CON CLIP PREVENTIVO")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore ratio features: {e}")
    
    # 3. ANOMALY INDICATORS con VALIDAZIONE ROBUSTA (INVARIATO)
    try:
        n_zscore = 0
        for i, col in enumerate(numeric_cols[:FIXED_MAX_ANOMALY]):
            col_data = df_enhanced[col].values
            col_mean = np.mean(col_data)
            col_std = np.std(col_data)
            
            # ✅ VALIDAZIONE: Solo se std > 0
            if col_std > 1e-10:
                zscore_val = np.abs((col_data - col_mean) / col_std)
                # ✅ Clip valori estremi per prevenire inf
                zscore_val = np.clip(zscore_val, 0.0, 10.0)
                df_enhanced[f'zscore_{i}'] = zscore_val
                n_zscore += 1
            else:
                # Se std = 0, tutti i valori sono costanti → zscore = 0
                df_enhanced[f'zscore_{i}'] = np.zeros(len(col_data))
                n_zscore += 1
        
        features_added += n_zscore
        print(f"[{client_id}] ✅ Aggiunti {n_zscore} anomaly indicators ROBUSTI (max {FIXED_MAX_ANOMALY})")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore anomaly features: {e}")
    
    # 4. INTERACTION FEATURES con VALIDAZIONE (INVARIATO)
    try:
        n_interactions = 0
        for i in range(0, min(10, len(numeric_cols))):
            for j in range(i+1, min(i+3, len(numeric_cols))):
                if n_interactions >= FIXED_MAX_INTERACTIONS:
                    break
                    
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                
                # ✅ Calcola prodotto e clip per prevenire overflow
                interaction = df_enhanced[col_i] * df_enhanced[col_j]
                interaction = np.clip(interaction, -1e10, 1e10)
                
                df_enhanced[f'interact_{i}_{j}'] = interaction
                n_interactions += 1
            
            if n_interactions >= FIXED_MAX_INTERACTIONS:
                break
        
        features_added += n_interactions
        print(f"[{client_id}] ✅ Aggiunte {n_interactions} interaction features ROBUSTE (max {FIXED_MAX_INTERACTIONS})")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore interaction features: {e}")
    
    # ✅ VERIFICA FINALE: Conta NaN/Inf creati
    new_cols = [col for col in df_enhanced.columns if col not in df.columns and col != 'marker']
    if new_cols:
        new_data = df_enhanced[new_cols].select_dtypes(include=[np.number])
        nan_created = new_data.isna().sum().sum()
        inf_created = np.isinf(new_data.values).sum()
    
    new_features = len(df_enhanced.columns) - original_features
    if 'marker' in df_enhanced.columns:
        new_features -= 1
    
    print(f"[{client_id}] 🎯 Feature engineering ROBUSTO E DETERMINISTICO completato:")
    print(f"[{client_id}]   Features originali: {original_features}")
    print(f"[{client_id}]   Features aggiunte: {new_features}")
    print(f"[{client_id}]   Features totali: {original_features + new_features}")
    print(f"[{client_id}]   Shape finale: {df_enhanced.shape}")
    print(f"[{client_id}]   ✅ NaN creati: {nan_created} (target: <1000)")
    print(f"[{client_id}]   ✅ Inf creati: {inf_created} (target: <100)")
    
    if nan_created > 1000:
        print(f"[{client_id}]   ⚠️ ATTENZIONE: Troppi NaN creati dal feature engineering!")
    if inf_created > 100:
        print(f"[{client_id}]   ⚠️ ATTENZIONE: Troppi Inf creati dal feature engineering!")
    
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
    VERSIONE OTTIMIZZATA: Preprocessing minimale per Random Forest.
    """
    set_reproducibility_seeds()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")
    
    df = pd.read_csv(file_path)
    
    print(f"=== PREPROCESSING FEDERATO RANDOM FOREST - VERSIONE STABILE ===")
    print(f"Feature engineering: DISABILITATA")
    print(f"Imputazione mediana: DISABILITATA (RF gestisce NaN nativamente)")

    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"[Client {client_id}] Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
    
    # STEP 1: Pulizia SOLO inf (NaN sono gestiti da RF)
    print(f"[Client {client_id}] Pulizia valori infiniti...")
    X_cleaned = clean_data_for_pca(X)
    X_array = np.array(X_cleaned, dtype=float)
    
    # Gestisci SOLO infiniti (non NaN)
    inf_mask = np.isinf(X_array)
    if np.any(inf_mask):
        print(f"[Client {client_id}] Trovati {np.sum(inf_mask)} valori infiniti, li sostituisco...")
        for col in range(X_array.shape[1]):
            col_data = X_array[:, col]
            finite_mask = np.isfinite(col_data)
            if np.any(finite_mask):
                percentile_99 = np.percentile(col_data[finite_mask], 99.9)
                percentile_01 = np.percentile(col_data[finite_mask], 0.1)
                X_array[np.isposinf(col_data), col] = percentile_99
                X_array[np.isneginf(col_data), col] = percentile_01
            else:
                X_array[:, col] = 0.0

    # Suddivisione train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_array, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"[Client {client_id}] Suddivisione: {len(X_train_raw)} training, {len(X_val_raw)} validation")

    # NESSUN preprocessing aggiuntivo - RF è robusto
    X_train_final = X_train_raw
    X_val_final = X_val_raw
    
    print(f"[Client {client_id}] ✅ Preprocessing minimale completato: {X_train_final.shape}, {X_val_final.shape}")
    
    # Verifica NaN finali
    nan_count_train = np.isnan(X_train_final).sum()
    nan_count_val = np.isnan(X_val_final).sum()
    print(f"[Client {client_id}] NaN nel training set: {nan_count_train} (gestiti da RF)")
    print(f"[Client {client_id}] NaN nel validation set: {nan_count_val} (gestiti da RF)")
        
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
        'pca_enabled': False,
        'feature_engineering_enabled': False,
        'preprocessing_method': 'minimal_for_rf',
        'compatibility_guaranteed': True
    }
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO ===")
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_random_forest_model():
    """
    Crea il modello Random Forest per SmartGrid.
    VERSIONE STABILE: Configurazione standard ottimizzata.
    """
    set_reproducibility_seeds()

    print(f"[Client {client_id}] === CREAZIONE RANDOM FOREST STABILE ===")
    print(f"[Client {client_id}] Modello: Random Forest con {RF_N_ESTIMATORS} alberi")
    print(f"[Client {client_id}] Configurazione: Standard ottimizzata per SmartGrid")

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,      
        criterion=RF_CRITERION,             
        max_depth=RF_MAX_DEPTH,             # None = massima capacità
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        bootstrap=RF_BOOTSTRAP,
        random_state=RANDOM_SEED,           # ✅ SEED FISSO per tutti i client
        n_jobs=-1,
        class_weight=RF_CLASS_WEIGHT,
        oob_score=True
    )
    
    print(f"[Client {client_id}] Parametri Random Forest:")
    print(f"  - N. estimatori: {RF_N_ESTIMATORS}")
    print(f"  - Criterio: {RF_CRITERION}")
    print(f"  - Max depth: {RF_MAX_DEPTH} (illimitata per max capacità)")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED} (fisso)")
    
    return model

def calculate_tree_diversity(tree1, tree2, X_sample):
    """
    Calcola la diversità tra due alberi con sampling RANDOMIZZATO per massimizzare diversità.
    
    Args:
        tree1, tree2: Due decision trees
        X_sample: Campione di dati per calcolare diversità
        
    Returns:
        float: Punteggio diversità [0,1] (1 = massima diversità)
    """
    try:
        # ✅ SAMPLING RANDOMIZZATO per catturare vera diversità
        # NON usiamo seed fisso qui per massimizzare la diversità rilevata
        sample_size = min(200, len(X_sample))  # Aumentato da 100 a 200
        
        # ✅ Random sampling invece di linspace deterministico
        if len(X_sample) > sample_size:
            # Usa random state locale per non interferire con training
            rng = np.random.RandomState(seed=None)  # Seed None = random vero
            sample_indices = rng.choice(len(X_sample), size=sample_size, replace=False)
        else:
            sample_indices = np.arange(len(X_sample))
        
        X_random = X_sample[sample_indices]
        
        # Predizioni degli alberi
        pred1 = tree1.predict(X_random)
        pred2 = tree2.predict(X_random)
        
        # Calcola disagreement rate (diversità)
        disagreement = np.mean(pred1 != pred2)
        
        return float(disagreement)
    except Exception as e:
        print(f"⚠️ Errore calcolo diversità: {e}")
        return 0.0

def extract_trees_from_forest(model, X_val, y_val):
    """
    Estrae gli alberi dal Random Forest e calcola le loro performance individuali REALI.
    VERSIONE STABILE: Solo weighted accuracy, nessuna diversity.
    """
    print(f"[Client {client_id}] === ESTRAZIONE ALBERI CON WEIGHTED ACCURACY REALI ===")

    if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        print(f"[Client {client_id}] ⚠️ Modello non ancora addestrato, nessun albero disponibile")
        return []
    
    print(f"[Client {client_id}] Calcolo weighted accuracy per {len(model.estimators_)} alberi")
    
    trees_performance = []
    
    for i, tree in enumerate(model.estimators_):
        tree_predictions = tree.predict(X_val)
        
        # Accuracy standard
        accuracy_real = accuracy_score(y_val, tree_predictions)
        
        # Weighted accuracy
        class_counts = np.bincount(y_val)
        weights = 1.0 / class_counts
        class_weights_norm = weights / weights.sum()
        
        weighted_acc_real = 0.0
        for class_label in np.unique(y_val):
            class_mask = (y_val == class_label)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_val[class_mask], tree_predictions[class_mask])
                weighted_acc_real += class_accuracy * class_weights_norm[class_label]
        
        trees_performance.append((tree, accuracy_real, weighted_acc_real))
        
        if i < 5:
            print(f"[Client {client_id}] Albero {i+1}: Acc={accuracy_real:.4f}, W_Acc={weighted_acc_real:.4f}")
    
    # Ordina per weighted accuracy
    trees_performance.sort(key=lambda x: x[2], reverse=True)
    
    best_tree = trees_performance[0]
    worst_tree = trees_performance[-1]
    print(f"[Client {client_id}] Migliore albero: Acc={best_tree[1]:.4f}, W_Acc={best_tree[2]:.4f}")
    print(f"[Client {client_id}] Peggiore albero: Acc={worst_tree[1]:.4f}, W_Acc={worst_tree[2]:.4f}")
    
    print(f"[Client {client_id}] ✅ Estrazione completata - {len(trees_performance)} alberi")
    
    return trees_performance

def serialize_trees_for_aggregation(trees_performance, max_trees=None):
    """
    Serializza gli alberi con le loro weighted accuracy reali.
    VERSIONE STABILE: Formato semplificato senza diversity.
    """
    print(f"[Client {client_id}] === SERIALIZZAZIONE ALBERI (WEIGHTED ACCURACY) ===")
    
    if max_trees is not None:
        selected_trees = trees_performance[:max_trees]
        print(f"[Client {client_id}] Selezionati {len(selected_trees)} migliori alberi")
    else:
        selected_trees = trees_performance
        print(f"[Client {client_id}] Invio tutti i {len(selected_trees)} alberi")
    
    serialized_data = []
    
    for i, (tree, accuracy_real, weighted_accuracy_real) in enumerate(selected_trees):
        try:
            tree_data = {
                'tree': tree,
                'accuracy': accuracy_real,
                'weighted_accuracy': weighted_accuracy_real,
                'tree_index': i,
                'client_id': client_id,
                'accuracy_type': 'REAL'  # ✅ Formato standard
            }
            
            tree_bytes = pickle.dumps(tree_data, protocol=pickle.HIGHEST_PROTOCOL)
            tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
            serialized_data.append(tree_array)

            if i < 3:  # Mostra solo primi 3
                print(f"[Client {client_id}] Albero {i+1}: Acc={accuracy_real:.4f}, W_Acc={weighted_accuracy_real:.4f}")

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore serializzazione albero {i+1}: {e}")
            continue
    
    print(f"[Client {client_id}] Serializzati {len(serialized_data)} alberi")
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

        # ✅ PRESERVA diversità tra client
        set_reproducibility_seeds(preserve_client_diversity=True)
    
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
        Valuta il modello Random Forest - VERSIONE STABILE.
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

            print(f"[Client {client_id}] Val Accuracy: {accuracy:.4f}, Val F1: {f1_score_val:.4f}")
            print(f"[Client {client_id}] Val Balanced Acc: {balanced_acc:.4f}, Val AUC: {auc:.4f}")
            print(f"[Client {client_id}] Classification report (per classe):")
            print(classification_report(y_val, val_predictions, target_names=["natural", "attack"], zero_division=0))
            print(f"[Client {client_id}] Confusion matrix:")
            print(f"tn: {conf_matrix[0, 0]}, fp: {conf_matrix[0, 1]}, fn: {conf_matrix[1, 0]}, tp: {conf_matrix[1, 1]}")
            
            # Simula loss per compatibilità (Random Forest non ha loss)
            loss = 1 - accuracy  # Loss simulata
            
            # ✅ CORREZIONE: Converti None in valore valido per Flower
            model_max_depth = model.max_depth if model.max_depth is not None else -1
            
            # Metriche
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "balanced_accuracy": float(balanced_acc),
                "val_samples": int(len(X_val)),
                "precision_natural": float(report["natural"]["precision"]),
                "recall_natural": float(report["natural"]["recall"]),
                "f1_natural": float(report["natural"]["f1-score"]),
                "precision_attack": float(report["attack"]["precision"]),
                "recall_attack": float(report["attack"]["recall"]),
                "f1_attack": float(report["attack"]["f1-score"]),
                "support_natural": int(report["natural"]["support"]),
                "support_attack": int(report["attack"]["support"]),
                # Confusion matrix
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1]),
                
                # ✅ CORRETTO: model_max_depth non è mai None ora
                "model_n_estimators": int(model.n_estimators),
                "model_max_depth": int(model_max_depth),  # -1 se illimitato
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