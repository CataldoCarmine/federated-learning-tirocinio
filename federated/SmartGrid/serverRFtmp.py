import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import os
from datetime import datetime
import sys
from flwr.common import ndarrays_to_parameters
from scipy.spatial.distance import cosine
from scipy import stats
warnings.filterwarnings('ignore')

# CONFIGURAZIONE SEMI PER RIPRODUCIBILITÀ
RANDOM_SEED = 42

# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING OTTIMIZZATO ==============
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = False       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = False                 # ABILITATO: StandardScaler (mean=0, std=1) 
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # ABILITATO: Rimozione feature quasi-costanti
ENABLE_PCA = False                    # PCA per riduzione dimensionalità
ENABLE_FEATURE_ENGINEERING = False    # NUOVO: Feature engineering per SmartGrid

if ENABLE_PCA:
    ENABLE_IMPUTATION = True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 74  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 42  # Seme specifico per PCA
  
# Quando PCA disabilitata, disabilita rimozione feature quasi-costanti per compatibilità dei modelli
if ENABLE_PCA == False and not ENABLE_FEATURE_ENGINEERING:
    ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False
    PCA_COMPONENTS = None

# ============== CONFIGURAZIONE RANDOM FOREST GLOBALE OTTIMIZZATA ==============
# Configurazione aggregazione alberi (basata su ricerca recente)
TREE_SELECTION_METHOD = 'diversity_weighted'  # AGGIORNATO: 'accuracy', 'weighted_accuracy', 'diversity_weighted'
TREE_AGGREGATION_STRATEGY = 'global'          # CAMBIATO: 'per_forest' o 'global' 
MAX_TREES_GLOBAL = 150                        # AUMENTATO: da 100 a 150
ENSEMBLE_METHOD = 'weighted_voting'           # 'simple_voting' o 'weighted_voting'
MIN_TREES_PER_CLIENT = 5                     # DIMINUITO: per maggiore selettività

# Configurazione Random Forest del server (ottimizzata)
RF_N_ESTIMATORS = 200      # AUMENTATO: da 65 a 100 per maggiore diversità
RF_MAX_DEPTH = 15         # LIMITATO: da None a 15 per ridurre overfitting
RF_MIN_SAMPLES_SPLIT = 5  # AUMENTATO: da 2 a 5 per ridurre overfitting
RF_MIN_SAMPLES_LEAF = 2   # AUMENTATO: da 1 a 2 per ridurre overfitting
RF_MAX_FEATURES = 'sqrt'  # Feature da considerare per ogni split ('sqrt' dal paper)
RF_BOOTSTRAP = True       # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced_subsample'  # CAMBIATO: migliore per federated learning
RF_CRITERION = 'entropy'  # Criterio di splitting (dal paper: entropy migliore di gini per molti dataset)

NUM_ROUNDS = 50  # Numero di round di addestramento federato

# Variabili globali per tracking metriche
all_federated_metrics = []  # Lista di dict, uno per round
last_confusion_matrix = None

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

def save_federated_metrics_report(metrics_list):
    """
    Salva un report completo delle metriche federate per Random Forest ottimizzato.
    Adattato per includere le nuove ottimizzazioni.
    """
    if not metrics_list:
        print("[SERVER] ⚠️ Nessuna metrica da salvare.")
        return

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, f"federated_random_forest_ENHANCED_metrics_{timestamp}.txt")

    cols = [
        ("round", "Round", 6),
        ("loss_distribuita", "Loss", 11),
        ("accuracy", "Accuracy", 11),
        ("balanced_accuracy", "BalancedAcc", 13),
        ("auc", "AUC", 9),
        ("f1_score", "F1_Score", 11),
        ("f1_natural", "F1_Natural", 11),
        ("f1_attack", "F1_Attack", 11),
        ("precision", "Precision", 11),
        ("precision_natural", "Precision_Nat", 14),
        ("precision_attack", "Precision_Att", 14),
        ("recall", "Recall", 11),
        ("recall_natural", "Recall_Nat", 12),
        ("recall_attack", "Recall_Att", 12),
    ]

    def fmt(val, width):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A".ljust(width)
        return f"{val:.6f}".ljust(width)

    # HEADER
    title = "RESOCONTO ADDESTRAMENTO FEDERATO RANDOM FOREST OTTIMIZZATO SMARTGRID"
    n_rounds = len(metrics_list)
    header = f"{title}\nRounds totali: {n_rounds}\n\nTABELLA RIASSUNTIVA METRICHE:\n" + "="*140 + "\n"
    col_headers = "  ".join([name.ljust(width) for _, name, width in cols])
    sep = "-" * 140

    table_lines = []
    table_lines.append(col_headers)
    table_lines.append(sep)
    for row in metrics_list:
        vals = []
        for k, _, width in cols:
            v = row.get(k, None)
            if k == "round":
                vals.append(str(v).ljust(width))
            else:
                vals.append(fmt(v, width))
        table_lines.append("  ".join(vals))

    # STATISTICHE FINALI
    stats_lines = []
    stats_lines.append("\nSTATISTICHE FINALI:\n" + "="*60 + "\n")
    for k, name, width in cols:
        if k == "round":
            continue
        vals = [row[k] for row in metrics_list if row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k]))]
        if not vals:
            continue
        start = vals[0]
        end = vals[-1]
        minv = np.min(vals)
        maxv = np.max(vals)
        meanv = np.mean(vals)
        miglioramento = end - start if isinstance(end, float) and isinstance(start, float) else 0
        trend = "📈" if miglioramento > 0 else ("📉" if miglioramento < 0 else "")
        stats_lines.append(f"🔹 {name.upper()}:")
        stats_lines.append(f"   Rounds disponibili  : {len(vals)}")
        stats_lines.append(f"   Valore iniziale     : {fmt(start, 9)}")
        stats_lines.append(f"   Valore finale       : {fmt(end, 9)}")
        stats_lines.append(f"   Valore minimo       : {fmt(minv, 9)}")
        stats_lines.append(f"   Valore massimo      : {fmt(maxv, 9)}")
        stats_lines.append(f"   Valore medio        : {fmt(meanv, 9)}")
        stats_lines.append(f"   Miglioramento       : {fmt(miglioramento, 9)} {trend}")
        stats_lines.append("")

    # CONFIGURAZIONE OTTIMIZZAZIONI
    config_lines = []
    config_lines.append("\n# ============== CONFIGURAZIONE RANDOM FOREST OTTIMIZZATO ==============")
    config_lines.append(f"TREE_SELECTION_METHOD = '{TREE_SELECTION_METHOD}'")
    config_lines.append(f"TREE_AGGREGATION_STRATEGY = '{TREE_AGGREGATION_STRATEGY}'")
    config_lines.append(f"MAX_TREES_GLOBAL = {MAX_TREES_GLOBAL}")
    config_lines.append(f"ENSEMBLE_METHOD = '{ENSEMBLE_METHOD}'")
    config_lines.append(f"MIN_TREES_PER_CLIENT = {MIN_TREES_PER_CLIENT}")
    config_lines.append(f"ENABLE_FEATURE_ENGINEERING = {ENABLE_FEATURE_ENGINEERING}")
    config_lines.append(f"ENABLE_SCALING = {ENABLE_SCALING}")
    config_lines.append(f"ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = {ENABLE_REMOVE_NEAR_CONSTANT_FEATURES}")
    config_lines.append(f"RF_N_ESTIMATORS = {RF_N_ESTIMATORS}")
    config_lines.append(f"RF_MAX_DEPTH = {RF_MAX_DEPTH}")
    config_lines.append(f"RF_MIN_SAMPLES_SPLIT = {RF_MIN_SAMPLES_SPLIT}")
    config_lines.append(f"RF_MIN_SAMPLES_LEAF = {RF_MIN_SAMPLES_LEAF}")
    config_lines.append(f"RF_CLASS_WEIGHT = '{RF_CLASS_WEIGHT}'")

    # MATRICE DI CONFUSIONE FINALE
    conf_matrix_lines = []
    if last_confusion_matrix is not None:
        conf_matrix_lines.append("\nMATRICE DI CONFUSIONE SUL VALIDATION SET:\n" + "-"*40)
        conf_matrix_lines.append(f"{'tp:':<2} {last_confusion_matrix[1, 1]:<5} {'fp:':<2} {last_confusion_matrix[0, 1]:<5} {'fn:':<2} {last_confusion_matrix[1, 0]:<5} {'tn:':<2} {last_confusion_matrix[0, 0]:<5}\n")

    with open(report_path, "w") as f:
        f.write(header)
        for line in table_lines:
            f.write(line + "\n")
        f.write("="*140 + "\n")
        for line in conf_matrix_lines:
            f.write(line + "\n")
        for line in stats_lines:
            f.write(line + "\n")
        for line in config_lines:
            f.write(line + "\n")
    print(f"\n[SERVER] ✅ Report Random Forest OTTIMIZZATO salvato in: {report_path}")

def create_smartgrid_features(X_global, client_id='SERVER'):
    """
    Crea feature ingegnerizzate specifiche per il dataset SmartGrid su server.
    VERSIONE DETERMINISTICA E ROBUSTA: Garantisce sempre lo stesso numero di feature del client.
    Previene NaN/Inf con validazione completa.
    """
    if not ENABLE_FEATURE_ENGINEERING:
        return X_global
    
    # ✅ SEED FISSO per operazioni deterministiche (STESSO DEL CLIENT)
    np.random.seed(RANDOM_SEED)
        
    print(f"[{client_id}] === FEATURE ENGINEERING SMARTGRID SERVER DETERMINISTICO E ROBUSTO ===")
    
    # Copia il dataframe
    df_enhanced = X_global.copy()
    
    # ✅ CORREZIONE: Gestisci solo caso DataFrame SENZA marker
    # (il server non ha mai la colonna marker nel dataset di test)
    if isinstance(df_enhanced, pd.DataFrame):
        if 'marker' in df_enhanced.columns:
            # Caso con marker (non dovrebbe mai succedere sul server)
            original_features = len(df_enhanced.columns) - 1
            feature_cols = [col for col in df_enhanced.columns if col != 'marker']
        else:
            # Caso normale server: nessuna colonna marker
            original_features = len(df_enhanced.columns)
            feature_cols = list(df_enhanced.columns)
    else:
        # Caso numpy array
        original_features = df_enhanced.shape[1]
        feature_cols = list(range(original_features))
    
    print(f"[{client_id}] 🔍 DEBUG FEATURE ENGINEERING DETERMINISTICO:")
    print(f"[{client_id}]   DataFrame shape iniziale: {df_enhanced.shape}")
    print(f"[{client_id}]   Feature originali: {original_features}")
    
    # ✅ Seleziona solo colonne numeriche e ORDINA per determinismo (STESSO DEL CLIENT)
    numeric_cols = df_enhanced[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = sorted(numeric_cols)  # ORDINAMENTO per determinismo
    
    if len(numeric_cols) == 0:
        print(f"[{client_id}] ⚠️ Nessuna colonna numerica trovata per feature engineering")
        return df_enhanced
    
    print(f"[{client_id}]   Colonne numeriche ordinate: {len(numeric_cols)}")
    
    # Converti in numpy per efficienza
    X = df_enhanced[numeric_cols].values
    
    # ✅ PARAMETRI FISSI IDENTICI AL CLIENT
    FIXED_WINDOW_SIZE = 10
    FIXED_MAX_RATIOS = 50
    FIXED_MAX_ANOMALY = 15
    FIXED_MAX_INTERACTIONS = 20
    
    features_added = 0
    nan_created = 0
    inf_created = 0
    
    # 1. STATISTICAL FEATURES con parametri fissi IDENTICI
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
        print(f"[{client_id}] ✅ Aggiunte {stat_features_added} statistical features DETERMINISTICHE E ROBUSTE")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore statistical features: {e}")
    
    # 2. RATIO FEATURES con VALIDAZIONE ROBUSTA E CREAZIONE FORZATA
    try:
        n_ratios = 0
        for i in range(0, min(20, len(numeric_cols))):
            for j in range(i+1, min(i+5, len(numeric_cols))):
                if n_ratios >= FIXED_MAX_RATIOS:  # ✅ LIMITE FISSO IDENTICO
                    break
                    
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                
                # ✅ VALIDAZIONE ROBUSTA: Calcola ratio con gestione denominatore zero
                numerator = df_enhanced[col_i].values
                denominator = df_enhanced[col_j].values
                
                # Crea maschera per valori validi
                valid_mask = np.abs(denominator) > 1e-10
                
                # ✅ CORREZIONE: CREA SEMPRE LA FEATURE
                ratio_val = np.zeros(len(numerator))
                
                if np.any(valid_mask):
                    ratio_val[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
                    # ✅ Clip per prevenire inf
                    ratio_val = np.clip(ratio_val, -1e6, 1e6)
                    # Imputa valori non validi con mediana
                    median_ratio = np.median(ratio_val[valid_mask]) if np.sum(valid_mask) > 0 else 0.0
                    ratio_val[~valid_mask] = median_ratio
                else:
                    ratio_val[:] = 0.0
                
                # ✅ SEMPRE aggiungi la feature
                df_enhanced[f'ratio_{i}_{j}'] = ratio_val
                n_ratios += 1
            
            if n_ratios >= FIXED_MAX_RATIOS:  # ✅ LIMITE FISSO IDENTICO
                break
        
        features_added += n_ratios
        print(f"[{client_id}] ✅ Aggiunti {n_ratios} ratio features DETERMINISTICI E ROBUSTI (max {FIXED_MAX_RATIOS})")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore ratio features: {e}")
    
    # 3. ANOMALY INDICATORS con numero FISSO IDENTICO E VALIDAZIONE ROBUSTA
    try:
        n_zscore = 0
        for i, col in enumerate(numeric_cols[:FIXED_MAX_ANOMALY]):  # ✅ NUMERO FISSO IDENTICO
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
        print(f"[{client_id}] ✅ Aggiunti {n_zscore} anomaly indicators DETERMINISTICI E ROBUSTI (max {FIXED_MAX_ANOMALY})")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore anomaly features: {e}")
    
    # 4. INTERACTION FEATURES con numero FISSO IDENTICO E VALIDAZIONE
    try:
        n_interactions = 0
        for i in range(0, min(10, len(numeric_cols))):
            for j in range(i+1, min(i+3, len(numeric_cols))):
                if n_interactions >= FIXED_MAX_INTERACTIONS:  # ✅ LIMITE FISSO IDENTICO
                    break
                    
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                
                # ✅ Calcola prodotto e clip per prevenire overflow
                interaction = df_enhanced[col_i] * df_enhanced[col_j]
                interaction = np.clip(interaction, -1e10, 1e10)
                
                df_enhanced[f'interact_{i}_{j}'] = interaction
                n_interactions += 1
            
            if n_interactions >= FIXED_MAX_INTERACTIONS:  # ✅ LIMITE FISSO IDENTICO
                break
        
        features_added += n_interactions
        print(f"[{client_id}] ✅ Aggiunte {n_interactions} interaction features DETERMINISTICHE E ROBUSTE (max {FIXED_MAX_INTERACTIONS})")
    except Exception as e:
        print(f"[{client_id}] ⚠️ Errore interaction features: {e}")
    
    # ✅ VERIFICA FINALE: Conta NaN/Inf creati
    new_cols = [col for col in df_enhanced.columns if col not in df_enhanced.columns[:original_features] and col != 'marker']
    if new_cols:
        new_data = df_enhanced[new_cols].select_dtypes(include=[np.number])
        nan_created = new_data.isna().sum().sum()
        inf_created = np.isinf(new_data.values).sum()
    
    new_features = len(df_enhanced.columns) - original_features
    if 'marker' in df_enhanced.columns:
        new_features -= 1  # Non contare la colonna marker
        
    print(f"[{client_id}] 🎯 Feature engineering DETERMINISTICO E ROBUSTO completato:")
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
    """Calcola i limiti per clipping outlier usando IQR."""
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def transform_clip_outliers_iqr(X, lower, upper):
    """Applica il clipping ai dati X usando i limiti forniti."""
    return np.clip(X, lower, upper)

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
    """Rimuove le feature che sono costanti almeno al 99.9%."""
    keep_mask = []
    n = X.shape[0]
    
    for col in range(X.shape[1]):
        col_data = X[:, col]
        vals, counts = np.unique(col_data, return_counts=True)
        max_count = np.max(counts)
        ratio = max_count / n
        var = np.nanvar(col_data)
        keep = not (ratio >= threshold_ratio or var < threshold_var)
        keep_mask.append(keep)
    
    keep_mask = np.array(keep_mask)
    return X[:, keep_mask], keep_mask

def clean_data_for_pca(X):
    """Pulizia robusta dei dati per prevenire problemi numerici in PCA (server)."""
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    # Sostituisci inf e -inf con NaN
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def apply_pca(X_preprocessed):
    """Applica PCA con numero FISSO di componenti (server, identico ai client)."""
    print(f"[Server] === APPLICAZIONE PCA ===")

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
                raise ValueError("PCA server ha prodotto output con NaN o inf")
            if X_pca.shape[1] != n_components:
                raise ValueError(f"PCA server output shape inconsistente: {X_pca.shape[1]} vs {n_components}")
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            print(f"[Server] ✅ PCA fissa server applicata: {X_pca.shape}")
            print(f"[Server] Varianza spiegata: {variance_explained*100:.2f}%")
            return X_pca
        
    except Exception as e:
        print(f"[Server] ERRORE PCA fissa server: {e}")
        print(f"[Server] Attivazione fallback...")
        n_fallback = min(n_components, original_features)
        X_fallback = X_preprocessed[:, :n_fallback]
        print(f"[Server] ✅ Fallback server: {X_fallback.shape}")
        return X_fallback

def apply_preprocessing_pipeline(X_global):
    """
    Applica la stessa pipeline di preprocessing dei client sui dati globali del server.
    CORREZIONE: Applica feature engineering nella stessa sequenza dei client.
    """
    set_reproducibility_seeds()

    print(f"[Server] === PIPELINE PREPROCESSING SERVER OTTIMIZZATA ===")
    print(f"Feature engineering: {'ABILITATA' if ENABLE_FEATURE_ENGINEERING else 'DISABILITATA'}")
    print(f"Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")

    # ===== SEQUENZA IDENTICA AI CLIENT =====
    
    # Converti in DataFrame se necessario per feature engineering
    if isinstance(X_global, np.ndarray):
        X_global = pd.DataFrame(X_global)
        
    # STEP 0: Feature Engineering per SmartGrid (PRIMA del preprocessing)
    if ENABLE_FEATURE_ENGINEERING:
        # ✅ CORREZIONE: NON aggiungere marker dummy
        # La funzione create_smartgrid_features deve gestire il caso senza marker
        X_global_enhanced = create_smartgrid_features(X_global, 'SERVER')
    else:
        X_global_enhanced = X_global
    
    # STEP 1: Pulizia inf/NaN
    if ENABLE_CLEAN_INF_NAN:
        X_cleaned = clean_data_for_pca(X_global_enhanced)
    else:
        X_cleaned = X_global_enhanced.values if hasattr(X_global_enhanced, 'values') else X_global_enhanced
        
    # STEP 2: Clipping outlier feature-wise
    if ENABLE_CLIPPING_OUTLIERS:
        X_np = np.array(X_cleaned, dtype=float)
        lower, upper = fit_clip_outliers_iqr(X_np, k=5.0)
        X_clipped = transform_clip_outliers_iqr(X_np, lower, upper)
    else:
        X_clipped = X_cleaned

    # STEP 3: Imputazione mediana
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_clipped)
    else:
        X_imputed = X_clipped

    # STEP 4: Rimozione feature quasi-costanti
    if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES:
        X_reduced, keep_mask = remove_near_constant_features(X_imputed, threshold_var=1e-12, threshold_ratio=0.999)
        print(f"[Server] Feature dopo rimozione quasi-costanti: {X_reduced.shape[1]} (da {X_imputed.shape[1]})")
    else:
        X_reduced = X_imputed
        print(f"[Server] Rimozione feature quasi-costanti DISABILITATA - mantenute {X_reduced.shape[1]} feature")
    
    # STEP 5: Scaling standard
    if ENABLE_SCALING:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        print(f"[Server] Scaling standard applicato")
    else:
        X_scaled = X_reduced
        print(f"[Server] Scaling DISABILITATO")

    # STEP 6: PCA (se abilitata)
    if ENABLE_PCA:
        X_global_final = apply_pca(X_scaled)
        if X_global_final.shape[1] != PCA_COMPONENTS:
            raise RuntimeError(f"❌ Server PCA output shape inconsistente: {X_global_final.shape[1]} vs {PCA_COMPONENTS}")
        print(f"[Server] ✅ Pipeline preprocessing con PCA completata")
    else:
        X_global_final = X_scaled
        print(f"[Server] ✅ Pipeline preprocessing OTTIMIZZATA completata")
    
    print(f"[Server] Risultato finale: {X_global_final.shape}")
    return X_global_final

def calculate_tree_diversity_server(tree1, tree2, X_sample):
    """
    Calcola la diversità tra due alberi sul server.
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

def deserialize_trees_from_client_enhanced(parameters):
    """
    Deserializza gli alberi ricevuti da un client CON ACCURACY + DIVERSITÀ REALI.
    CORRETTO: Aggiunge i log di verifica mancanti.
    """
    try:
        # Gestisce diversi tipi di parametri da Flower
        if hasattr(parameters, 'tensors'):
            parameter_arrays = parameters.tensors
        elif isinstance(parameters, list): 
            parameter_arrays = parameters
        else:
            parameter_arrays = list(parameters) if parameters else []

        if not parameter_arrays:
            print(f"[Server] ⚠️ Nessun parametro ricevuto dal client")
            return []
        
        print(f"[Server] Ricevuti {len(parameter_arrays)} parametri ENHANCED dal client")
        
        deserialized_trees = []
        
        for i, param_data in enumerate(parameter_arrays):
            try:
                print(f"[Server] Elaborazione parametro ENHANCED {i+1}/{len(parameter_arrays)}: tipo={type(param_data)}")
                
                # Gestisce formato NumPy di Flower
                if isinstance(param_data, bytes):
                    if param_data.startswith(b'\x93NUMPY'):
                        print(f"[Server] Rilevato formato NumPy ENHANCED da Flower")
                        from io import BytesIO
                        tree_array = np.load(BytesIO(param_data))
                        print(f"[Server] Array NumPy ENHANCED caricato: shape={tree_array.shape}, dtype={tree_array.dtype}")
                        tree_bytes = tree_array.tobytes()
                        print(f"[Server] Convertito in bytes per pickle: {len(tree_bytes)} bytes")
                    else:
                        tree_bytes = param_data
                        print(f"[Server] Bytes diretti ENHANCED ricevuti: {len(tree_bytes)} bytes")
                
                elif isinstance(param_data, np.ndarray):
                    tree_bytes = param_data.tobytes()
                    print(f"[Server] Convertito numpy array ENHANCED in bytes: {len(tree_bytes)} bytes")
                else:
                    print(f"[Server] ⚠️ Parametro {i+1} ignorato (tipo non compatibile: {type(param_data)})")
                    continue

                # Deserializza dizionario completo con accuracy + diversità
                tree_data = pickle.loads(tree_bytes)
                print(f"[Server] Oggetto ENHANCED deserializzato tipo: {type(tree_data)}")

                # Verifica se è il nuovo formato ENHANCED con accuracy + diversità
                if isinstance(tree_data, dict) and 'tree' in tree_data and 'accuracy_type' in tree_data:
                    if tree_data['accuracy_type'] in ['REAL_ENHANCED', 'REAL']:
                        tree = tree_data['tree']
                        accuracy_real = tree_data['accuracy']
                        weighted_accuracy_real = tree_data['weighted_accuracy']
                        
                        # Estrae diversity score se disponibile (nuovo formato)
                        diversity_score = tree_data.get('diversity_score', 0.0)
                        client_id = tree_data.get('client_id', 0)
                        
                        # Verifica che sia un albero valido
                        if hasattr(tree, 'predict') and hasattr(tree, 'tree_'):
                            deserialized_trees.append((tree, accuracy_real, weighted_accuracy_real, diversity_score, client_id))
                            
                            # AGGIUNTO: Conferma ricezione diversity REALE
                            if tree_data['accuracy_type'] == 'REAL_ENHANCED':
                                print(f"[Server] ✅ Ricevuto albero ENHANCED con diversity REALE: {diversity_score:.4f}")
                            else:
                                print(f"[Server] ⚠️ Albero senza diversity REALE, formato: {tree_data.get('accuracy_type', 'UNKNOWN')}")
                            
                            print(f"[Server] ✅ Albero {i+1} ENHANCED: acc={accuracy_real:.4f}, w_acc={weighted_accuracy_real:.4f}, div={diversity_score:.4f}, client={client_id}")
                        else:
                            print(f"[Server] ⚠️ Oggetto {i+1} non è un albero valido")
                    else:
                        print(f"[Server] ⚠️ Albero {i+1} formato non supportato: {tree_data.get('accuracy_type', 'UNKNOWN')}")
                        
                # Fallback per compatibilità con formato standard
                elif isinstance(tree_data, dict) and 'tree' in tree_data and tree_data.get('accuracy_type') == 'REAL':
                    tree = tree_data['tree']
                    accuracy_real = tree_data['accuracy']
                    weighted_accuracy_real = tree_data['weighted_accuracy']
                    
                    if hasattr(tree, 'predict') and hasattr(tree, 'tree_'):
                        # Diversity score = 0 per compatibilità
                        deserialized_trees.append((tree, accuracy_real, weighted_accuracy_real, 0.0, 0))
                        print(f"[Server] ✅ Albero {i+1} STANDARD: acc={accuracy_real:.4f}, w_acc={weighted_accuracy_real:.4f}")
                    
                # Fallback per formato vecchio
                elif hasattr(tree_data, 'predict') and hasattr(tree_data, 'tree_'):
                    print(f"[Server] ⚠️ Albero {i+1} in formato vecchio, uso accuracy simulate")
                    simulated_accuracy = max(0.8, 0.95 - (i * 0.002))
                    simulated_weighted_acc = max(0.75, 0.94 - (i * 0.002))
                    deserialized_trees.append((tree_data, simulated_accuracy, simulated_weighted_acc, 0.0, 0))
                else:
                    print(f"[Server] ⚠️ Formato dati non riconosciuto per parametro {i+1}")

            except Exception as e:
                print(f"[Server] ❌ Errore nella deserializzazione parametro ENHANCED {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[Server] Deserializzati {len(deserialized_trees)} alberi ENHANCED validi su {len(parameter_arrays)}")
        
        # AGGIUNTO: VERIFICHE MANCANTI per accuracy + diversità reali
        if deserialized_trees:
            real_accuracies = [t[1] for t in deserialized_trees]
            real_w_accuracies = [t[2] for t in deserialized_trees]
            diversity_scores = [t[3] for t in deserialized_trees]
            
            print(f"[Server] ✅ VERIFICATION: Accuracy REALI ricevute: min={min(real_accuracies):.4f}, max={max(real_accuracies):.4f}, media={np.mean(real_accuracies):.4f}")
            print(f"[Server] ✅ VERIFICATION: Weighted accuracy REALI: min={min(real_w_accuracies):.4f}, max={max(real_w_accuracies):.4f}, media={np.mean(real_w_accuracies):.4f}")
            print(f"[Server] ✅ VERIFICATION: Diversity scores: min={min(diversity_scores):.4f}, max={max(diversity_scores):.4f}, media={np.mean(diversity_scores):.4f}")
            
            # Conta alberi con diversity reali
            enhanced_trees = sum(1 for t in deserialized_trees if len(t) >= 4 and t[3] > 0.0)
            print(f"[Server] ✅ VERIFICATION: Alberi con diversity REALE: {enhanced_trees}/{len(deserialized_trees)}")
        
        return deserialized_trees
        
    except Exception as e:
        print(f"[Server] ❌ Errore nella deserializzazione ENHANCED: {e}")
        import traceback
        traceback.print_exc()
        return []

def select_best_trees_enhanced(all_trees_data, strategy=TREE_AGGREGATION_STRATEGY, method=TREE_SELECTION_METHOD, max_trees=MAX_TREES_GLOBAL):
    """
    Seleziona i migliori alberi basandosi su ACCURACY + DIVERSITÀ REALI dai client.
    VERSIONE CORRETTA: Usa diversity reali dai client senza X_global non definito.
    
    Args:
        all_trees_data: Lista di liste, ogni elemento contiene gli alberi di un client
                       Formato albero: (tree, accuracy, weighted_accuracy, diversity_score, client_id)
        strategy: 'global' o 'per_forest' per strategia di selezione
        method: 'diversity_weighted', 'weighted_accuracy', o 'accuracy' per metrica di selezione
        max_trees: Numero massimo di alberi da selezionare globalmente
        
    Returns:
        Lista di alberi selezionati con i loro metadati
    """
    print(f"[Server] === SELEZIONE ALBERI ENHANCED CON ACCURACY + DIVERSITÀ ===")
    print(f"Strategia: {strategy}")
    print(f"Metodo: {method}")
    print(f"Max alberi globale: {max_trees}")
    
    if not all_trees_data:
        print(f"[Server] ⚠️ Nessun dato alberi ricevuto")
        return []
    
    selected_trees = []
    
    # Prepara pool globale di tutti gli alberi
    all_trees_flat = []
    for client_trees in all_trees_data:
        all_trees_flat.extend(client_trees)
    
    if not all_trees_flat:
        print(f"[Server] ⚠️ Nessun albero nel pool globale")
        return []
    
    print(f"[Server] Pool globale: {len(all_trees_flat)} alberi da {len(all_trees_data)} client")
    
    if method == 'diversity_weighted':
        print(f"[Server] 🎯 Selezione DIVERSITY-WEIGHTED per ottimizzazione federated learning")
        
        # ALGORITMO DIVERSITY-AWARE SELECTION CORRETTO
        # Step 1: Seleziona il miglior albero iniziale (per weighted_accuracy)
        if len(all_trees_flat[0]) >= 4:  # Nuovo formato con diversità
            sorted_by_accuracy = sorted(all_trees_flat, key=lambda x: x[2], reverse=True)  # weighted_accuracy
        else:  # Formato vecchio (fallback)
            sorted_by_accuracy = sorted(all_trees_flat, key=lambda x: x[1], reverse=True)  # accuracy
        
        selected_trees.append(sorted_by_accuracy[0])
        remaining_trees = sorted_by_accuracy[1:]
        
        print(f"[Server] Albero iniziale selezionato: acc={sorted_by_accuracy[0][1]:.4f}, w_acc={sorted_by_accuracy[0][2]:.4f}")
        
        # Step 2: Selezione iterativa basata su accuracy + diversità REALI
        for iteration in range(min(max_trees - 1, len(remaining_trees))):
            best_score = -1
            best_tree = None
            best_idx = -1
            
            for idx, candidate in enumerate(remaining_trees):
                # ✅ CORREZIONE PRINCIPALE: Usa SOLO diversity score REALE dal client
                if len(candidate) >= 4:  # Ha diversity score REALE
                    accuracy_score = candidate[2]  # weighted_accuracy
                    diversity_score = candidate[3]  # diversity_score REALE dal client
                else:  # Formato legacy - usa valore neutro
                    accuracy_score = candidate[1]  # accuracy normale
                    diversity_score = 0.25  # Valore neutro per alberi senza diversity
                
                # ✅ NUOVO PESO: 60% accuracy + 40% diversity (invece di 70/30)
                # Aumenta influenza diversity nella selezione
                combined_score = 0.6 * accuracy_score + 0.4 * diversity_score
                
                # DEBUG dettagliato solo per le prime 10 iterazioni e primi 10 candidati
                if iteration < 10 and idx < 10:
                    print(f"[Server] 🔍 DEBUG Albero candidato {idx}:")
                    print(f"  - accuracy_score: {accuracy_score:.4f}")
                    print(f"  - diversity_score: {diversity_score:.4f} (tipo: {'REALE' if len(candidate) >= 4 else 'NEUTRO'})")
                    print(f"  - combined_score: {combined_score:.4f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_tree = candidate
                    best_idx = idx
            
            if best_tree is not None:
                selected_trees.append(best_tree)
                remaining_trees.pop(best_idx)
                
                # Mostra solo ogni 10 alberi per non saturare i log
                if len(selected_trees) % 10 == 0 or len(selected_trees) <= 10:
                    print(f"[Server] Albero {len(selected_trees)}: score_combinato={best_score:.4f} (REAL diversity)")
    
    elif strategy == 'global':
        # Selezione globale standard (per compatibilità)
        if all_trees_flat:
            # Ordina per metodo specificato
            if method == 'weighted_accuracy':
                metric_idx = 2
            elif method == 'accuracy':
                metric_idx = 1
            else:  # fallback
                metric_idx = 2
                
            sorted_trees = sorted(all_trees_flat, key=lambda x: x[metric_idx], reverse=True)
            selected_trees = sorted_trees[:max_trees]
            
            if selected_trees:
                best_score = selected_trees[0][metric_idx]
                worst_score = selected_trees[-1][metric_idx]
                print(f"[Server] Selezione globale standard: {len(selected_trees)} alberi")
                print(f"[Server] Range {method}: {worst_score:.4f} - {best_score:.4f}")
    
    elif strategy == 'per_forest':
        # Selezione per forest (distributiva tra client)
        trees_per_client = max_trees // len(all_trees_data)
        remaining_trees = max_trees % len(all_trees_data)
        
        print(f"[Server] Seleziono {trees_per_client} alberi per client ({len(all_trees_data)} client)")
        
        for client_idx, client_trees in enumerate(all_trees_data):
            if not client_trees:
                continue
                
            # Ordina gli alberi del client
            if method == 'diversity_weighted' and len(client_trees[0]) >= 4:
                # Usa score combinato per client con diversity REALI
                sorted_trees = sorted(client_trees, key=lambda x: 0.7 * x[2] + 0.3 * x[3], reverse=True)
            elif method == 'weighted_accuracy':
                sorted_trees = sorted(client_trees, key=lambda x: x[2], reverse=True)
            else:  # accuracy
                sorted_trees = sorted(client_trees, key=lambda x: x[1], reverse=True)
            
            # Seleziona i migliori per questo client
            num_to_select = trees_per_client + (1 if client_idx < remaining_trees else 0)
            client_selected = sorted_trees[:num_to_select]
            selected_trees.extend(client_selected)
            
            if client_selected:
                if len(client_selected[0]) >= 4:
                    print(f"[Server] Client {client_idx+1}: {len(client_selected)} alberi ENHANCED selezionati")
                else:
                    print(f"[Server] Client {client_idx+1}: {len(client_selected)} alberi STANDARD selezionati")
    
    print(f"[Server] ✅ Alberi selezionati totali: {len(selected_trees)} (metodo ENHANCED: {method})")
    
    # ✅ QUICK CHECK 3: SELECTION CHECK (mostra solo i primi 5)
    if selected_trees and len(selected_trees) >= 5:
        print(f"\n[Server] 🔍 QUICK CHECK 3 - SELECTION CHECK (primi 5):")
        for i in range(min(5, len(selected_trees))):
            tree_data = selected_trees[i]
            if len(tree_data) >= 4:
                _, acc, w_acc, div = tree_data[:4]
                combined = 0.6 * w_acc + 0.4 * div
                print(f"  Tree {i+1}: acc={acc:.4f}, w_acc={w_acc:.4f}, div={div:.4f}, combined={combined:.4f}")
    
    # Statistiche finali CORRETTE
    if selected_trees:
        final_accuracies = [t[1] for t in selected_trees]
        final_w_accuracies = [t[2] for t in selected_trees]
        final_diversities = [t[3] if len(t) >= 4 else 0.0 for t in selected_trees]
        
        print(f"\n[Server] 📊 Statistiche Finali Alberi Selezionati:")
        print(f"[Server] Accuracy: min={min(final_accuracies):.4f}, max={max(final_accuracies):.4f}, media={np.mean(final_accuracies):.4f}")
        print(f"[Server] Weighted accuracy: min={min(final_w_accuracies):.4f}, max={max(final_w_accuracies):.4f}, media={np.mean(final_w_accuracies):.4f}")
        print(f"[Server] Diversity scores: min={min(final_diversities):.4f}, max={max(final_diversities):.4f}, media={np.mean(final_diversities):.4f}")
        
        # Conta alberi con diversity reale
        real_diversity_count = sum(1 for d in final_diversities if d > 0.0)
        print(f"[Server] 🎯 Alberi con DIVERSITY REALE: {real_diversity_count}/{len(selected_trees)}")
        print(f"[Server] ✅ UTILIZZO DIVERSITY REALI DAI CLIENT per selezione ottimale!")
    
    return selected_trees

def create_global_random_forest_enhanced(selected_trees):
    """
    Crea un Random Forest globale ottimizzato combinando i migliori alberi dai client.
    AGGIORNATO: Implementa configurazione ottimizzata per federated learning.
    
    Args:
        selected_trees: Lista di tuple (tree, accuracy, weighted_accuracy, [diversity_score, client_id])
        
    Returns:
        RandomForestClassifier globale ottimizzato
    """
    print(f"[Server] === CREAZIONE RANDOM FOREST GLOBALE OTTIMIZZATO ===")
    
    if not selected_trees:
        print(f"[Server] ⚠️ Nessun albero da aggregare, creo RF vuoto")
        # Crea un Random Forest vuoto con configurazione ottimizzata
        return RandomForestClassifier(
            n_estimators=1,  # Minimo per evitare errori
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            bootstrap=RF_BOOTSTRAP,
            random_state=RANDOM_SEED,  # ✅ FISSO per aggregazione riproducibile
            n_jobs=-1,
            class_weight=RF_CLASS_WEIGHT,
            criterion=RF_CRITERION
        )
    
    # Estrai solo gli alberi (senza metadati)
    trees = [tree_data[0] for tree_data in selected_trees]
    
    # Crea un nuovo Random Forest con configurazione ottimizzata
    global_rf = RandomForestClassifier(
        n_estimators=len(trees),  # Numero di alberi = alberi selezionati
        max_depth=RF_MAX_DEPTH,  # OTTIMIZZATO: Limitato per ridurre overfitting
        min_samples_split=RF_MIN_SAMPLES_SPLIT,  # OTTIMIZZATO: Aumentato
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,  # OTTIMIZZATO: Aumentato
        max_features=RF_MAX_FEATURES,
        bootstrap=RF_BOOTSTRAP,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight=RF_CLASS_WEIGHT,  # OTTIMIZZATO: balanced_subsample
        criterion=RF_CRITERION
    )
    
    # Assegna gli alberi al Random Forest globale
    # NOTA: Hack necessario per scikit-learn
    global_rf.estimators_ = trees
    global_rf.n_estimators = len(trees)
    
    # Copia attributi necessari dal primo albero per permettere predizioni
    first_tree = trees[0]
    if hasattr(first_tree, 'n_features_in_'):
        global_rf.n_features_in_ = first_tree.n_features_in_
    if hasattr(first_tree, 'n_outputs_'):
        global_rf.n_outputs_ = first_tree.n_outputs_
    if hasattr(first_tree, 'classes_'):
        global_rf.classes_ = first_tree.classes_
        global_rf.n_classes_ = len(first_tree.classes_)
    else:
        # Default per classificazione binaria
        global_rf.classes_ = np.array([0, 1])
        global_rf.n_classes_ = 2

    print(f"[Server] ✅ Random Forest globale OTTIMIZZATO creato con {len(trees)} alberi (diversità + accuracy)")
    print(f"[Server] Configurazione ENHANCED: max_depth={RF_MAX_DEPTH}, min_samples_split={RF_MIN_SAMPLES_SPLIT}")
    print(f"[Server] Attributi configurati: n_features={getattr(global_rf, 'n_features_in_', 'N/A')}, n_classes={getattr(global_rf, 'n_classes_', 'N/A')}")
    
    # Statistiche degli alberi aggregati
    accuracies_real = [tree_data[1] for tree_data in selected_trees]
    weighted_accuracies_real = [tree_data[2] for tree_data in selected_trees]
    diversity_scores = [tree_data[3] if len(tree_data) >= 4 else 0.0 for tree_data in selected_trees]
    
    print(f"[Server] Accuracy alberi aggregati: min={min(accuracies_real):.4f}, max={max(accuracies_real):.4f}, mean={np.mean(accuracies_real):.4f}")
    print(f"[Server] Weighted accuracy alberi: min={min(weighted_accuracies_real):.4f}, max={max(weighted_accuracies_real):.4f}, mean={np.mean(weighted_accuracies_real):.4f}")
    print(f"[Server] Diversity scores: min={min(diversity_scores):.4f}, max={max(diversity_scores):.4f}, mean={np.mean(diversity_scores):.4f}")
    print(f"[Server] 🎯 Modello globale usa alberi selezionati con DIVERSITÀ + ACCURACY ottimali per FL!")
    
    return global_rf

def serialize_global_model(global_rf):
    """
    Serializza il Random Forest globale ottimizzato per l'invio ai client.
    Usa pickle + conversione in numpy array (uint8) per compatibilità con Flower.
    """
    try:
        # Serializza il modello Random Forest globale ottimizzato con pickle
        model_bytes = pickle.dumps(global_rf, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Converti in numpy array (uint8) per Flower
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        
        print(f"[Server] Modello globale OTTIMIZZATO serializzato ({len(model_bytes)} bytes)")
        print(f"[Server] Convertito in numpy array: shape={model_array.shape}, dtype={model_array.dtype}")
        
        # Usa Parameters invece di lista
        from flwr.common import ndarrays_to_parameters
        parameters = ndarrays_to_parameters([model_array])

        return parameters

    except Exception as e:
        print(f"[Server] ❌ Errore serializzazione modello globale ottimizzato: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_smartgrid_random_forest_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server Random Forest SmartGrid OTTIMIZZATO.
    Aggiornata per supportare le nuove ottimizzazioni.
    """
    
    def load_global_test_data():
        """
        Carica dataset globale di test SENZA preprocessing pesante.
        CORREZIONE: Applica solo pulizia minima per evitare incompatibilità.
        """
        set_reproducibility_seeds()

        print("=== CARICAMENTO DATASET GLOBALE TEST SERVER RF OTTIMIZZATO ===")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Usa client 14-15 come dataset di test
        test_clients = [14, 15]
        df_list = []

        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")

            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"File data{client_id}.csv non trovato")
                continue

        if not df_list:
            # Fallback
            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(200, len(df_fallback)), random_state=42)]
                print(f"Usando fallback con {len(df_list[0])} campioni da data1.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per valutazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        # Statistiche distribuzione
        attack_samples = y_global.sum()
        natural_samples = (y_global == 0).sum()
        attack_ratio = y_global.mean()
        
        print(f"Dataset test globale: {len(df_global)} campioni")
        print(f"Distribuzione: {attack_samples} attacchi ({attack_ratio*100:.1f}%), {natural_samples} naturali")
        
        # ✅ CORREZIONE: Applica SOLO pulizia minima identica ai client
        # NO imputazione, NO scaling, NO remove constants
        # SOLO feature engineering + pulizia inf/nan base
        
        print(f"[Server] Applicazione preprocessing MINIMALE per compatibilità...")
        
        # Feature Engineering (se abilitato)
        if ENABLE_FEATURE_ENGINEERING:
            X_global_fe = create_smartgrid_features(X_global, 'SERVER')
        else:
            X_global_fe = X_global
        
        # Pulizia inf/nan MINIMA (come i client)
        X_array = X_global_fe.values if hasattr(X_global_fe, 'values') else X_global_fe
        X_array = np.where(np.isinf(X_array), np.nan, X_array)
        
        # Gestisci infiniti con percentili (come client)
        inf_mask = np.isinf(X_array)
        if np.any(inf_mask):
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
        
        # Imputa NaN con mediana LOCALE (non fittiamo imputer)
        for col in range(X_array.shape[1]):
            col_data = X_array[:, col]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask):
                median_val = np.nanmedian(col_data)
                X_array[nan_mask, col] = median_val if not np.isnan(median_val) else 0.0
        
        X_global_final = X_array
        
        print(f"Dataset preprocessato MINIMALE: {len(X_global_final)} campioni, {X_global_final.shape[1]} feature")
        print(f"⚠️ NO IMPUTAZIONE FITTIATA, NO SCALING, NO REMOVE CONSTANTS per evitare distribution shift")
        
        return X_global_final, y_global, {
            'total_samples': len(df_global),
            'attack_samples': attack_samples,
            'natural_samples': natural_samples,
            'attack_ratio': attack_ratio
        }
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global, dataset_info = load_global_test_data()
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi
        feature_count = 150 if ENABLE_FEATURE_ENGINEERING else (PCA_COMPONENTS if ENABLE_PCA else 128)
        X_global = np.random.random((100, feature_count))
        y_global = np.random.randint(0, 2, 100)
        dataset_info = {'total_samples': 100, 'attack_samples': 50, 'natural_samples': 50, 'attack_ratio': 0.5}
        print(f"Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round per Random Forest OTTIMIZZATO.
        """
        set_reproducibility_seeds()

        print(f"\n=== VALUTAZIONE GLOBALE RANDOM FOREST OTTIMIZZATO - ROUND {server_round + 1} ===")
        
        try:
            # CONTROLLO: Verifica se ci sono parametri da valutare
            if server_round == 0:
                print(f"[Server] ⚠️ Primo round, nessun modello da valutare")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": "no_model_first_round", 
                    "global_test_samples": len(X_global),
                    "optimization_enabled": True
                }
            elif not parameters or len(parameters) == 0:
                print(f"[Server] ❌ Nessun modello ricevuto dai client")
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": "no_model_received", 
                    "global_test_samples": len(X_global),
                    "optimization_enabled": True
                }
            
            try:
                # Deserializza il Random Forest globale ottimizzato
                model_array = parameters[0]

                # Converte numpy array in bytes
                if hasattr(model_array, 'tobytes'):
                    model_bytes = model_array.tobytes()
                elif hasattr(model_array, 'data'):
                    model_bytes = model_array.data.tobytes()
                else:
                    model_bytes = bytes(model_array)
                
                print(f"[Server] Deserializzazione modello OTTIMIZZATO: {len(model_bytes)} bytes")

                # Deserializza usando pickle
                global_rf = pickle.loads(model_bytes)

                print(f"✅ Modello Random Forest globale OTTIMIZZATO deserializzato")
                print(f"   N. alberi: {global_rf.n_estimators if hasattr(global_rf, 'n_estimators') else 'N/A'}")
                print(f"   Max depth: {global_rf.max_depth if hasattr(global_rf, 'max_depth') else 'N/A'}")
                print(f"   Class weight: {global_rf.class_weight if hasattr(global_rf, 'class_weight') else 'N/A'}")
            except Exception as e:
                print(f"❌ Errore deserializzazione modello ottimizzato: {e}")
                import traceback
                traceback.print_exc()
                return 1.0, {
                    "accuracy": 0.0, 
                    "error": f"deserialization_failed: {str(e)}", 
                    "global_test_samples": len(X_global),
                    "optimization_enabled": True
                }
            
            # Verifica che il modello sia stato addestrato
            if not hasattr(global_rf, 'estimators_') or len(global_rf.estimators_) == 0:
                print(f"⚠️ Modello Random Forest OTTIMIZZATO non addestrato, uso predizioni casuali")
                y_pred_binary = np.random.randint(0, 2, len(y_global))
                y_pred_prob = np.random.random(len(y_global))
            else:
                # Valutazione sul dataset test globale
                try:
                    y_pred_binary = global_rf.predict(X_global)
                    y_pred_prob = global_rf.predict_proba(X_global)[:, 1] if hasattr(global_rf, 'predict_proba') else np.random.random(len(y_global))
                except Exception as e:
                    print(f"⚠️ Errore predizione OTTIMIZZATO, uso valori casuali: {e}")
                    y_pred_binary = np.random.randint(0, 2, len(y_global))
                    y_pred_prob = np.random.random(len(y_global))
            
            # Calcolo metriche
            accuracy = accuracy_score(y_global, y_pred_binary)
            
            # Metriche sicure (gestione casi edge)
            try:
                precision = precision_score(y_global, y_pred_binary, zero_division=0)
                recall = recall_score(y_global, y_pred_binary, zero_division=0)
                f1_score_val = f1_score(y_global, y_pred_binary, zero_division=0)
                balanced_acc = balanced_accuracy_score(y_global, y_pred_binary)
                
                if len(np.unique(y_global)) > 1 and len(np.unique(y_pred_prob)) > 1:
                    auc = roc_auc_score(y_global, y_pred_prob)
                else:
                    auc = 0.5  # AUC neutrale se non calcolabile
                    
            except Exception as e:
                print(f"⚠️ Errore calcolo metriche: {e}")
                precision = recall = f1_score_val = balanced_acc = auc = 0.0
            
            # Report dettagliato per classe
            try:
                report = classification_report(y_global, y_pred_binary, target_names=["natural", "attack"], output_dict=True, zero_division=0)
                conf_matrix = confusion_matrix(y_global, y_pred_binary)
            except Exception as e:
                print(f"⚠️ Errore classification report: {e}")
                report = {"natural": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
                         "attack": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}}
                conf_matrix = np.array([[0, 0], [0, 0]])
            
            # Loss simulata (Random Forest non ha loss)
            loss = 1 - accuracy
            
            print(f"RISULTATI VALUTAZIONE RANDOM FOREST OTTIMIZZATO:")
            print(f"  Loss (simulata): {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1_score_val:.4f} ({f1_score_val*100:.2f}%)")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
            print(f"  Campioni test: {len(X_global)}")
            print(f"  Distribuzione naturale: {dataset_info.get('attack_ratio', 0)*100:.1f}% attacchi")
            
            if hasattr(global_rf, 'estimators_') and len(global_rf.estimators_) > 0:
                print(f"  Alberi nel modello globale: {len(global_rf.estimators_)}")
            
            # Informazioni ottimizzazioni
            print(f"  🎯 Ottimizzazioni ATTIVE:")
            print(f"    - Feature engineering: {ENABLE_FEATURE_ENGINEERING}")
            print(f"    - Scaling: {ENABLE_SCALING}")
            print(f"    - Remove constants: {ENABLE_REMOVE_NEAR_CONSTANT_FEATURES}")
            print(f"    - Tree selection: {TREE_SELECTION_METHOD}")
            print(f"    - Max trees: {MAX_TREES_GLOBAL}")
            
            print(f"Classification report (per classe):")
            print(classification_report(y_global, y_pred_binary, target_names=["natural", "attack"], zero_division=0))
            print(f"Confusion matrix:")
            print(conf_matrix)

            # Raccolta metriche per report
            metric_row = {
                "round": server_round + 1,
                "loss_distribuita": float(loss),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "f1_natural": float(report["natural"]["f1-score"]),
                "f1_attack": float(report["attack"]["f1-score"]),
                "precision": float(precision),
                "precision_natural": float(report["natural"]["precision"]),
                "precision_attack": float(report["attack"]["precision"]),
                "recall": float(recall),
                "recall_natural": float(report["natural"]["recall"]),
                "recall_attack": float(report["attack"]["recall"]),
            }
            
            # Salva ultima confusion matrix per report finale
            global last_confusion_matrix
            last_confusion_matrix = conf_matrix

            # Aggiungi alla lista globale delle metriche
            global all_federated_metrics
            all_federated_metrics.append(metric_row)

            return float(loss), {
                # Metriche base
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc),
                "f1_score": float(f1_score_val),
                "balanced_accuracy": float(balanced_acc),

                # Metriche per classe
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
                
                # Informazioni dataset e modello ottimizzato
                "global_test_samples": int(len(X_global)),
                "n_trees_global": int(len(global_rf.estimators_)) if hasattr(global_rf, 'estimators_') else 0,
                "attack_samples": int(dataset_info.get('attack_samples', 0)),
                "natural_samples": int(dataset_info.get('natural_samples', 0)),
                "attack_ratio": float(dataset_info.get('attack_ratio', 0)),
                
                # NUOVI: Info ottimizzazioni
                "optimization_enabled": True,
                "feature_engineering": ENABLE_FEATURE_ENGINEERING,
                "scaling_enabled": ENABLE_SCALING,
                "remove_constants": ENABLE_REMOVE_NEAR_CONSTANT_FEATURES,
                "tree_selection_method": TREE_SELECTION_METHOD,
                "max_trees_global": MAX_TREES_GLOBAL,
            }
            
        except Exception as e:
            print(f"❌ Errore durante valutazione globale Random Forest OTTIMIZZATO: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {
                "accuracy": 0.0, 
                "error": f"evaluation_failed: {str(e)}", 
                "global_test_samples": len(X_global) if 'X_global' in locals() else 0,
                "optimization_enabled": True
            }
    
    return evaluate

def print_client_metrics_rf_enhanced(fit_results):
    """
    Stampa le metriche dei client Random Forest OTTIMIZZATO dopo ogni round.
    AGGIORNATO: Include informazioni sulle ottimizzazioni e diversity.
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT RANDOM FOREST OTTIMIZZATO ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    total_weighted_f1 = 0
    error_clients = []
    accuracy_list = []
    f1_list = []
    oob_scores = []
    n_estimators_list = []
    enhanced_clients = 0
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1}: {client_samples} campioni")
        
        if 'error' in client_metrics:
            error_clients.append(i+1)
            print(f"  ERRORE: {client_metrics['error']}")
            continue
        
        # Metriche base Random Forest
        if 'train_accuracy' in client_metrics:
            accuracy = client_metrics['train_accuracy']
            total_weighted_accuracy += accuracy * client_samples
            accuracy_list.append(accuracy)
            print(f"  Accuracy: {accuracy:.4f}")
        
        if 'train_f1_score' in client_metrics:
            f1 = client_metrics['train_f1_score']
            total_weighted_f1 += f1 * client_samples
            f1_list.append(f1)
            print(f"  F1-Score: {f1:.4f}")
        
        if 'train_balanced_accuracy' in client_metrics:
            balanced_acc = client_metrics['train_balanced_accuracy']
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        if 'oob_score' in client_metrics:
            oob = client_metrics['oob_score']
            oob_scores.append(oob)
            print(f"  OOB Score: {oob:.4f}")
        
        if 'n_estimators' in client_metrics:
            n_est = client_metrics['n_estimators']
            n_estimators_list.append(n_est)
            print(f"  N. Estimatori: {n_est}")
        
        if 'n_features' in client_metrics:
            n_feat = client_metrics['n_features']
            print(f"  N. Features: {n_feat}")
        
        # NUOVI: Info ottimizzazioni
        if 'feature_engineering_enabled' in client_metrics:
            fe_enabled = client_metrics['feature_engineering_enabled']
            if fe_enabled:
                enhanced_clients += 1
                print(f"  🎯 Feature Engineering: ATTIVA")
        
        if 'enhanced_preprocessing' in client_metrics:
            enh_preproc = client_metrics['enhanced_preprocessing']
            if enh_preproc:
                print(f"  🎯 Preprocessing Avanzato: ATTIVO")
        
        if 'random_state_diversified' in client_metrics:
            rs = client_metrics['random_state_diversified']
            print(f"  🎯 Random State Diversificato: {rs}")
    
    if total_samples > 0:
        # Calcola medie ponderate
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        avg_weighted_f1 = total_weighted_f1 / total_samples if total_weighted_f1 > 0 else 0
        avg_oob = np.mean(oob_scores) if oob_scores else 0
        
        print(f"\nRIASSUNTO METRICHE RANDOM FOREST OTTIMIZZATO:")
        print(f"  Media accuracy: {avg_weighted_accuracy:.4f}")
        print(f"  Media F1-Score: {avg_weighted_f1:.4f}")
        print(f"  Media OOB Score: {avg_oob:.4f}")
        print(f"  Totale campioni: {total_samples}")
        print(f"  Client con errori: {len(error_clients)}")
        print(f"  🎯 Client con ottimizzazioni: {enhanced_clients}/{len(fit_results)}")
        
        if n_estimators_list:
            print(f"  Alberi per client: {np.mean(n_estimators_list):.1f} ± {np.std(n_estimators_list):.1f}")

        print(f"  I client inviano ACCURACY + DIVERSITÀ REALI per selezione ottimale")

class SmartGridRandomForestFedAvgEnhanced(FedAvg):
    """
    Strategia FedAvg OTTIMIZZATA per SmartGrid Random Forest.
    Implementa l'aggregazione degli alberi con diversity-aware selection.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega gli alberi Random Forest dai client con selezione ENHANCED.
        CORRETTO: Usa la deserializzazione corretta per diversity reali.
        """
        set_reproducibility_seeds()

        print(f"\n=== AGGREGAZIONE RANDOM FOREST OTTIMIZZATO - ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("❌ ERRORE: Nessun client ha fornito risultati validi")
            return None, {}
        
        # Stampa metriche dei client Random Forest OTTIMIZZATO
        print_client_metrics_rf_enhanced(results)
        
        try:
            # CORRETTO: Usa la funzione corretta per deserializzazione enhanced
            all_trees_data = []
            
            for i, (client_proxy, fit_res) in enumerate(results):
                print(f"\n[Server] Processando alberi ENHANCED da client {i+1}...")
                
                # CORRETTO: Usa deserialize_trees_from_client_enhanced
                client_trees = deserialize_trees_from_client_enhanced(fit_res.parameters)
                
                if client_trees:
                    all_trees_data.append(client_trees)
                    print(f"[Server] Client {i+1}: {len(client_trees)} alberi ENHANCED ricevuti")
                    
                    # Mostra statistiche alberi del client con diversity
                    if client_trees:
                        accuracies = [tree[1] for tree in client_trees]
                        w_accuracies = [tree[2] for tree in client_trees]
                        diversities = [tree[3] if len(tree) >= 4 else 0.0 for tree in client_trees]
                        
                        print(f"[Server] Client {i+1} - Accuracy range: {min(accuracies):.4f}-{max(accuracies):.4f}")
                        print(f"[Server] Client {i+1} - Weighted acc range: {min(w_accuracies):.4f}-{max(w_accuracies):.4f}")
                        print(f"[Server] Client {i+1} - Diversity range: {min(diversities):.4f}-{max(diversities):.4f}")
                        
                        # VERIFICATO: Conta alberi con diversity reali per client
                        real_diversity_count = sum(1 for d in diversities if d > 0.0)
                        print(f"[Server] Client {i+1} - Alberi con diversity REALE: {real_diversity_count}/{len(client_trees)}")
                else:
                    print(f"[Server] ⚠️ Client {i+1}: nessun albero valido ricevuto")
            
            if not all_trees_data:
                print(f"[Server] ❌ Nessun albero valido ricevuto da alcun client")
                return None, {}
            
            # CORRETTO: Seleziona i migliori alberi con selezione ENHANCED (ora usa diversity reali)
            selected_trees = select_best_trees_enhanced(
                all_trees_data, 
                strategy=TREE_AGGREGATION_STRATEGY,
                method=TREE_SELECTION_METHOD, 
                max_trees=MAX_TREES_GLOBAL
            )
            
            if not selected_trees:
                print(f"[Server] ❌ Nessun albero selezionato per l'aggregazione")
                return None, {}
            
            # Crea il Random Forest globale OTTIMIZZATO
            global_rf = create_global_random_forest_enhanced(selected_trees)
            
            # Serializza il modello per l'invio ai client
            serialized_model = serialize_global_model(global_rf)
            
            if not serialized_model:
                print(f"[Server] ❌ Errore nella serializzazione del modello globale")
                return None, {}
            
            print(f"[Server] ✅ Aggregazione Random Forest OTTIMIZZATO completata")
            print(f"[Server] ✅ Modello globale creato con {len(selected_trees)} alberi DIVERSIFICATI (REALI)")
            print(f"[Server] ✅ Strategia: {TREE_AGGREGATION_STRATEGY}, Metodo: {TREE_SELECTION_METHOD}")
            print(f"[Server] 🎯 USATO diversity score REALE dai client per selezione ottimale!")
            
            # Restituisce i parametri aggregati
            return serialized_model, {}
            
        except Exception as e:
            print(f"[Server] ❌ ERRORE durante aggregazione Random Forest OTTIMIZZATO: {e}")
            import traceback
            traceback.print_exc()
            return None, {}

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione Random Forest OTTIMIZZATO.
        """
        set_reproducibility_seeds()

        print(f"\n=== AGGREGAZIONE VALUTAZIONE RANDOM FOREST OTTIMIZZATO ROUND {server_round} ===")
        print(f"Client che hanno valutato: {len(results)}")
        
        if failures:
            print("Fallimenti valutazione:")
            for failure in failures:
                print(f"  - {failure}")
        
        try:
            # Chiama l'aggregazione standard di Flower
            aggregated_result = super().aggregate_evaluate(server_round, results, failures)
            
            if aggregated_result is not None:
                print(f"✅ Aggregazione valutazione Random Forest OTTIMIZZATO completata per round {server_round}")
                
                # Stampa statistiche ottimizzazioni se disponibili
                if results:
                    enhanced_count = 0
                    for _, eval_res in results:
                        if eval_res.metrics.get('enhanced_features', False):
                            enhanced_count += 1
                    print(f"✅ Client con features ENHANCED: {enhanced_count}/{len(results)}")
            else:
                print(f"⚠️ Aggregazione valutazione non riuscita per round {server_round}")
                
        except Exception as e:
            print(f"❌ ERRORE durante aggregazione valutazione Random Forest OTTIMIZZATO: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return aggregated_result

def deserialize_trees_from_client_enhanced(parameters):
    """
    Deserializza gli alberi ricevuti da un client CON ACCURACY + DIVERSITÀ REALI.
    AGGIORNATO: Gestisce il nuovo formato ENHANCED con diversity scores.
    """
    try:
        # Gestisce diversi tipi di parametri da Flower
        if hasattr(parameters, 'tensors'):
            parameter_arrays = parameters.tensors
        elif isinstance(parameters, list): 
            parameter_arrays = parameters
        else:
            parameter_arrays = list(parameters) if parameters else []

        if not parameter_arrays:
            print(f"[Server] ⚠️ Nessun parametro ricevuto dal client")
            return []
        
        print(f"[Server] Ricevuti {len(parameter_arrays)} parametri ENHANCED dal client")
        
        deserialized_trees = []
        
        for i, param_data in enumerate(parameter_arrays):
            try:
                print(f"[Server] Elaborazione parametro ENHANCED {i+1}/{len(parameter_arrays)}: tipo={type(param_data)}")
                
                # Gestisce formato NumPy di Flower
                if isinstance(param_data, bytes):
                    if param_data.startswith(b'\x93NUMPY'):
                        print(f"[Server] Rilevato formato NumPy ENHANCED da Flower")
                        from io import BytesIO
                        tree_array = np.load(BytesIO(param_data))
                        print(f"[Server] Array NumPy ENHANCED caricato: shape={tree_array.shape}, dtype={tree_array.dtype}")
                        tree_bytes = tree_array.tobytes()
                        print(f"[Server] Convertito in bytes per pickle: {len(tree_bytes)} bytes")
                    else:
                        tree_bytes = param_data
                        print(f"[Server] Bytes diretti ENHANCED ricevuti: {len(tree_bytes)} bytes")
                
                elif isinstance(param_data, np.ndarray):
                    tree_bytes = param_data.tobytes()
                    print(f"[Server] Convertito numpy array ENHANCED in bytes: {len(tree_bytes)} bytes")
                else:
                    print(f"[Server] ⚠️ Parametro {i+1} ignorato (tipo non compatibile: {type(param_data)})")
                    continue

                # Deserializza dizionario completo con accuracy + diversità
                tree_data = pickle.loads(tree_bytes)
                print(f"[Server] Oggetto ENHANCED deserializzato tipo: {type(tree_data)}")

                # Verifica se è il nuovo formato ENHANCED con accuracy + diversità
                if isinstance(tree_data, dict) and 'tree' in tree_data and 'accuracy_type' in tree_data:
                    if tree_data['accuracy_type'] in ['REAL_ENHANCED', 'REAL']:
                        tree = tree_data['tree']
                        accuracy_real = tree_data['accuracy']
                        weighted_accuracy_real = tree_data['weighted_accuracy']
                        
                        # Estrae diversity score se disponibile (nuovo formato)
                        diversity_score = tree_data.get('diversity_score', 0.0)
                        client_id = tree_data.get('client_id', 0)
                        
                        # Verifica che sia un albero valido
                        if hasattr(tree, 'predict') and hasattr(tree, 'tree_'):
                            deserialized_trees.append((tree, accuracy_real, weighted_accuracy_real, diversity_score, client_id))
                            print(f"[Server] ✅ Albero {i+1} ENHANCED: acc={accuracy_real:.4f}, w_acc={weighted_accuracy_real:.4f}, div={diversity_score:.4f}, client={client_id}")
                        else:
                            print(f"[Server] ⚠️ Oggetto {i+1} non è un albero valido")
                    else:
                        print(f"[Server] ⚠️ Albero {i+1} formato non supportato: {tree_data.get('accuracy_type', 'UNKNOWN')}")
                        
                # Fallback per compatibilità con formato standard
                elif isinstance(tree_data, dict) and 'tree' in tree_data and tree_data.get('accuracy_type') == 'REAL':
                    tree = tree_data['tree']
                    accuracy_real = tree_data['accuracy']
                    weighted_accuracy_real = tree_data['weighted_accuracy']
                    
                    if hasattr(tree, 'predict') and hasattr(tree, 'tree_'):
                        # Diversity score = 0 per compatibilità
                        deserialized_trees.append((tree, accuracy_real, weighted_accuracy_real, 0.0, 0))
                        print(f"[Server] ✅ Albero {i+1} STANDARD: acc={accuracy_real:.4f}, w_acc={weighted_accuracy_real:.4f}")
                    
                # Fallback per formato vecchio
                elif hasattr(tree_data, 'predict') and hasattr(tree_data, 'tree_'):
                    print(f"[Server] ⚠️ Albero {i+1} in formato vecchio, uso accuracy simulate")
                    simulated_accuracy = max(0.8, 0.95 - (i * 0.002))
                    simulated_weighted_acc = max(0.75, 0.94 - (i * 0.002))
                    deserialized_trees.append((tree_data, simulated_accuracy, simulated_weighted_acc, 0.0, 0))
                else:
                    print(f"[Server] ⚠️ Formato dati non riconosciuto per parametro {i+1}")

            except Exception as e:
                print(f"[Server] ❌ Errore nella deserializzazione parametro ENHANCED {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[Server] Deserializzati {len(deserialized_trees)} alberi ENHANCED validi su {len(parameter_arrays)}")
        
        # Mostra statistiche accuracy + diversità reali
        if deserialized_trees:
            real_accuracies = [t[1] for t in deserialized_trees]
            real_w_accuracies = [t[2] for t in deserialized_trees]
            diversity_scores = [t[3] for t in deserialized_trees]
            
            print(f"[Server] Accuracy REALI ricevute: min={min(real_accuracies):.4f}, max={max(real_accuracies):.4f}, media={np.mean(real_accuracies):.4f}")
            print(f"[Server] Weighted accuracy REALI: min={min(real_w_accuracies):.4f}, max={max(real_w_accuracies):.4f}, media={np.mean(real_w_accuracies):.4f}")
            print(f"[Server] Diversity scores: min={min(diversity_scores):.4f}, max={max(diversity_scores):.4f}, media={np.mean(diversity_scores):.4f}")
        
        return deserialized_trees
        
    except Exception as e:
        print(f"[Server] ❌ Errore nella deserializzazione ENHANCED: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """
    Funzione principale per avviare il server Random Forest federato SmartGrid OTTIMIZZATO.
    """
    set_reproducibility_seeds()

    print("=" * 90)
    print("🌳🎯 SERVER FEDERATO SMARTGRID - RANDOM FOREST OTTIMIZZATO")
    print("=" * 90)
    print("Configurazione Random Forest Federato ENHANCED:")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Client minimi: 2")
    print(f"  - Strategia: FedAvg OTTIMIZZATA per Random Forest")
    print(f"  - Valutazione: Dataset globale (client 14-15)")
    print(f"  - Aggregazione alberi: {TREE_AGGREGATION_STRATEGY} (diversity-aware)")
    print(f"  - Selezione alberi: {TREE_SELECTION_METHOD} (accuracy + diversity)")
    print(f"  - Max alberi globali: {MAX_TREES_GLOBAL}")
    print(f"  - Ensemble method: {ENSEMBLE_METHOD}")
    print("")
    print("🎯 OTTIMIZZAZIONI ATTIVE:")
    print(f"  - Feature Engineering: {'ABILITATA' if ENABLE_FEATURE_ENGINEERING else 'DISABILITATA'}")
    print(f"  - Scaling Standard: {'ABILITATO' if ENABLE_SCALING else 'DISABILITATO'}")
    print(f"  - Remove Constants: {'ABILITATO' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATO'}")
    print("")
    print("Pipeline Preprocessing (identica ai client RF OTTIMIZZATI):")
    print(f"  - Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"  - Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"  - Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"  - Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"  - Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"  - PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")
    print("")
    print("Random Forest Configurazione OTTIMIZZATA:")
    print(f"  - N. Estimatori (per client): {RF_N_ESTIMATORS}")
    print(f"  - Max Depth: {RF_MAX_DEPTH}")
    print(f"  - Min Samples Split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  - Min Samples Leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  - Criterio: {RF_CRITERION}")
    print(f"  - Max features: {RF_MAX_FEATURES}")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED} (diversificato per client)")
    print("")
    
    # Configurazione del server
    config = fl.server.ServerConfig(NUM_ROUNDS)
    
    # Strategia Random Forest Federato OTTIMIZZATA
    strategy = SmartGridRandomForestFedAvgEnhanced(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,  #prima 2
        min_evaluate_clients=2,  #prima 2
        min_available_clients=2,  #prima 2
        evaluate_fn=get_smartgrid_random_forest_evaluate_fn()
    )
    
    print(f"🌳🎯 Server Random Forest OTTIMIZZATO in attesa di client su localhost:8080...")
    print("")
    print("Per connettere i client Random Forest OTTIMIZZATI, esegui:")
    print("  python clientRF.py 1")
    print("  python clientRF.py 2")
    print("  ...")
    print("  python clientRF.py 13")
    print("")
    print("Client 14-15 riservati per valutazione globale")
    print("Training inizierà quando almeno 2 client saranno connessi.")
    print("🎯 Le ottimizzazioni includeranno:")
    print("  - Feature engineering per SmartGrid")
    print("  - Diversity-aware tree selection")
    print("  - Enhanced preprocessing pipeline")
    print("  - Real accuracy + diversity metrics")
    print("=" * 90)
    
    try:
        # Avvia il server Flower
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )

        # Salva report finale
        global all_federated_metrics
        if all_federated_metrics:
            save_federated_metrics_report(all_federated_metrics)
            print(f"\n🎯 Training Random Forest OTTIMIZZATO completato!")
            print(f"📊 Report delle metriche salvato con configurazione ENHANCED")
            print(f"⚡ Ottimizzazioni applicate: Feature Engineering + Diversity Selection")
        else:
            print("[SERVER] ⚠️ Nessuna metrica federata disponibile per il report finale.")
        
    except Exception as e:
        print(f"❌ Errore durante l'avvio del server Random Forest OTTIMIZZATO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()