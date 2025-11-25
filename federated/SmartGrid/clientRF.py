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
warnings.filterwarnings('ignore')
from tqdm import tqdm  # Progress bar per adversarial training

# CONFIGURAZIONE SEMI PER RIPRODUCIBILITÀ
RANDOM_SEED = 42

# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==============
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = False       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = True                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False  # PCA per riduzione dimensionalità

if ENABLE_PCA:
    ENABLE_IMPUTATION = True # Per eseguire la PCA non si possono avere NaN
else:
    ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Quando PCA disabilitata, disabilita rimozione feature quasi-costanti per compatibilità dei modelli
    PCA_COMPONENTS = None

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 21  # NUMERO FISSO - garantisce compatibilità automatica (prima 74)
PCA_RANDOM_SEED = 42  # Seme specifico per PCA

# CONFIGURAZIONE MODELLO RANDOM FOREST
# Basato sui risultati del paper: hyperparameter tuning per ottimizzare performance
RF_N_ESTIMATORS = 65      # Numero di alberi nella foresta (dal paper: ottimo tra 65-93)
RF_MAX_DEPTH = None       # Profondità massima degli alberi (None = illimitata)
RF_MIN_SAMPLES_SPLIT = 2  # Campioni minimi per effettuare uno split
RF_MIN_SAMPLES_LEAF = 1   # Campioni minimi in una foglia
RF_MAX_FEATURES = 'sqrt'  # Feature da considerare per ogni split ('sqrt' dal paper)
RF_BOOTSTRAP = True       # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced'  # Gestione automatica dello sbilanciamento
RF_CRITERION = 'entropy'  # Criterio di splitting (dal paper: entropy migliore di gini per molti dataset)

# CONFIGURAZIONE ENSEMBLE PER FEDERATED RANDOM FOREST
# Basato sulla metodologia del paper per aggregazione degli alberi
ENSEMBLE_METHOD = 'weighted_voting'  # 'simple_voting' o 'weighted_voting'
TREE_SELECTION_METHOD = 'accuracy_based'  # Come selezionare i migliori alberi per l'aggregazione

# ============== 🆕 IMPORT CONFIGURAZIONE DIFESA ADVERSARIAL TRAINING CON CACHE INTELLIGENTE DA ==============
script_dir = os.path.dirname(os.path.abspath(__file__))  # federated/SmartGrid/
project_root = os.path.join(script_dir, '..', '..')       # Root del progetto
sys.path.insert(0, project_root)

from attacks.defense_config import (
    DEFENSE_CONFIG,
    get_hsj_config_for_training
)
from attacks.defense_utils import (
    get_smartgrid_physical_constraints_advanced,
    apply_adaptive_constraints,
    calculate_feature_importance_for_defense
)

# USA CONFIGURAZIONE DA defense_config.py
ENABLE_ADVERSARIAL_TRAINING = DEFENSE_CONFIG['ENABLE_ADVERSARIAL_TRAINING']
ADV_TRAINING_EPSILON = DEFENSE_CONFIG['EPSILON']
ADV_TRAINING_MAX_SAMPLES = DEFENSE_CONFIG['MAX_ADVERSARIAL_SAMPLES']
ADV_TRAINING_HSJ_MAX_ITER = DEFENSE_CONFIG['HSJ_MAX_ITER']
ADV_TRAINING_HSJ_MAX_EVAL = DEFENSE_CONFIG['HSJ_MAX_EVAL']
ADV_TRAINING_HSJ_INIT_EVAL = DEFENSE_CONFIG['HSJ_INIT_EVAL']

# CONFIGURAZIONE CACHE INTELLIGENTE
ADV_TRAINING_CACHE_ENABLED = True    # Abilita cache adversarial examples
ADV_TRAINING_REGEN_FREQUENCY = 5     # Rigenera ogni N round (se modello NON cambia)

# Cache globale adversarial examples (condivisa tra round)
adversarial_cache = {
    'X_adv': None,                    # Esempi adversarial cached
    'y_adv': None,                    # Etichette adversarial cached
    'generated_at_round': -1,         # Round di generazione
    'model_hash': None,               # Hash del modello (per rilevare cambiamenti)
    'n_samples': 0                    # Numero campioni cached
}

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
    """
    # Imposta semi per riproducibilità del preprocessing
    set_reproducibility_seeds()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")
    
    df = pd.read_csv(file_path)
    print(f"=== PREPROCESSING FEDERATO RANDOM FOREST ===")
    print(f"Pulizia inf/NaN: {'ABILITATA' if ENABLE_CLEAN_INF_NAN else 'DISABILITATA'}")
    print(f"Clipping outlier: {'ABILITATA' if ENABLE_CLIPPING_OUTLIERS else 'DISABILITATA'}")
    print(f"Imputazione mediana: {'ABILITATA' if ENABLE_IMPUTATION else 'DISABILITATA'}")
    print(f"Rimozione feature quasi-costanti: {'ABILITATA' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else 'DISABILITATA'}")
    print(f"Scaling standard: {'ABILITATA' if ENABLE_SCALING else 'DISABILITATA'}")
    print(f"PCA: {'ABILITATA' if ENABLE_PCA else 'DISABILITATA'}")

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
    if ENABLE_IMPUTATION:
        print(f"[Client {client_id}] Applicazione imputazione mediana...")
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_val_imputed = imputer.transform(X_val_clipped)
    else:
        X_train_imputed = X_train_clipped
        X_val_imputed = X_val_clipped

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
        'remove_near_constant_enabled': ENABLE_REMOVE_NEAR_CONSTANT_FEATURES,
        'pca_components_fixed': PCA_COMPONENTS if ENABLE_PCA else None,
        'preprocessing_method': f"robust_for_rf{'_pca' if ENABLE_PCA else ''}",
        'compatibility_guaranteed': True
    }
    print(f"[Client {client_id}] === CARICAMENTO COMPLETATO ===")
    return X_train_final, y_train, X_val_final, y_val, dataset_info

def create_random_forest_model():
    """
    Crea il modello Random Forest per SmartGrid.
    Implementa la configurazione ottimale basata sul paper.
    
    Returns:
        Modello RandomForestClassifier configurato secondo i risultati del paper
    """

    print(f"[Client {client_id}] === CREAZIONE RANDOM FOREST ===")
    print(f"[Client {client_id}] Modello: Random Forest con {RF_N_ESTIMATORS} alberi")
    print(f"[Client {client_id}] Criterio: {RF_CRITERION} (dal paper: migliore per molti dataset)")
    print(f"[Client {client_id}] Max features: {RF_MAX_FEATURES} (feature selection automatica)")
    print(f"[Client {client_id}] Class weight: {RF_CLASS_WEIGHT} (gestione sbilanciamento)")
    
    # PARAMETRI OTTIMIZZATI BASATI SUL PAPER
    # Il paper mostra che entropy come criterio e sqrt per max_features danno risultati migliori sui dataset di intrusion detection
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,           # Numero di alberi (dal paper: 65-93 range ottimo)
        criterion=RF_CRITERION,                 # Criterio di splitting (entropy vs gini)
        max_depth=RF_MAX_DEPTH,                 # Profondità massima degli alberi
        min_samples_split=RF_MIN_SAMPLES_SPLIT, # Campioni minimi per split
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,   # Campioni minimi per foglia
        max_features=RF_MAX_FEATURES,           # Feature da considerare per ogni split
        bootstrap=RF_BOOTSTRAP,                 # Bootstrap sampling
        random_state=RANDOM_SEED + client_id,               # diversifica per client
        n_jobs=-1,                              # Usa tutti i core disponibili
        class_weight=RF_CLASS_WEIGHT,           # Gestione automatica dello sbilanciamento
        oob_score=True                          # Calcola out-of-bag score per validazione
    )
    
    print(f"[Client {client_id}] Parametri Random Forest:")
    print(f"  - N. estimatori: {RF_N_ESTIMATORS}")
    print(f"  - Criterio: {RF_CRITERION}")
    print(f"  - Max depth: {RF_MAX_DEPTH}")
    print(f"  - Min samples split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  - Min samples leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  - Max features: {RF_MAX_FEATURES}")
    print(f"  - Bootstrap: {RF_BOOTSTRAP}")
    print(f"  - Class weight: {RF_CLASS_WEIGHT}")
    print(f"  - Random state: {RANDOM_SEED}")
    print(f"  - OOB Score: True")
    
    return model

# ============== 🆕 FUNZIONI CACHE INTELLIGENTE ==============

def compute_model_hash(model):
    """
    Calcola hash del modello per rilevare cambiamenti dopo aggregazione.
    
    SPIEGAZIONE:
    In Federated Learning, il server aggrega i modelli dei client e
    invia il modello globale aggiornato. Vogliamo rigenerare adversarial
    SOLO se il modello è cambiato.
    
    STRATEGIA:
    - Serializza primi 5 alberi del Random Forest (rappresentativi)
    - Calcola hash MD5
    - Se hash diverso da cache → modello cambiato → rigenera
    
    Args:
        model: RandomForestClassifier
        
    Returns:
        hash_value: Stringa hash MD5
    """
    import hashlib
    import pickle
    
    try:
        # Serializza primi 5 alberi (rappresentativi ma leggeri)
        trees_sample = model.estimators_[:min(5, len(model.estimators_))]
        model_bytes = pickle.dumps(trees_sample, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Hash MD5
        hash_value = hashlib.md5(model_bytes).hexdigest()
        
        return hash_value
        
    except Exception as e:
        print(f"[Client {client_id}] ⚠️ Errore calcolo hash: {e}")
        # Fallback: usa timestamp come hash
        import time
        return str(int(time.time()))


def should_regenerate_adversarial(current_round, model_instance):
    """
    Decide se rigenerare esempi adversarial (LOGICA CACHE INTELLIGENTE).
    
    DECISIONE BASATA SU 3 FATTORI:
    
    1. PRIMO ROUND: Genera sempre
       - Non ci sono adversarial cached
       - Necessario bootstrap iniziale
    
    2. MODELLO CAMBIATO: Genera sempre
       - Hash modello diverso da cache
       - Significa che server ha aggregato
       - Adversarial vecchi potrebbero non essere efficaci
    
    3. FREQUENZA SCHEDULATA: Rigenera ogni N round
       - Fallback per garantire freshness
       - Anche se modello NON cambia
       - Configurabile (default: 5 round)
    
    MOTIVAZIONE CACHE INTELLIGENTE:
    
    Se modello NON cambia (es. client non selezionato per aggregazione):
    → RIUSA cache (NO spreco computazionale)
    
    Se modello CAMBIA (aggregazione server):
    → RIGENERA (adversarial allineati al nuovo modello)
    
    Args:
        current_round: Round corrente
        model_instance: Random Forest corrente
        
    Returns:
        should_regen: True se deve rigenerare, False se usa cache
    """
    global adversarial_cache
    
    # ========== CASO 1: PRIMO ROUND ==========
    if current_round == 1 or adversarial_cache['X_adv'] is None:
        print(f"[Client {client_id}] Round {current_round}: PRIMO ROUND → Genera adversarial")
        return True
    
    # ========== CASO 2: MODELLO CAMBIATO ==========
    # Calcola hash modello corrente
    current_hash = compute_model_hash(model_instance)
    cached_hash = adversarial_cache.get('model_hash', None)
    
    if current_hash != cached_hash:
        print(f"[Client {client_id}] Round {current_round}: MODELLO CAMBIATO → Rigenera adversarial")
        print(f"  Hash precedente: {cached_hash[:8] if cached_hash else 'N/A'}...")
        print(f"  Hash corrente:   {current_hash[:8]}...")
        return True
    
    # ========== CASO 3: FREQUENZA SCHEDULATA ==========
    rounds_since_last_gen = current_round - adversarial_cache.get('generated_at_round', 0)
    
    if rounds_since_last_gen >= ADV_TRAINING_REGEN_FREQUENCY:
        print(f"[Client {client_id}] Round {current_round}: RIGENERAZIONE SCHEDULATA → Genera adversarial")
        print(f"  Ultimo generato al round: {adversarial_cache['generated_at_round']}")
        print(f"  Round passati: {rounds_since_last_gen} >= {ADV_TRAINING_REGEN_FREQUENCY}")
        return True
    
    # ========== CASO 4: USA CACHE ==========
    print(f"[Client {client_id}] Round {current_round}: USA CACHE adversarial")
    print(f"  Generati al round: {adversarial_cache['generated_at_round']}")
    print(f"  Round passati: {rounds_since_last_gen}/{ADV_TRAINING_REGEN_FREQUENCY}")
    print(f"  Modello: INVARIATO (hash identico)")
    
    return False


# ============== 🆕 ADVERSARIAL TRAINING CON CACHE INTELLIGENTE ==============

def local_adversarial_training(model_instance, X_train, y_train, X_val, y_val, client_id, current_round):
    """
    Adversarial training locale CON CACHE INTELLIGENTE.
    
    ✅ CORREZIONE APPLICATA: Reset indici Pandas per evitare KeyError
    
    PROBLEMA RISOLTO:
    y_train dopo train_test_split() mantiene gli indici originali del DataFrame.
    Quando facciamo y_train[attack_mask], otteniamo una Series con indici NON sequenziali.
    random.sample() genera indici numerici [0, 1, 2, ...] che NON corrispondono
    agli indici Pandas della Series filtrata.
    
    SOLUZIONE:
    Convertiamo le Series Pandas in numpy array PRIMA del sottocampionamento.
    Gli array numpy usano solo posizioni (senza indici label), quindi
    indices = [0, 1, 2, ...] funziona correttamente.
    
    Args:
        model_instance: Random Forest addestrato su dati puliti
        X_train: Training set pulito (numpy array)
        y_train: Etichette training (può essere Series Pandas o numpy array)
        X_val: Validation set
        y_val: Etichette validation
        client_id: ID del client
        current_round: Round corrente
        
    Returns:
        model_robust: Random Forest addestrato su puliti + adversarial
        success: True se completato con successo
    """
    global adversarial_cache
    
    print(f"\n[Client {client_id}] {'='*60}")
    print(f"[Client {client_id}] 🛡️ ADVERSARIAL TRAINING - ROUND {current_round}")
    print(f"[Client {client_id}] {'='*60}")
    
    try:
        # ✅ CORREZIONE: Converti y_train in numpy array se è Pandas Series
        if isinstance(y_train, pd.Series):
            print(f"[Client {client_id}] 🔧 Conversione y_train da Pandas Series a numpy array...")
            y_train_np = y_train.values  # Estrai numpy array SENZA indici Pandas
        else:
            y_train_np = y_train
        
        # STEP 1: Seleziona campioni Attack
        attack_mask = (y_train_np == 1)
        X_attack = X_train[attack_mask]
        y_attack = y_train_np[attack_mask]  # ✅ Usa y_train_np (numpy array)
        
        if len(X_attack) == 0:
            print(f"[Client {client_id}] ⚠️ Nessun campione Attack")
            return model_instance, False
        
        print(f"[Client {client_id}] Campioni Attack: {len(X_attack)}")
        
        # STEP 2: Sottocampionamento ADATTIVO
        max_adv_samples = min(ADV_TRAINING_MAX_SAMPLES, len(X_attack) // 2)
        
        if len(X_attack) > max_adv_samples:
            import random
            random.seed(RANDOM_SEED)
            indices = random.sample(range(len(X_attack)), max_adv_samples)
            
            # ✅ ORA FUNZIONA: y_attack è numpy array, indices sono posizioni
            X_attack_sub = X_attack[indices]
            y_attack_sub = y_attack[indices]  # ✅ Nessun KeyError
        else:
            X_attack_sub = X_attack
            y_attack_sub = y_attack
        
        print(f"[Client {client_id}] Sottocampionamento: {len(X_attack_sub)}")
        
        # ========== 🆕 STEP 3: DECISIONE CACHE INTELLIGENTE ==========
        should_regen = should_regenerate_adversarial(current_round, model_instance)
        
        if should_regen:
            # ========== RIGENERA ADVERSARIAL ==========
            print(f"[Client {client_id}] 🔄 GENERAZIONE NUOVI ADVERSARIAL")
            
            # Wrap modello per ART
            from art.estimators.classification import SklearnClassifier
            from art.attacks.evasion import HopSkipJump
            
            art_classifier = SklearnClassifier(model=model_instance)

            # CONFIGURA HSJ DA defense_config. py (USA get_hsj_config_for_training)
            hsj_config = get_hsj_config_for_training()
            
            hsj_local = HopSkipJump(
                classifier=art_classifier,
                targeted=False,
                norm=hsj_config['norm'],
                max_iter=ADV_TRAINING_HSJ_MAX_ITER,
                max_eval=ADV_TRAINING_HSJ_MAX_EVAL,
                init_eval=ADV_TRAINING_HSJ_INIT_EVAL,
                verbose=hsj_config['verbose']
            )
            
            print(f"[Client {client_id}] ✅ HSJ configurato da defense_config.py:")
            print(f"  - max_iter: {hsj_config['max_iter']}")
            print(f"  - max_eval: {hsj_config['max_eval']}")
            print(f"  - init_eval: {hsj_config['init_eval']}")
            print(f"  - norm: L{hsj_config['norm']}")
            
            # Genera adversarial
            import time
            start_time = time.time()

            print(f"[Client {client_id}] 🔄 Generazione adversarial per {len(X_attack_sub)} campioni...")
            print(f"[Client {client_id}] HSJ: max_iter={ADV_TRAINING_HSJ_MAX_ITER}, max_eval={ADV_TRAINING_HSJ_MAX_EVAL}")

            # ✅ GENERAZIONE CON PROGRESS BAR CAMPIONE PER CAMPIONE
            X_adv_list = []

            with tqdm(total=len(X_attack_sub), 
                    desc=f"[Client {client_id}] HSJ Generation", 
                    unit="campioni", 
                    ncols=100,
                    colour='green') as pbar:
                
                for i in range(len(X_attack_sub)):
                    try:
                        # Genera adversarial per singolo campione
                        x_adv_i = hsj_local.generate(x=X_attack_sub[i:i+1])
                        X_adv_list.append(x_adv_i[0])
                        
                        # Aggiorna progress bar
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n[Client {client_id}] ⚠️ Errore campione {i+1}: {e}")
                        # Usa campione originale in caso di errore
                        X_adv_list.append(X_attack_sub[i])
                        pbar.update(1)
                        continue

            # Converti lista in array numpy
            X_adv = np.array(X_adv_list)
            
            elapsed = time.time() - start_time
            print(f"[Client {client_id}] ✅ Generazione in {elapsed:.1f}s ({len(X_adv)/elapsed:.2f} campioni/sec)")
            
            # Verifica output
            if X_adv is None or len(X_adv) == 0:
                print(f"[Client {client_id}] ⚠️ Generazione fallita")
                return model_instance, False
            
            # APPLICA VINCOLI FISICI DA defense_utils. py
            print(f"[Client {client_id}] Applicazione vincoli fisici SmartGrid...")
            
            # Calcola vincoli fisici con percentili configurabili
            constraints = get_smartgrid_physical_constraints_advanced(
                X_train,
                percentile_low=DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_LOW'],
                percentile_high=DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_HIGH']
            )

            print(f"[Client {client_id}] Vincoli fisici SmartGrid:")
            print(f"  - Range globale: [{constraints['feature_min']. min():.3f}, {constraints['feature_max'].max():.3f}]")
            print(f"  - Percentili: {DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_LOW']}-{DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_HIGH']}")

            # Opzionale: Calcola feature importance per vincoli adattivi
            if DEFENSE_CONFIG. get('USE_ADAPTIVE_CONSTRAINTS', False):
                print(f"[Client {client_id}] Calcolo feature importance per vincoli adattivi...")
                feature_importance = calculate_feature_importance_for_defense(
                    model_instance, X_train, method='gini'
                )
            else:
                feature_importance = None

            # Applica vincoli adattivi (con o senza feature importance)
            X_adv_constrained = apply_adaptive_constraints(
                X_adv,
                X_attack_sub,
                constraints,
                DEFENSE_CONFIG['EPSILON'],
                feature_importance=feature_importance
            )

            print(f"[Client {client_id}] ✅ Vincoli fisici applicati")
            print(f"  - Epsilon: {DEFENSE_CONFIG['EPSILON']}")
            if feature_importance is not None:
                print(f"  - Vincoli adattivi: ABILITATI (feature importance)")
            else:
                print(f"  - Vincoli adattivi: DISABILITATI (epsilon uniforme)")
            
            y_adv = y_attack_sub
            
            # ========== 🆕 AGGIORNA CACHE ==========
            model_hash = compute_model_hash(model_instance)
            adversarial_cache = {
                'X_adv': X_adv_constrained.copy(),
                'y_adv': y_adv.copy(),
                'generated_at_round': current_round,
                'model_hash': model_hash,
                'n_samples': len(X_adv_constrained)
            }
            
            print(f"[Client {client_id}] 💾 Cache aggiornata:")
            print(f"  Round: {current_round}")
            print(f"  Campioni: {len(X_adv_constrained)}")
            print(f"  Hash modello: {model_hash[:8]}...")
            
        else:
            # ========== USA CACHE ==========
            print(f"[Client {client_id}] 📦 USO CACHE ADVERSARIAL")
            
            X_adv_constrained = adversarial_cache['X_adv']
            y_adv = adversarial_cache['y_adv']
            
            print(f"[Client {client_id}] Cache info:")
            print(f"  Generati al round: {adversarial_cache['generated_at_round']}")
            print(f"  Campioni cached: {adversarial_cache['n_samples']}")
            print(f"  Hash modello: {adversarial_cache['model_hash'][:8] if adversarial_cache['model_hash'] else 'N/A'}...")
            
            # Verifica compatibilità dimensionale
            if len(X_adv_constrained) != len(X_attack_sub):
                print(f"[Client {client_id}] ⚠️ Cache size mismatch: {len(X_adv_constrained)} vs {len(X_attack_sub)}")
                print(f"[Client {client_id}] Fallback: rigenero...")
                
                # Invalida cache e rigenera
                adversarial_cache['X_adv'] = None
                return local_adversarial_training(model_instance, X_train, y_train_np, X_val, y_val, client_id, current_round)
        
        # STEP 4: Data Augmentation
        X_aug = np.concatenate([X_train, X_adv_constrained], axis=0)
        y_aug = np.concatenate([y_train_np, y_adv], axis=0)
        
        # Shuffle
        indices_shuffle = np.random.permutation(len(X_aug))
        X_aug = X_aug[indices_shuffle]
        y_aug = y_aug[indices_shuffle]
        
        print(f"[Client {client_id}] Dataset: {len(X_train)} → {len(X_aug)} (+{len(X_adv_constrained)} adversarial)")
        
        # STEP 5: Riaddestra Random Forest
        model_robust = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            criterion=RF_CRITERION,
            max_features=RF_MAX_FEATURES,
            class_weight=RF_CLASS_WEIGHT,
            random_state=RANDOM_SEED + client_id,
            n_jobs=-1
        )
        
        model_robust.fit(X_aug, y_aug)
        
        # STEP 6: Valuta
        if X_val is not None and len(X_val) > 0:
            val_acc_clean = model_instance.score(X_val, y_val)
            val_acc_robust = model_robust.score(X_val, y_val)
            
            print(f"[Client {client_id}] 📊 Validation Accuracy:")
            print(f"  Pulito:  {val_acc_clean:.4f}")
            print(f"  Robusto: {val_acc_robust:.4f}")
            print(f"  Δ:       {val_acc_robust - val_acc_clean:+.4f}")
        
        print(f"[Client {client_id}] {'='*60}")
        print(f"[Client {client_id}] ✅ ADVERSARIAL TRAINING COMPLETATO")
        print(f"[Client {client_id}] {'='*60}\n")
        
        return model_robust, True
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return model_instance, False

# ============== FINE ADVERSARIAL TRAINING CON CACHE ==============

def extract_trees_from_forest(model, X_val, y_val):
    """
    Estrae gli alberi dal Random Forest e calcola le loro performance individuali REALI.
    Implementa la metodologia del paper per la selezione degli alberi migliori.
    
    Args:
        model: Random Forest addestrato
        X_val: Dati di validazione
        y_val: Etichette di validazione
        
    Returns:
        Lista di tuple (tree, accuracy_reale, weighted_accuracy_reale) per ogni albero
    """
    print(f"[Client {client_id}] === ESTRAZIONE ALBERI CON ACCURACY REALI ===")

    print(f"[Client {client_id}] 🔍 DEBUG extract_trees_from_forest: INIZIO")
    print(f"[Client {client_id}] 🔍 DEBUG: model type = {type(model)}")
    print(f"[Client {client_id}] 🔍 DEBUG: X_val shape = {X_val.shape}")
    print(f"[Client {client_id}] 🔍 DEBUG: y_val shape = {y_val.shape if hasattr(y_val, 'shape') else len(y_val)}")

    # CONTROLLO: Verifica se il modello è addestrato
    if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        print(f"[Client {client_id}] ⚠️ Modello non ancora addestrato, nessun albero disponibile")
        return []  # Restituisce lista vuota
    
    # ✅ Converti y_val in numpy array se è Pandas Series
    if isinstance(y_val, pd.Series):
        y_val_np = y_val.values
    else:
        y_val_np = y_val
    
    print(f"[Client {client_id}] 🔍 DEBUG: Modello ha {len(model.estimators_)} alberi")
    print(f"[Client {client_id}] === CALCOLO ACCURACY REALI PER {len(model.estimators_)} ALBERI ===")
    
    trees_performance = []
    
    for i, tree in enumerate(model.estimators_):
        # Predizioni dell'albero singolo
        tree_predictions = tree.predict(X_val)
        
        # Calcola accuracy standard REALE
        accuracy_real = accuracy_score(y_val_np, tree_predictions)
        
        # Calcola weighted accuracy REALE
        # Weighted accuracy considera la distribuzione delle classi
        class_counts = np.bincount(y_val_np)
        weights = 1.0 / class_counts  # Peso inversamente proporzionale alla frequenza
        class_weights_norm = weights / weights.sum()  # Normalizza i pesi
        
        # Calcola accuracy pesata per classe REALE
        weighted_acc_real = 0.0
        for class_label in np.unique(y_val_np):
            class_mask = (y_val_np == class_label)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_val_np[class_mask], tree_predictions[class_mask])
                weighted_acc_real += class_accuracy * class_weights_norm[class_label]
        
        trees_performance.append((tree, accuracy_real, weighted_acc_real))
        
        if i < 5:  # Stampa info per i primi 5 alberi
            print(f"[Client {client_id}] Albero {i+1}: Accuracy REALE={accuracy_real:.4f}, Weighted Accuracy REALE={weighted_acc_real:.4f}")
    
    print(f"[Client {client_id}] 🔍 DEBUG extract_trees_from_forest: COMPLETATO con {len(trees_performance)} alberi CON ACCURACY REALI")

    # Ordina gli alberi per performance REALE (weighted accuracy come nel paper)
    trees_performance.sort(key=lambda x: x[2], reverse=True)  # Ordina per weighted accuracy REALE
    
    print(f"[Client {client_id}] Migliore albero (REALE): Accuracy={trees_performance[0][1]:.4f}, Weighted Accuracy={trees_performance[0][2]:.4f}")
    print(f"[Client {client_id}] Peggiore albero (REALE): Accuracy={trees_performance[-1][1]:.4f}, Weighted Accuracy={trees_performance[-1][2]:.4f}")
    
    return trees_performance

def serialize_trees_for_aggregation(trees_performance, max_trees=None):
    """
    Serializza gli alberi con le loro accuracy reali per l'invio al server.
    Invia dizionario completo con accuracy reali.
    """
    print(f"[Client {client_id}] === SERIALIZZAZIONE ALBERI CON ACCURACY REALI ===")
    
    if max_trees is not None:
        selected_trees = trees_performance[:max_trees]
        print(f"[Client {client_id}] Selezionati {len(selected_trees)} migliori alberi su {len(trees_performance)}")
    else:
        selected_trees = trees_performance
        print(f"[Client {client_id}] Invio tutti i {len(selected_trees)} alberi")
    
    serialized_data = []
    
    for i, (tree, accuracy_real, weighted_accuracy_real) in enumerate(selected_trees):
        try:
            # CORREZIONE: Crea dizionario con albero + accuracy REALI
            tree_data = {
                'tree': tree,
                'accuracy': accuracy_real,
                'weighted_accuracy': weighted_accuracy_real,
                'tree_index': i,
                'accuracy_type': 'REAL'  # Flag per indicare che sono accuracy reali
            }
            
            # Serializza l'intero dizionario con pickle
            tree_bytes = pickle.dumps(tree_data, protocol=pickle.HIGHEST_PROTOCOL)

            # Converti in array uint8 (formato sicuro per Flower)
            tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
            serialized_data.append(tree_array)

            print(f"[Client {client_id}] ✅ Albero {i+1} serializzato con accuracy REALI ({len(tree_bytes)} bytes)")
            print(f"[Client {client_id}]    Accuracy REALE: {accuracy_real:.4f}, Weighted REALE: {weighted_accuracy_real:.4f}")

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore serializzazione albero {i+1}: {e}")
            import traceback; traceback.print_exc()
            continue
    
    print(f"[Client {client_id}] Serializzati {len(serialized_data)} alberi con ACCURACY REALI")

    # ===== DEBUG FLOWER FORMAT =====
    if serialized_data:
        first = serialized_data[0]
        print(f"[Client {client_id}] DEBUG Primo albero serializzato CON ACCURACY REALI:")
        print(f"  Tipo: {type(first)}, dtype: {first.dtype}, shape: {first.shape}")
        print(f"  Prime 10 byte: {first[:10].tolist()}")
    else:
        print(f"[Client {client_id}] ⚠️ Nessun albero serializzato!")

    return serialized_data

class SmartGridRandomForestClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con Random Forest.
    Implementa la metodologia del paper per l'aggregazione federata di Random Forest.
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
            print(f"[Client {client_id}] 🔍 DEBUG: Chiamo extract_trees_from_forest...")
            # Estrai e valuta le performance degli alberi
            trees_performance = extract_trees_from_forest(model, X_val, y_val)
            print(f"[Client {client_id}] 🔍 DEBUG: extract_trees_from_forest completata, {len(trees_performance)} alberi")

            print(f"[Client {client_id}] 🔍 DEBUG: Chiamo serialize_trees_for_aggregation...")
            # Serializza gli alberi con verifica
            serialized_trees = serialize_trees_for_aggregation(trees_performance)
            print(f"[Client {client_id}] 🔍 DEBUG: serialize_trees_for_aggregation completata, {len(serialized_trees)} alberi")

            # Debug se non ci sono alberi serializzati
            if len(serialized_trees) == 0:
                print(f"[Client {client_id}] ⚠️ Nessun albero serializzato — invio parametri vuoti")
                return []
        
            print(f"[Client {client_id}] 🔍 DEBUG: Invio {len(serialized_trees)} alberi al server")
            # Gli alberi sono già numpy arrays (uint8) pronti per Flower
            print(f"[Client {client_id}] Invio {len(serialized_trees)} alberi al server")
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
                print(f"[Client {client_id}] ✅ Modello aggregato ricevuto dal server")
                print(f"[Client {client_id}] Nuovo modello ha {model.n_estimators} alberi")
                    
        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore nell'impostazione parametri: {e}")
            import traceback
            traceback.print_exc()
            # Mantieni il modello corrente in caso di errore
            pass

    def fit(self, parameters, config):
        """
        Addestra il modello Random Forest locale CON ADVERSARIAL TRAINING (cache intelligente).
        """
        global model, X_train, y_train, X_val, y_val, dataset_info
        
        # 🆕 Estrai round corrente da config (Flower passa automaticamente)
        current_round = config.get("server_round", 1)
        
        print(f"[Client {client_id}] === ROUND {current_round} - TRAINING ===")
        
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
            
            # ============== STEP 1: TRAINING SU DATI PULITI ==============
            print(f"[Client {client_id}] Addestramento Random Forest su {len(X_train_clean)} campioni...")
            model.fit(X_train_clean, y_train)
            
            # Verifica che il modello sia stato addestrato
            if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
                raise RuntimeError("Random Forest non addestrato correttamente - nessun albero trovato")
            
            print(f"[Client {client_id}] ✅ Random Forest addestrato con {len(model.estimators_)} alberi")
            
            # ============== STEP 2: ADVERSARIAL TRAINING CON CACHE ==============
            if ENABLE_ADVERSARIAL_TRAINING:
                print(f"[Client {client_id}] 🛡️ Adversarial Training ABILITATO")
                
                # 🆕 Passa round a local_adversarial_training
                model_robust, success = local_adversarial_training(
                    model,
                    X_train_clean,
                    y_train,
                    X_val,
                    y_val,
                    client_id,
                    current_round  # 🆕 Passa round corrente
                )
                
                if success:
                    model = model_robust
                    print(f"[Client {client_id}] ✅ Modello ROBUSTO attivo")
                else:
                    print(f"[Client {client_id}] ⚠️ Adversarial training fallito, uso modello pulito")
            else:
                print(f"[Client {client_id}] Adversarial Training DISABILITATO")
            
            # ============== STEP 3: CALCOLA METRICHE ==============
            # ✅ Converti y_train in numpy array se è Series Pandas
            if isinstance(y_train, pd.Series):
                y_train_np = y_train.values
            else:
                y_train_np = y_train
            
            train_predictions = model.predict(X_train_clean)
            train_prob = model.predict_proba(X_train_clean)[:, 1]
            
            train_accuracy = accuracy_score(y_train_np, train_predictions)
            train_precision = precision_score(y_train_np, train_predictions, zero_division=0)
            train_recall = recall_score(y_train_np, train_predictions, zero_division=0)
            train_f1 = f1_score(y_train_np, train_predictions, zero_division=0)
            train_balanced_acc = balanced_accuracy_score(y_train_np, train_predictions)
            
            try:
                train_auc = roc_auc_score(y_train_np, train_prob)
            except:
                train_auc = 0.0
            
            oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else 0.0
            
            print(f"[Client {client_id}] Training completato!")
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
            
            # 🆕 Adversarial training info con cache
            'adversarial_training_enabled': bool(ENABLE_ADVERSARIAL_TRAINING),
            'adversarial_cache_used': bool(not should_regenerate_adversarial(current_round, model) if ENABLE_ADVERSARIAL_TRAINING else False),
            'adversarial_cache_round': int(adversarial_cache.get('generated_at_round', -1))
        }
        
        # Restituisce gli alberi del modello addestrato
        try:
            trees_perf_real = extract_trees_from_forest(model, X_val, y_val)
            serialized_trees = serialize_trees_for_aggregation(trees_perf_real)
            
            print(f"[Client {client_id}] Invio {len(serialized_trees)} alberi CON ACCURACY REALI al server...")
            return serialized_trees, len(X_train), metrics

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore serializzazione finale: {e}")
            import traceback; traceback.print_exc()
            return [], 0, {'error': f'serialization_failed: {str(e)}'}

    def evaluate(self, parameters, config):
        """
        Valuta il modello Random Forest.
        """
        global model, X_val, y_val

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
            
            # ✅ Converti y_val in numpy array se è Series Pandas
            if isinstance(y_val, pd.Series):
                y_val_np = y_val.values
            else:
                y_val_np = y_val
            
            # Valutazione Random Forest
            val_predictions = model.predict(X_val_clean)
            val_prob = model.predict_proba(X_val_clean)[:, 1]  # Probabilità classe positiva
            
            # Calcola metriche
            accuracy = accuracy_score(y_val_np, val_predictions)
            precision = precision_score(y_val_np, val_predictions, zero_division=0)
            recall = recall_score(y_val_np, val_predictions, zero_division=0)
            f1_score_val = f1_score(y_val_np, val_predictions, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_val_np, val_predictions)
            
            # AUC
            try:
                auc = roc_auc_score(y_val_np, val_prob)
            except:
                auc = 0.0
            
            # Metriche per classe
            report = classification_report(y_val_np, val_predictions, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_val_np, val_predictions)

            print(f"[Client {client_id}] Val Accuracy: {accuracy:.4f}, Val F1: {f1_score_val:.4f}")
            print(f"[Client {client_id}] Val Balanced Acc: {balanced_acc:.4f}, Val AUC: {auc:.4f}")
            print(f"[Client {client_id}] Classification report (per classe):")
            print(classification_report(y_val_np, val_predictions, target_names=["natural", "attack"], zero_division=0))
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
                "tp": int(conf_matrix[1, 1])
            }
            
            return loss, len(X_val), metrics
            
        except Exception as e:
            print(f"[Client {client_id}] Errore durante valutazione: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}


def main():
    """
    Funzione principale per avviare il client SmartGrid Random Forest.
    """

    global client_id, model, X_train, y_train, X_val, y_val, dataset_info

    # Imposta semi all'avvio del client
    set_reproducibility_seeds()
    
    if len(sys.argv) != 2:
        print("Uso: python clientRF.py <client_id>")
        print("Esempio: python clientRF.py 1")
        sys.exit(1)
    
    # momentaneamente modificato per testare su client 1-13 invece che 14-15
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("⚠️ Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"❌ Errore: Client ID non valido. {e}")
        sys.exit(1)
    
    print(f"=== AVVIO CLIENT RANDOM FOREST {client_id} ===")
    print(f"🛡️ Adversarial Training: {'ABILITATO' if ENABLE_ADVERSARIAL_TRAINING else 'DISABILITATO'}")
    if ENABLE_ADVERSARIAL_TRAINING:
        print(f"   Epsilon: {ADV_TRAINING_EPSILON}")
        print(f"   Max samples: {ADV_TRAINING_MAX_SAMPLES}")
        print(f"   HSJ config: max_iter={ADV_TRAINING_HSJ_MAX_ITER}, max_eval={ADV_TRAINING_HSJ_MAX_EVAL}")
        print(f"   Cache intelligente: {'ABILITATA' if ADV_TRAINING_CACHE_ENABLED else 'DISABILITATA'}")
        if ADV_TRAINING_CACHE_ENABLED:
            print(f"   Frequenza rigenerazione: ogni {ADV_TRAINING_REGEN_FREQUENCY} round")
    
    try:
        # Carica i dati con preprocessing minimale per Random Forest
        print(f"[Client {client_id}] Caricamento dati per Random Forest...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)

        # Crea il modello Random Forest
        model = create_random_forest_model()

        print(f"[Client {client_id}] === RIASSUNTO CLIENT RANDOM FOREST ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['final_features']}")
        print(f"[Client {client_id}] Modello: Random Forest con {model.n_estimators} alberi")
        print(f"[Client {client_id}] Criterio: {model.criterion}, Max features: {model.max_features}")
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