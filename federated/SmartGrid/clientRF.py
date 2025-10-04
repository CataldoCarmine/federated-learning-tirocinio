import flwr as fl
import pandas as pd
import numpy as np
import sys
import os
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
warnings.filterwarnings('ignore')

# CONFIGURAZIONE SEMI PER RIPRODUCIBILITÀ
RANDOM_SEED = 42

# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==============
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = True       # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = False                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False  # PCA

if ENABLE_PCA:
    ENABLE_IMPUTATION = True # Per eseguire la PCA non si possono avere NaN

# CONFIGURAZIONE PCA STATICA
PCA_COMPONENTS = 74  # NUMERO FISSO - garantisce compatibilità automatica
PCA_RANDOM_SEED = 42  # Seme specifico per PCA

# Quando PCA disabilitata, disabilita rimozione feature quasi-costanti per compatibilità dei modelli
if ENABLE_PCA == False:
    ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False
    PCA_COMPONENTS = None

# ============== CONFIGURAZIONE RANDOM FOREST ==============
RF_N_ESTIMATORS = 100          # Numero di alberi nella foresta (come nel paper)
RF_MAX_DEPTH = None            # Profondità massima degli alberi (None = illimitata)
RF_MIN_SAMPLES_SPLIT = 2       # Campioni minimi per effettuare uno split
RF_MIN_SAMPLES_LEAF = 1        # Campioni minimi in una foglia
RF_MAX_FEATURES = 'sqrt'       # Feature da considerare per ogni split
RF_BOOTSTRAP = True            # Usa bootstrap sampling
RF_CLASS_WEIGHT = 'balanced'   # Gestione automatica dello sbilanciamento
RF_CRITERION = 'gini'          # Splitting rule: 'gini' o 'entropy'

# ============== FUNZIONI DI RIPRODUCIBILITÀ ==============
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

# ============== FUNZIONI DI PREPROCESSING ==============
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
    - NON usa threshold fissi
    - NON azzera valori piccoli
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
                raise ValueError(f"PCA client {client_id} ha prodotto output con NaN o inf")
            if X_pca.shape[1] != n_components:
                raise ValueError(f"PCA output shape inconsistente: {X_pca.shape[1]} vs {n_components}")

            variance_explained = np.sum(pca.explained_variance_ratio_)
            print(f"[Client {client_id}] ✅ PCA fissa applicata: {X_pca.shape}")
            print(f"[Client {client_id}] Varianza spiegata: {variance_explained*100:.2f}%")
            return X_pca

    except Exception as e:
        print(f"[Client {client_id}] ERRORE PCA: {e}")
        print(f"[Client {client_id}] Attivazione fallback semplificato...")
        n_fallback = min(n_components, original_features)
        X_fallback = X_preprocessed[:, :n_fallback]
        print(f"[Client {client_id}] ✅ Fallback: {X_fallback.shape}")
        return X_fallback

# ============== CARICAMENTO DATI ==============
def load_client_smartgrid_data(client_id):
    """
    Carica i dati SmartGrid per un client specifico con preprocessing configurabile.
    Applica la stessa pipeline del client DNN per mantenere compatibilità.
    """
    # Imposta semi per riproducibilità del preprocessing
    set_reproducibility_seeds()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    df = pd.read_csv(file_path)
    print(f"=== PREPROCESSING FEDERATO RF ===")
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
    if ENABLE_CLEAN_INF_NAN:
        X_cleaned = clean_data_for_pca(X)
    else:
        X_cleaned = X.values if hasattr(X, 'values') else X

    # Suddivisione train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_cleaned, y,
        test_size=0.3,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"[Client {client_id}] Suddivisione: {len(X_train_raw)} training, {len(X_val_raw)} validation")

    # STEP 2: Clipping outlier per quantili SOLO su training, applicato anche a validation usando limiti del train
    if ENABLE_CLIPPING_OUTLIERS:
        X_train_np = np.array(X_train_raw, dtype=float)
        X_val_np = np.array(X_val_raw, dtype=float)
        lower, upper = fit_clip_outliers_iqr(X_train_np, k=5.0)
        X_train_clipped = transform_clip_outliers_iqr(X_train_np, lower, upper)
        X_val_clipped = transform_clip_outliers_iqr(X_val_np, lower, upper)
    else:
        X_train_clipped = X_train_raw
        X_val_clipped = X_val_raw

    # STEP 3: Imputazione mediana
    if ENABLE_IMPUTATION:
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_val_imputed = imputer.transform(X_val_clipped)
    else:
        X_train_imputed = X_train_clipped
        X_val_imputed = X_val_clipped

    # STEP 4: Rimozione feature quasi-costanti (CONDIZIONALE)
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
        print(f"[Client {client_id}] Preprocessing completato (clipping, imputazione, {('costanti, ' if ENABLE_REMOVE_NEAR_CONSTANT_FEATURES else '')}scaling)")
    else:
        X_train_scaled = X_train_reduced
        X_val_scaled = X_val_reduced
        print(f"[Client {client_id}] Scaling DISABILITATO - usando dati preprocessati direttamente: {X_train_scaled.shape}")

    # STEP 6: PCA (CONDIZIONALE)
    if ENABLE_PCA:
        X_train_final = apply_pca(X_train_scaled, client_id=client_id)
        X_val_final = apply_pca(X_val_scaled, client_id=client_id)
        expected_features = PCA_COMPONENTS
        if X_train_final.shape[1] != expected_features:
            raise RuntimeError(f"Client {client_id}: PCA output shape inconsistente: {X_train_final.shape} vs {expected_features}")
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        print(f"[Client {client_id}] PCA DISABILITATA - usando dati scalati direttamente: {X_train_final.shape}")

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
        'final_features': X_train_final.shape[1]
    }

    return X_train_final, y_train, X_val_final, y_val, dataset_info

# ============== FUNZIONI DI SERIALIZZAZIONE ALBERI ==============
def serialize_tree(tree):
    """
    Serializza un albero di decisione di scikit-learn.
    Usa pickle per convertire l'albero in bytes, poi lo codifica in numpy array.
    
    Args:
        tree: DecisionTreeClassifier di scikit-learn
        
    Returns:
        dict con albero serializzato e metadati
    """
    try:
        # Serializza l'albero usando pickle
        tree_bytes = pickle.dumps(tree)
        # Converti in numpy array per compatibilità Flower
        tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)

        return {
            'tree_data': tree_array,
            'tree_size': len(tree_array)
        }
    except Exception as e:
        print(f"Errore serializzazione albero: {e}")
        raise

def deserialize_tree(tree_dict):
    """
    Deserializza un albero di decisione.
    
    Args:
        tree_dict: dizionario con albero serializzato
        
    Returns:
        DecisionTreeClassifier deserializzato
    """
    try:
        # Ricostruisci i bytes dall'array numpy
        tree_bytes = tree_dict['tree_data'].tobytes()
        # Deserializza l'albero
        tree = pickle.loads(tree_bytes)

        return tree
    except Exception as e:
        print(f"Errore deserializzazione albero: {e}")
        raise

def serialize_random_forest_trees(rf_model, X_val, y_val):
    """
    Serializza tutti gli alberi del Random Forest con metadati per aggregazione.
    Calcola accuracy e weighted accuracy per ogni albero come nel paper.
    
    Args:
        rf_model: RandomForestClassifier addestrato
        X_val: dati di validazione per calcolare accuracy
        y_val: etichette di validazione
        
    Returns:
        lista di dizionari con alberi serializzati e metadati
    """
    trees_data = []

    # Estrai i singoli alberi dalla foresta
    estimators = rf_model.estimators_

    print(f"Serializzazione {len(estimators)} alberi...")

    for idx, tree in enumerate(estimators):
        # Predizione con l'albero singolo
        y_pred = tree.predict(X_val)

        # Calcola accuracy per questo albero
        tree_accuracy = accuracy_score(y_val, y_pred)

        # Calcola weighted accuracy (basato sulla distribuzione delle classi)
        # Usa balanced accuracy come approssimazione del weighted accuracy
        tree_weighted_accuracy = balanced_accuracy_score(y_val, y_pred)

        # Serializza l'albero
        tree_serialized = serialize_tree(tree)

        # Aggiungi metadati
        tree_data = {
            'tree_index': idx,
            'tree_data': tree_serialized['tree_data'],
            'tree_size': tree_serialized['tree_size'],
            'accuracy': tree_accuracy,
            'weighted_accuracy': tree_weighted_accuracy
        }

        trees_data.append(tree_data)

    print(f"✅ {len(trees_data)} alberi serializzati")
    return trees_data

def deserialize_random_forest_trees(trees_data):
    """
    Deserializza una lista di alberi e ricostruisce un Random Forest.
    
    Args:
        trees_data: lista di dizionari con alberi serializzati
        
    Returns:
        RandomForestClassifier ricostruito
    """
    try:
        # Deserializza tutti gli alberi
        trees = []
        for tree_dict in trees_data:
            tree = deserialize_tree(tree_dict)
            trees.append(tree)

        # Crea un nuovo Random Forest con gli alberi deserializzati
        # Nota: scikit-learn non supporta direttamente la creazione di RF da alberi esistenti
        # Quindi creiamo un RF vuoto e sostituiamo gli estimators
        rf = RandomForestClassifier(
            n_estimators=len(trees),
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        # Impostiamo gli alberi (questo è un workaround per scikit-learn)
        rf.estimators_ = trees
        rf.n_estimators = len(trees)

        # Impostiamo altri attributi necessari dal primo albero
        if len(trees) > 0:
            rf.n_features_in_ = trees[0].n_features_in_
            rf.n_classes_ = trees[0].n_classes_
            rf.classes_ = trees[0].classes_
            rf.n_outputs_ = trees[0].n_outputs_

        print(f"✅ Random Forest ricostruito con {len(trees)} alberi")
        return rf

    except Exception as e:
        print(f"Errore deserializzazione Random Forest: {e}")
        raise

# ============== CREAZIONE MODELLO ==============
def create_random_forest_model():
    """
    Crea un modello Random Forest con i parametri configurati.
    """
    print(f"[Client] === CREAZIONE RANDOM FOREST ===")
    print(f"[Client] N. estimatori: {RF_N_ESTIMATORS}")
    print(f"[Client] Max depth: {RF_MAX_DEPTH}")
    print(f"[Client] Min samples split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"[Client] Min samples leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"[Client] Max features: {RF_MAX_FEATURES}")
    print(f"[Client] Bootstrap: {RF_BOOTSTRAP}")
    print(f"[Client] Class weight: {RF_CLASS_WEIGHT}")
    print(f"[Client] Criterion: {RF_CRITERION}")
    print(f"[Client] Random state: {RANDOM_SEED}")

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        bootstrap=RF_BOOTSTRAP,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight=RF_CLASS_WEIGHT,
        criterion=RF_CRITERION
    )

    return model

# ============== VARIABILI GLOBALI ==============
client_id = None
rf_model = None
X_train = None
y_train = None
X_val = None
y_val = None
dataset_info = None

# ============== CLIENT FLOWER ==============
class SmartGridRFClient(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con Random Forest.
    Implementa l'architettura federata del paper:
    - Ogni client addestra un Random Forest locale
    - Gli alberi vengono serializzati e inviati al server
    - Il server aggrega gli alberi in un Global Random Forest
    """

    def get_parameters(self, config):
        """
        Restituisce i parametri del modello (alberi serializzati).
        Per il primo round, restituisce una lista vuota.
        """
        global rf_model, X_val, y_val

        # Se il modello non è ancora stato addestrato, restituisci lista vuota
        if rf_model is None or not hasattr(rf_model, 'estimators_'):
            print(f"[Client {client_id}] Nessun modello addestrato ancora, restituisco parametri vuoti")
            return []

        try:
            # Serializza gli alberi del Random Forest
            trees_data = serialize_random_forest_trees(rf_model, X_val, y_val)

            # Converti in formato compatibile con Flower (lista di numpy arrays)
            # Flower si aspetta una lista di numpy arrays, quindi dobbiamo convertire
            parameters = []
            for tree_data in trees_data:
                # Crea un array che contiene tutti i dati del tree
                # Format: [tree_index, tree_size, accuracy, weighted_accuracy, tree_data...]
                tree_params = np.concatenate([
                    np.array([tree_data['tree_index']], dtype=np.float32),
                    np.array([tree_data['tree_size']], dtype=np.float32),
                    np.array([tree_data['accuracy']], dtype=np.float32),
                    np.array([tree_data['weighted_accuracy']], dtype=np.float32),
                    tree_data['tree_data'].astype(np.float32)
                ])
                parameters.append(tree_params)

            print(f"[Client {client_id}] ✅ Restituiti {len(parameters)} alberi serializzati")
            return parameters

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore get_parameters: {e}")
            import traceback
            traceback.print_exc()
            return []

    def fit(self, parameters, config):
        """
        Addestra il Random Forest locale.
        
        Args:
            parameters: parametri ricevuti dal server (non usati nel primo round)
            config: configurazione dell'addestramento
            
        Returns:
            Tuple con (parametri_aggiornati, numero_campioni, metriche)
        """
        global rf_model, X_train, y_train, X_val, y_val, dataset_info

        # Imposta semi per riproducibilità
        set_reproducibility_seeds()

        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO RF ===")

        if len(X_train) == 0:
            print(f"[Client {client_id}] Nessun dato di training!")
            return [], 0, {}

        try:
            # Crea un nuovo Random Forest per questo round
            rf_model = create_random_forest_model()

            print(f"[Client {client_id}] Addestramento Random Forest su {len(X_train)} campioni...")

            # Addestra il Random Forest
            rf_model.fit(X_train, y_train)

            print(f"[Client {client_id}] ✅ Addestramento completato")

            # Valutazione sul training set
            y_train_pred = rf_model.predict(X_train)
            y_train_pred_proba = rf_model.predict_proba(X_train)[:, 1]

            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = 0.0
            train_recall = 0.0
            train_auc = 0.0
            train_f1 = 0.0
            train_balanced_acc = 0.0

            # Calcola metriche se ci sono entrambe le classi
            if len(np.unique(y_train)) > 1:
                from sklearn.metrics import precision_score, recall_score
                train_precision = precision_score(y_train, y_train_pred, zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, zero_division=0)
                train_auc = roc_auc_score(y_train, y_train_pred_proba)
                train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
                train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

            print(f"[Client {client_id}] Train Accuracy: {train_accuracy:.4f}")
            print(f"[Client {client_id}] Train F1: {train_f1:.4f}, Balanced Acc: {train_balanced_acc:.4f}")

            # Serializza gli alberi
            trees_data = serialize_random_forest_trees(rf_model, X_val, y_val)

            # Converti in formato Flower
            parameters = []
            for tree_data in trees_data:
                tree_params = np.concatenate([
                    np.array([tree_data['tree_index']], dtype=np.float32),
                    np.array([tree_data['tree_size']], dtype=np.float32),
                    np.array([tree_data['accuracy']], dtype=np.float32),
                    np.array([tree_data['weighted_accuracy']], dtype=np.float32),
                    tree_data['tree_data'].astype(np.float32)
                ])
                parameters.append(tree_params)

            # Metriche da inviare al server
            metrics = {
                # Metriche base
                'train_accuracy': float(train_accuracy),
                'train_precision': float(train_precision),
                'train_recall': float(train_recall),
                'train_auc': float(train_auc),
                'train_f1_score': float(train_f1),
                'train_balanced_accuracy': float(train_balanced_acc),

                # Info Random Forest
                'n_estimators': int(RF_N_ESTIMATORS),
                'n_trees_sent': len(parameters),

                # Dataset info
                'client_id': int(dataset_info['client_id']),
                'train_samples': int(dataset_info['train_samples']),
            }

            print(f"[Client {client_id}] ✅ Invio {len(parameters)} alberi al server")

            return parameters, len(X_train), metrics

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore durante addestramento: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, {'error': f'training_failed: {str(e)}'}

    def evaluate(self, parameters, config):
        """
        Valuta il modello Random Forest globale ricevuto dal server.
        
        Args:
            parameters: alberi del Random Forest globale
            config: configurazione della valutazione
            
        Returns:
            Tuple con (loss_simulata, numero_campioni, metriche)
        """
        global rf_model, X_val, y_val

        # Imposta semi per riproducibilità
        set_reproducibility_seeds()

        print(f"[Client {client_id}] === VALUTAZIONE RF GLOBALE ===")

        if len(X_val) == 0:
            return 0.0, 0, {"accuracy": 0.0}

        # Se non ci sono parametri, usa il modello locale
        if len(parameters) == 0:
            print(f"[Client {client_id}] Nessun parametro ricevuto, uso modello locale")
            if rf_model is None:
                return 1.0, 0, {"accuracy": 0.0, "error": "no_model"}
        else:
            try:
                # Ricostruisci gli alberi dal formato Flower
                print(f"[Client {client_id}] Ricostruzione RF globale da {len(parameters)} parametri...")
                trees_data = []
                for tree_params in parameters:
                    tree_index = int(tree_params[0])
                    tree_size = int(tree_params[1])
                    accuracy = float(tree_params[2])
                    weighted_accuracy = float(tree_params[3])
                    tree_data_array = tree_params[4:].astype(np.uint8)

                    tree_dict = {
                        'tree_index': tree_index,
                        'tree_size': tree_size,
                        'accuracy': accuracy,
                        'weighted_accuracy': weighted_accuracy,
                        'tree_data': tree_data_array
                    }
                    trees_data.append(tree_dict)

                # Ricostruisci il Random Forest
                rf_model = deserialize_random_forest_trees(trees_data)
                print(f"[Client {client_id}] ✅ RF globale ricostruito con {len(trees_data)} alberi")

            except Exception as e:
                print(f"[Client {client_id}] ❌ Errore ricostruzione RF: {e}")
                import traceback
                traceback.print_exc()
                return 1.0, len(X_val), {"accuracy": 0.0, "error": f"reconstruction_failed: {str(e)}"}

        try:
            # Valutazione
            y_pred = rf_model.predict(X_val)
            y_pred_proba = rf_model.predict_proba(X_val)[:, 1]

            # Calcolo metriche
            accuracy = accuracy_score(y_val, y_pred)

            # Metriche che richiedono gestione dei casi edge
            precision = 0.0
            recall = 0.0
            auc = 0.0
            f1_score_val = 0.0
            balanced_acc = 0.0

            if len(np.unique(y_val)) > 1:
                from sklearn.metrics import precision_score, recall_score
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                auc = roc_auc_score(y_val, y_pred_proba)
                f1_score_val = f1_score(y_val, y_pred, zero_division=0)
                balanced_acc = balanced_accuracy_score(y_val, y_pred)

            # Report per classe
            report = classification_report(y_val, y_pred, target_names=["natural", "attack"], output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_val, y_pred)

            print(f"[Client {client_id}] Val Accuracy: {accuracy:.4f}")
            print(f"[Client {client_id}] Val F1: {f1_score_val:.4f}, Val Balanced Acc: {balanced_acc:.4f}")
            print(f"[Client {client_id}] Classification report (per classe):")
            print(classification_report(y_val, y_pred, target_names=["natural", "attack"], zero_division=0))
            print(f"[Client {client_id}] Confusion matrix:")
            print(f"tn: {conf_matrix[0, 0]}, fp: {conf_matrix[0, 1]}, fn: {conf_matrix[1, 0]}, tp: {conf_matrix[1, 1]}")

            # Metriche da restituire
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
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1])
            }

            # Simula una loss (Random Forest non ha loss, usiamo 1 - accuracy)
            loss = 1.0 - accuracy

            return loss, len(X_val), metrics

        except Exception as e:
            print(f"[Client {client_id}] ❌ Errore durante valutazione: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, len(X_val), {"accuracy": 0.0, "error": f"evaluation_failed: {str(e)}"}

# ============== MAIN ==============
def main():
    """
    Funzione principale per avviare il client SmartGrid Random Forest.
    """
    global client_id, rf_model, X_train, y_train, X_val, y_val, dataset_info

    # Imposta semi all'avvio del client
    set_reproducibility_seeds()

    if len(sys.argv) != 2:
        print("Usa: python clientRF.py <client_id>")
        print("Esempio: python clientRF.py 1")
        sys.exit(1)

    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("Client ID deve essere tra 1 e 13")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)

    print(f"=== AVVIO CLIENT RF {client_id} ===")

    try:
        # Carica i dati
        print(f"[Client {client_id}] Caricamento dati...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)

        # Imposta semi all'avvio del client
        set_reproducibility_seeds()

        print(f"[Client {client_id}] === RIASSUNTO CLIENT RF ===")
        print(f"[Client {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"[Client {client_id}] Distribuzione: {dataset_info['attack_ratio']*100:.1f}% attacchi")
        if ENABLE_PCA:
            print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['final_features']} (PCA attiva)")
        else:
            print(f"[Client {client_id}] Feature: {dataset_info['original_features']} → {dataset_info['final_features']} (nessuna riduzione - PCA disattiva)")
        print(f"[Client {client_id}] Modello: Random Forest con {RF_N_ESTIMATORS} alberi")
        print(f"[Client {client_id}] Connessione al server su localhost:8080...")

        # Avvia il client Flower
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridRFClient()
        )

    except Exception as e:
        print(f"[Client {client_id}] ❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()