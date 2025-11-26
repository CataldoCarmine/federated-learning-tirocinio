"""
attacks/defense_config.py

Configurazione centralizzata per Adversarial Training Defense.

Questo file contiene tutti i parametri configurabili per la difesa,
permettendo di modificarli in un unico posto senza toccare il codice.

UTILIZZO:
    from attacks.defense_config import DEFENSE_CONFIG
    
    if DEFENSE_CONFIG['ENABLE_ADVERSARIAL_TRAINING']:
        # Esegui adversarial training
        pass

AUTORE: Carmine Cataldo
DATA: 2025-01-24
"""

# ============== CONFIGURAZIONE DIFESA ADVERSARIAL ==============

DEFENSE_CONFIG = {
    # ========== ABILITA/DISABILITA DIFESA ==========
    'ENABLE_ADVERSARIAL_TRAINING': True,  # Flag globale ON/OFF
    
    # ========== PARAMETRI ADVERSARIAL TRAINING ==========
    
    # Budget perturbazione (epsilon)
    'EPSILON': 0.05,  # Perturbazione massima consentita
    
    # Numero massimo campioni Attack da usare per adversarial training
    'MAX_ADVERSARIAL_SAMPLES': 500,  # Limita per velocità
    
    # PER MIGLIORAMENTI FUTURI: Rapporto dati adversarial vs puliti: controlla la proporzione di esempi adversarial 
    # (es 90% adversarial: massima robustezza ma possibile overfitting su adversarial, 10% puliti: preserva accuracy pulita, robustezza minimale)
    'ADVERSARIAL_RATIO': 0.5,  # 50% adversarial, 50% puliti
    
    # ========== CONFIGURAZIONE HOPSKIPJUMP VELOCE ==========
    
    # HopSkipJump per federated learning (parametri ridotti)
    'HSJ_MAX_ITER': 10,      # Iterazioni (ridotte da 50)
    'HSJ_MAX_EVAL': 500,     # Query per campione (ridotte da 5000)
    'HSJ_INIT_EVAL': 20,     # Query inizializzazione
    'HSJ_NORM': 2,           # Norma L2
    'HSJ_VERBOSE': False,    # Silenzioso per non intasare log
    
    # ========== VINCOLI FISICI SMARTGRID ==========
    
    # Percentili per calcolo vincoli
    'CONSTRAINT_PERCENTILE_LOW': 0.1,   # Percentile inferiore
    'CONSTRAINT_PERCENTILE_HIGH': 99.9,  # Percentile superiore
    
    # PER MIGLIORAMENTI FUTURI: Permette di aumentare leggermente l'epsilon per i vincoli fisici rispetto all'epsilon base, Questo evita che i vincoli fisici "stringano troppo" le perturbazioni HSJ
    # (Epsilon base (0.01): Controllo perturbazione HSJ, Epsilon vincoli (0.01 * 1.5 = 0.015): Permette piccolo margine per vincoli fisici)
    'CONSTRAINT_EPSILON_MULTIPLIER': 1.5,  # Permette epsilon * 1.5 per vincoli

    # Vincoli adattivi
    'USE_ADAPTIVE_CONSTRAINTS': True,  # True = epsilon diverso per feature importante vs non importante
    
    # ========== STRATEGIA ADVERSARIAL TRAINING ==========
    
    # PER MIGLIORAMENTI FUTURI: Metodo di aggregazione delle perturbazioni, Se si generano multiple perturbazioni per lo stesso campione (es. con HSJ randomizzato), decidere come aggregarle
    # (mean: media semplice e veloce, median: mediana per robustezza contro outlier e più stabile, weighted_mean: peso basato su difficoltà della perturbazione)
    'AGGREGATION_METHOD': 'median',  # 'mean', 'median', 'weighted_mean'
    
    # PER MIGLIORAMENTI FUTURI: Riaddestramento modello
    # (full: modello completamente adatto ai dati augmentati e robustezza massima, ma lento e perde conoscenza sui dati puliti; incremental: veloce e preserva conoscenza originale, ma meno robusto)
    'RETRAIN_STRATEGY': 'full',  # 'full': riaddestra da zero su dataset puliti + adversarial, 'incremental': Mantieni alberi originali, aggiungi nuovi alberi addestrati solo su adversarial
    
    # PER MIGLIORAMENTI FUTURI: fa parte del miglioramento sopra, Numero alberi extra per adversarial training (se incremental)
    'ADVERSARIAL_EXTRA_TREES': 20,
    
    # ========== OTTIMIZZAZIONI PERFORMANCE ==========
    
    # PER MIGLIORAMENTI FUTURI: Parallelizza la generazione adversarial su più core CPU per accelerare il processo
    # (Generazione sequenziale (1 campione alla volta) → LENTO, Genera 50 campioni in parallelo su 4 worker → 4x PIÙ VELOCE)
    'USE_MULTIPROCESSING': False,  # Disabilitato per compatibilità Flower
    
    # PER MIGLIORAMENTI FUTURI: fa parte del miglioramento sopra
    'N_WORKERS': 4,
    
    # PER MIGLIORAMENTI FUTURI: fa parte del miglioramento sopra
    'BATCH_SIZE': 50,  # Genera 50 campioni alla volta
    
    # ========== LOGGING E DEBUG ==========
    
    # PER MIGLIORAMENTI FUTURI: Controlla la quantità di output stampato durante adversarial training
    # (Livello 0 (silenzioso): Solo errori → Produzione, Livello 1 (normale): Metriche principali → Default, Livello 2 (debug): Tutto (HSJ verbose, statistiche batch, timing) → Sviluppo)
    'VERBOSE_LEVEL': 1,  # 0: silenzioso, 1: normale, 2: debug
    
    # PER MIGLIORAMENTI FUTURI: Salva esempi adversarial per analisi
    # (Analisi post-hoc: Studiare pattern nelle perturbazioni, Debugging: Verificare se gli adversarial sono realistici, Riuso: Evitare di rigenerare gli stessi adversarial in run successive)
    'SAVE_ADVERSARIAL_EXAMPLES': False,  # Può occupare molto spazio
    
    # PER MIGLIORAMENTI FUTURI: fa parte del miglioramento sopra, Directory per salvare esempi
    'ADVERSARIAL_EXAMPLES_DIR': 'attacks/adversarial_examples',
    
    # ========== VALIDAZIONE E TESTING ==========
    
    # PER MIGLIORAMENTI FUTURI: Testa automaticamente l'efficacia della difesa alla fine dell'adversarial training generando adversarial sul modello robusto e calcolando l'ASR.
    # (feedback immediato sull'efficacia della difesa, identifica configurazioni non efficaci, invia ASR al server per confronto cross-client)
    'VALIDATE_DEFENSE': True,
    
    # PER MIGLIORAMENTI FUTURI: fa parte del miglioramento sopra,Usa subset per validazione rapida
    'VALIDATION_SUBSET_SIZE': 200,
}


# ========== FUNZIONI UTILITY PER CONFIGURAZIONE ==========

def get_defense_config():
    """
    Restituisce configurazione difesa corrente.
    
    Returns:
        Dictionary con configurazione completa
    """
    return DEFENSE_CONFIG.copy()


def update_defense_config(updates):
    """
    Aggiorna configurazione difesa.
    
    Args:
        updates: Dictionary con parametri da aggiornare
        
    Example:
        >>> update_defense_config({'EPSILON': 0.05, 'MAX_ADVERSARIAL_SAMPLES': 1000})
    """
    global DEFENSE_CONFIG
    DEFENSE_CONFIG.update(updates)
    print(f"[Config] Configurazione difesa aggiornata: {list(updates.keys())}")


def print_defense_config():
    """Stampa configurazione corrente in formato leggibile."""
    print("\n" + "="*80)
    print("CONFIGURAZIONE ADVERSARIAL TRAINING DEFENSE")
    print("="*80)
    
    print("\n🛡️ STATO DIFESA:")
    print(f"  Enabled: {DEFENSE_CONFIG['ENABLE_ADVERSARIAL_TRAINING']}")
    
    if DEFENSE_CONFIG['ENABLE_ADVERSARIAL_TRAINING']:
        print("\n⚙️ PARAMETRI ADVERSARIAL TRAINING:")
        print(f"  Epsilon: {DEFENSE_CONFIG['EPSILON']}")
        print(f"  Max samples: {DEFENSE_CONFIG['MAX_ADVERSARIAL_SAMPLES']}")
        print(f"  Adversarial ratio: {DEFENSE_CONFIG['ADVERSARIAL_RATIO']*100:.0f}%")
        
        print("\n🔧 CONFIGURAZIONE HOPSKIPJUMP:")
        print(f"  Max iterations: {DEFENSE_CONFIG['HSJ_MAX_ITER']}")
        print(f"  Max evaluations: {DEFENSE_CONFIG['HSJ_MAX_EVAL']}")
        print(f"  Init evaluations: {DEFENSE_CONFIG['HSJ_INIT_EVAL']}")
        
        print("\n📊 STRATEGIA:")
        print(f"  Aggregation: {DEFENSE_CONFIG['AGGREGATION_METHOD']}")
        print(f"  Retrain: {DEFENSE_CONFIG['RETRAIN_STRATEGY']}")
        print(f"  Validation: {DEFENSE_CONFIG['VALIDATE_DEFENSE']}")
    
    print("\n" + "="*80 + "\n")


def get_hsj_config_for_training():
    """
    Estrae configurazione HSJ per adversarial training.
    
    Returns:
        Dictionary con parametri HSJ
    """
    return {
        'max_iter': DEFENSE_CONFIG['HSJ_MAX_ITER'],
        'max_eval': DEFENSE_CONFIG['HSJ_MAX_EVAL'],
        'init_eval': DEFENSE_CONFIG['HSJ_INIT_EVAL'],
        'norm': DEFENSE_CONFIG['HSJ_NORM'],
        'verbose': DEFENSE_CONFIG['HSJ_VERBOSE']
    }


# ========== VALIDAZIONE CONFIGURAZIONE ==========

def validate_defense_config():
    """
    Valida configurazione per evitare errori.
    
    Raises:
        ValueError: Se configurazione non valida
    """
    # Valida epsilon
    if DEFENSE_CONFIG['EPSILON'] <= 0 or DEFENSE_CONFIG['EPSILON'] > 1:
        raise ValueError(f"EPSILON deve essere in (0, 1], ricevuto: {DEFENSE_CONFIG['EPSILON']}")
    
    # Valida max samples
    if DEFENSE_CONFIG['MAX_ADVERSARIAL_SAMPLES'] < 10:
        raise ValueError(f"MAX_ADVERSARIAL_SAMPLES troppo basso: {DEFENSE_CONFIG['MAX_ADVERSARIAL_SAMPLES']}")
    
    # Valida HSJ config
    if DEFENSE_CONFIG['HSJ_MAX_ITER'] < 1:
        raise ValueError(f"HSJ_MAX_ITER deve essere >= 1")
    
    if DEFENSE_CONFIG['HSJ_MAX_EVAL'] < DEFENSE_CONFIG['HSJ_INIT_EVAL']:
        raise ValueError(f"HSJ_MAX_EVAL deve essere >= HSJ_INIT_EVAL")
    
    print("[Config] ✅ Configurazione difesa validata")


# Valida automaticamente al caricamento modulo
if DEFENSE_CONFIG['ENABLE_ADVERSARIAL_TRAINING']:
    validate_defense_config()