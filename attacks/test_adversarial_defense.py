"""
attacks/test_adversarial_defense.py

Script di test per validare l'efficacia della difesa adversarial training.  

WORKFLOW COMPLETO (AGGIORNATO):
1. Addestra modello SENZA difesa (baseline)
2. Addestra modello CON difesa adversarial
3. Testa entrambi contro attacchi adversarial su validation set
4. Confronta robustezza (ASR, accuracy adversarial)

UTILIZZO:

    Eseguire dalla root del progetto:

    # Test SENZA robustezza (solo accuracy): valuta l'accuracy del modello con adversarial training sui dati puliti
    (utilizzato come test preliminare, per Verificare che adversarial training non degradi accuracy prima di testare robustezza)

    python attacks/test_adversarial_defense.py --client-id 1 --no-robustness-test

    # Test con difesa abilitata + robustezza: valuta accuracy del modello con adversarial training sui dati adversarial
    (per verificare che la difesa funzioni davvero contro attacchi adversarial)

    python attacks/test_adversarial_defense.py --client-id 1

    # TEST Confronto automatico baseline vs robusto: valuta entrambi i modelli sui dari puliti e adversarial
    (Dimostrare l'efficacia della difesa con confronto side-by-side)

    python attacks/test_adversarial_defense.py --client-id 1 --compare

    # Test baseline (senza difesa) + robustezza: valuta accuracy del modello senza adversarial training sui dati adversarial
    (Misurare vulnerabilità del modello senza difesa (punto di riferimento))
    
    python attacks/test_adversarial_defense.py --client-id 1 --disable-defense


AUTORE: Carmine Cataldo
DATA: 2025-01-25 
"""

import sys
import os
import numpy as np
import argparse
import time

# Aggiungi path per import moduli
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
# Import ART
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier

# Import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Progress bar

# Import moduli difesa
from attacks.defense_config import (
    DEFENSE_CONFIG,
    print_defense_config,
    update_defense_config,
    get_hsj_config_for_training
)
from attacks.defense_utils import (
    get_smartgrid_physical_constraints_advanced,
    apply_adaptive_constraints,
    calculate_feature_importance_for_defense
)


def load_client_data_for_test(client_id):
    """
    Carica dati di un client per testing.
    Usa STESSO preprocessing del federated. 
    
    Args:
        client_id: ID del client
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    print(f"[Test] Caricamento dati client {client_id}...")
    
    # Path dati
    data_path = os.path.join('data', 'SmartGrid', f'data{client_id}.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File non trovato: {data_path}")
    
    # Carica CSV
    df = pd.read_csv(data_path)
    
    # Separa X e y
    X = df.drop(columns=["marker"]).values
    y = (df["marker"] != "Natural").astype(int).values
    
    # Split train/val
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Preprocessing (identico al federated)
    # Pulizia inf/NaN
    X_train_clean = np.where(np.isinf(X_train_raw), np.nan, X_train_raw)
    X_val_clean = np.where(np.isinf(X_val_raw), np.nan, X_val_raw)
    
    # Imputazione
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    X_val_imputed = imputer.transform(X_val_clean)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_val = scaler.transform(X_val_imputed)
    
    print(f"[Test] ✅ Dati caricati: train={X_train.shape}, val={X_val.shape}")
    print(f"[Test] Distribuzione: {y_train.mean()*100:.1f}% attacchi")
    
    return X_train, y_train, X_val, y_val


# attacks/test_adversarial_defense.py

def local_adversarial_training_test(model_instance, X_train, y_train, X_val, y_val, client_id):
    """
    Versione TEST di adversarial training locale.
    ALLINEATA COMPLETAMENTE a clientRF.py
    
    Args:
        model_instance: Random Forest già addestrato su dati puliti
        X_train: Training set pulito
        y_train: Etichette training
        X_val: Validation set
        y_val: Etichette validation
        client_id: ID del client
        
    Returns:
        model_robust: Random Forest addestrato su dati puliti + adversarial
        success: True se completato con successo
    """
    print(f"\n[Test Client {client_id}] {'='*60}")
    print(f"[Test Client {client_id}] 🛡️ ADVERSARIAL TRAINING LOCALE (TEST)")
    print(f"[Test Client {client_id}] {'='*60}")
    
    try:
        # STEP 1: Seleziona campioni Attack
        attack_mask = (y_train == 1)
        X_attack = X_train[attack_mask]
        y_attack = y_train[attack_mask]
        
        if len(X_attack) == 0:
            print(f"[Test Client {client_id}] ⚠️ Nessun campione Attack")
            return model_instance, False
        
        print(f"[Test Client {client_id}] Campioni Attack: {len(X_attack)}")
        
        # STEP 2: Sottocampionamento
        max_adv_samples = min(DEFENSE_CONFIG['MAX_ADVERSARIAL_SAMPLES'], len(X_attack) // 2)
        
        if len(X_attack) > max_adv_samples:
            import random
            random.seed(42)
            indices = random.sample(range(len(X_attack)), max_adv_samples)
            X_attack_sub = X_attack[indices]
            y_attack_sub = y_attack[indices]
        else:
            X_attack_sub = X_attack
            y_attack_sub = y_attack
        
        print(f"[Test Client {client_id}] Sottocampionamento: {len(X_attack_sub)}")
        
        # STEP 3: Wrap modello ART
        art_classifier = SklearnClassifier(model=model_instance)
        
        # STEP 4: Configura HSJ VELOCE
        hsj_config = get_hsj_config_for_training()
        
        hsj_local = HopSkipJump(
            classifier=art_classifier,
            targeted=False,
            norm=hsj_config['norm'],
            max_iter=hsj_config['max_iter'],
            max_eval=hsj_config['max_eval'],
            init_eval=hsj_config['init_eval'],
            verbose=hsj_config['verbose']
        )
        
        print(f"[Test Client {client_id}] HSJ: max_iter={hsj_config['max_iter']}, max_eval={hsj_config['max_eval']}")
        
        # STEP 5: Genera adversarial con progress bar
        start_time = time.time()
        print(f"[Test Client {client_id}] 🔄 Generazione adversarial per {len(X_attack_sub)} campioni...")
        
        # ✅ GENERAZIONE CON PROGRESS BAR CAMPIONE PER CAMPIONE
        X_adv_list = []
        
        with tqdm(total=len(X_attack_sub), 
                  desc=f"[Test Client {client_id}] HSJ Training", 
                  unit="campioni", 
                  ncols=100,
                  colour='blue') as pbar:
            
            for i in range(len(X_attack_sub)):
                try:
                    # Genera adversarial per singolo campione
                    x_adv_i = hsj_local.generate(x=X_attack_sub[i:i+1])
                    X_adv_list.append(x_adv_i[0])
                    
                    # Aggiorna progress bar
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n[Test Client {client_id}] ⚠️ Errore campione {i+1}: {e}")
                    # Usa campione originale in caso di errore
                    X_adv_list.append(X_attack_sub[i])
                    pbar.update(1)
                    continue
        
        # Converti lista in array numpy
        X_adv = np.array(X_adv_list)
        
        elapsed = time.time() - start_time
        print(f"[Test Client {client_id}] ✅ Generazione completata in {elapsed:.1f}s ({len(X_adv)/elapsed:.2f} campioni/sec)")
        
        # STEP 6: Verifica output
        if X_adv is None or len(X_adv) == 0:
            print(f"[Test Client {client_id}] ⚠️ Generazione fallita")
            return model_instance, False
        
        # STEP 7: Vincoli fisici con percentili ESPLICITI da DEFENSE_CONFIG
        print(f"[Test Client {client_id}] Applicazione vincoli fisici SmartGrid...")
        
        # Calcola vincoli con percentili configurabili (COME clientRF)
        constraints = get_smartgrid_physical_constraints_advanced(
            X_train,
            percentile_low=DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_LOW'],   
            percentile_high=DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_HIGH']   
        )
        
        print(f"[Test Client {client_id}] Vincoli calcolati:")
        print(f"  - Range globale: [{constraints['feature_min'].min():.3f}, {constraints['feature_max'].max():.3f}]")
        print(f"  - Percentili: {DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_LOW']}-{DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_HIGH']}")
        
        # Feature importance condizionale (COME clientRF)
        if DEFENSE_CONFIG.get('USE_ADAPTIVE_CONSTRAINTS', False):
            print(f"[Test Client {client_id}] Calcolo feature importance per vincoli adattivi...")
            feature_importance = calculate_feature_importance_for_defense(
                model_instance, X_train, method='gini'
            )
        else:
            feature_importance = None
        
        # Applica vincoli (con o senza feature importance)
        X_adv_constrained = apply_adaptive_constraints(
            X_adv,
            X_attack_sub,
            constraints,
            DEFENSE_CONFIG['EPSILON'],
            feature_importance=feature_importance
        )
        
        print(f"[Test Client {client_id}] ✅ Vincoli fisici applicati")
        if feature_importance is not None:
            print(f"  - Vincoli adattivi: ABILITATI (feature importance)")
        else:
            print(f"  - Vincoli uniformi: epsilon={DEFENSE_CONFIG['EPSILON']}")
        
        y_adv = y_attack_sub
        
        # STEP 8: Data augmentation
        X_aug = np.concatenate([X_train, X_adv_constrained], axis=0)
        y_aug = np.concatenate([y_train, y_adv], axis=0)
        
        # Shuffle
        indices_shuffle = np.random.permutation(len(X_aug))
        X_aug = X_aug[indices_shuffle]
        y_aug = y_aug[indices_shuffle]
        
        print(f"[Test Client {client_id}] Dataset: {len(X_train)} → {len(X_aug)}")
        
        # STEP 9: Riaddestra
        model_robust = RandomForestClassifier(
            n_estimators=65,
            criterion='entropy',
            max_features='sqrt',
            class_weight='balanced',
            random_state=42 + client_id,
            n_jobs=-1
        )
        
        model_robust.fit(X_aug, y_aug)
        
        # STEP 10: Valuta
        if X_val is not None:
            val_acc_clean = model_instance.score(X_val, y_val)
            val_acc_robust = model_robust.score(X_val, y_val)
            
            print(f"[Test Client {client_id}] 📊 Validation:")
            print(f"  Clean:  {val_acc_clean:.4f}")
            print(f"  Robust: {val_acc_robust:.4f}")
            print(f"  Δ:      {val_acc_robust - val_acc_clean:+.4f}")
        
        print(f"[Test Client {client_id}] ✅ ADVERSARIAL TRAINING COMPLETATO")
        
        return model_robust, True
        
    except Exception as e:
        print(f"[Test Client {client_id}] ❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return model_instance, False


def test_model_robustness(model, X_val, y_val, client_id, model_name="Model"):
    """
    NUOVA FUNZIONE: Testa robustezza del modello generando adversarial sul validation set. 
    
    WORKFLOW:
    1. Seleziona campioni Attack dal validation
    2. Genera adversarial con HSJ (config identica al training)
    3. Applica vincoli fisici
    4. Calcola metriche robustezza (ASR, accuracy adversarial)
    
    Args:
        model: RandomForestClassifier da testare
        X_val: Validation set
        y_val: Etichette validation
        client_id: ID client (per logging)
        model_name: Nome modello (per logging)
        
    Returns:
        Dictionary con metriche robustezza:
            - accuracy_adversarial: Accuracy su dati adversarial
            - asr: Attack Success Rate
            - robustness_score: Metrica combinata (1 - ASR)
            - successful_evasions: Numero evasioni riuscite
    """
    print(f"\n[Test {model_name}] " + "="*60)
    print(f"[Test {model_name}] 🔬 TEST ROBUSTEZZA SUL VALIDATION SET")
    print(f"[Test {model_name}] " + "="*60)
    
    # ========== STEP 1: SELEZIONA CAMPIONI ATTACK ==========
    attack_mask = (y_val == 1)
    X_val_attacks = X_val[attack_mask]
    y_val_attacks = y_val[attack_mask]
    
    if len(X_val_attacks) == 0:
        print(f"[Test {model_name}] ⚠️ Nessun campione Attack in validation")
        return {
            'accuracy_adversarial': 0.0,
            'asr': 0.0,
            'robustness_score': 0.0,
            'successful_evasions': 0
        }
    
    print(f"[Test {model_name}] Campioni Attack validation: {len(X_val_attacks)}")
    
    # ========== STEP 2: CONFIGURA HSJ PER TEST ==========
    # ✅ USA STESSA CONFIG DEL TRAINING (per fairness)
    art_classifier = SklearnClassifier(model=model)
    hsj_config = get_hsj_config_for_training()
    
    hsj_test = HopSkipJump(
        classifier=art_classifier,
        targeted=False,
        norm=hsj_config['norm'],
        max_iter=hsj_config['max_iter'],
        max_eval=hsj_config['max_eval'],
        init_eval=hsj_config['init_eval'],
        verbose=hsj_config['verbose']
    )
    
    print(f"[Test {model_name}] HSJ configurato: max_iter={hsj_config['max_iter']}, max_eval={hsj_config['max_eval']}")
    
    # ========== STEP 3: GENERA ADVERSARIAL SU VALIDATION ==========
    print(f"\n[Test {model_name}] 🔄 Generazione adversarial su validation...")
    
    start_time = time.time()
    X_val_adv_list = []
    
    # ✅ Progress bar per generazione
    with tqdm(total=len(X_val_attacks), 
              desc=f"[Test {model_name}] HSJ Validation", 
              unit="campioni", 
              ncols=100,
              colour='cyan') as pbar:
        
        for i in range(len(X_val_attacks)):
            try:
                x_adv_i = hsj_test.generate(x=X_val_attacks[i:i+1])
                X_val_adv_list.append(x_adv_i[0])
                pbar.update(1)
            except Exception as e:
                print(f"\n[Test {model_name}] ⚠️ Errore campione {i+1}: {e}")
                X_val_adv_list.append(X_val_attacks[i])  # Usa originale
                pbar.update(1)
    
    X_val_adv = np.array(X_val_adv_list)
    elapsed = time.time() - start_time
    
    print(f"\n[Test {model_name}] ✅ Generati {len(X_val_adv)} adversarial in {elapsed:.1f}s")
    
    # ========== STEP 4: APPLICA VINCOLI FISICI ==========
    print(f"\n[Test {model_name}] Applicazione vincoli fisici...")
    
    constraints = get_smartgrid_physical_constraints_advanced(
        X_val,
        percentile_low=DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_LOW'],
        percentile_high=DEFENSE_CONFIG['CONSTRAINT_PERCENTILE_HIGH']
    )
    
    X_val_adv_constrained = apply_adaptive_constraints(
        X_val_adv,
        X_val_attacks,
        constraints,
        DEFENSE_CONFIG['EPSILON'],
        feature_importance=None  # No adaptive per test
    )
    
    # ========== STEP 5: CALCOLA METRICHE ROBUSTEZZA ==========
    print(f"\n[Test {model_name}] 📊 Calcolo metriche robustezza...")
    
    # Predizioni
    y_pred_clean = model.predict(X_val_attacks)
    y_pred_adv = model.predict(X_val_adv_constrained)
    
    # ASR (Attack Success Rate)
    # Successo = Campione originariamente Attack (pred=1) → classificato Natural (pred=0)
    evasion_mask = (y_pred_clean == 1) & (y_pred_adv == 0)
    successful_evasions = np.sum(evasion_mask)
    asr = successful_evasions / len(X_val_attacks)
    
    # Accuracy su adversarial
    acc_adv = accuracy_score(y_val_attacks, y_pred_adv)
    
    # Robustness score (1. 0 = perfettamente robusto, 0.0 = completamente vulnerabile)
    robustness_score = 1.0 - asr
    
    # Perturbazione media
    perturbation = X_val_adv_constrained - X_val_attacks
    l2_mean = np.mean(np.linalg.norm(perturbation, axis=1))
    
    # ========== STEP 6: STAMPA RISULTATI ==========
    print(f"\n[Test {model_name}] " + "="*60)
    print(f"[Test {model_name}] 📊 RISULTATI ROBUSTEZZA:")
    print(f"[Test {model_name}] " + "="*60)
    print(f"[Test {model_name}] ASR (Attack Success Rate): {asr*100:.2f}%")
    print(f"[Test {model_name}]   Evasioni riuscite: {successful_evasions}/{len(X_val_attacks)}")
    print(f"[Test {model_name}] Accuracy adversarial: {acc_adv:.4f}")
    print(f"[Test {model_name}] Robustness score: {robustness_score:.4f}")
    print(f"[Test {model_name}] Perturbazione L2 media: {l2_mean:.6f}")
    print(f"[Test {model_name}] " + "="*60 + "\n")
    
    return {
        'accuracy_adversarial': float(acc_adv),
        'asr': float(asr),
        'robustness_score': float(robustness_score),
        'successful_evasions': int(successful_evasions),
        'total_attacks_tested': int(len(X_val_attacks)),
        'l2_perturbation_mean': float(l2_mean)
    }


def test_defense_single_client(client_id, enable_defense=True, test_robustness=True):
    """
    Testa difesa su singolo client CON VALUTAZIONE ROBUSTEZZA. 
    
    ✅ MODIFICATA: Aggiunto parametro test_robustness per generare adversarial su validation
    
    Args:
        client_id: ID del client
        enable_defense: Se True, applica adversarial training
        test_robustness: Se True, genera adversarial su validation per testare robustezza
        
    Returns:
        Dictionary con risultati COMPLETI (clean + adversarial)
    """
    print("\n" + "="*80)
    print(f"TEST DIFESA ADVERSARIAL - CLIENT {client_id}")
    print(f"Modalità: {'CON DIFESA' if enable_defense else 'SENZA DIFESA (Baseline)'}")
    print(f"Test robustezza: {'ABILITATO' if test_robustness else 'DISABILITATO'}")
    print("="*80)
    
    # Carica dati
    X_train, y_train, X_val, y_val = load_client_data_for_test(client_id)
    
    # Crea modello baseline
    print(f"\n[Test] Creazione Random Forest baseline...")
    model = RandomForestClassifier(
        n_estimators=65,
        criterion='entropy',
        max_features='sqrt',
        class_weight='balanced',
        random_state=42 + client_id,
        n_jobs=-1
    )
    
    # ========== STEP 1: TRAINING SU DATI PULITI ==========
    print(f"\n[Test] Training baseline su dati puliti...")
    model.fit(X_train, y_train)
    
    # Valuta baseline su dati puliti
    acc_clean_baseline = model.score(X_val, y_val)
    print(f"[Test] Accuracy baseline (dati puliti): {acc_clean_baseline:.4f}")
    
    # ========== STEP 2: ADVERSARIAL TRAINING (se abilitato) ==========
    if enable_defense:
        print(f"\n[Test] Applicazione adversarial training...")
        
        model_robust, success = local_adversarial_training_test(
            model, X_train, y_train, X_val, y_val, client_id
        )
        
        if success:
            model = model_robust  # ✅ Sostituisci con modello robusto
            acc_robust_clean = model.score(X_val, y_val)
            print(f"[Test] Accuracy robusto (dati puliti): {acc_robust_clean:.4f}")
            print(f"[Test] Δ Accuracy puliti: {acc_robust_clean - acc_clean_baseline:+.4f}")
        else:
            print(f"[Test] ⚠️ Adversarial training fallito, uso baseline")
            success = False
    else:
        success = False
    
    # ========== STEP 3: TEST ROBUSTEZZA (SE ABILITATO) ==========
    results = {
        'client_id': client_id,
        'defense_enabled': enable_defense,
        'defense_success': success,
        'accuracy_clean': acc_clean_baseline if not enable_defense else model.score(X_val, y_val),
    }
    
    if test_robustness:
        print("\n" + "="*80)
        print("STEP 3: TEST ROBUSTEZZA CONTRO ATTACCHI ADVERSARIAL")
        print("="*80)
        
        # ✅ NUOVO: Genera adversarial examples sul VALIDATION SET
        results_robustness = test_model_robustness(
            model, 
            X_val, 
            y_val, 
            client_id,
            model_name="Robusto" if enable_defense else "Baseline"
        )
        
        # Aggiungi metriche robustezza
        results.update(results_robustness)
    
    # ========== STEP 4: STAMPA RISULTATI FINALI ==========
    print(f"\n[Test] ✅ Test completato")
    print(f"  Accuracy su dati puliti: {results['accuracy_clean']:.4f}")
    
    if test_robustness:
        print(f"  Accuracy su dati adversarial: {results['accuracy_adversarial']:.4f}")
        print(f"  ASR (Attack Success Rate): {results['asr']*100:.2f}%")
        print(f"  Robustness score: {results['robustness_score']:.4f}")
    
    return results


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Test Adversarial Defense su Random Forest locale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:

  # Test CON difesa + robustezza
  %(prog)s --client-id 1

  # Test SENZA difesa (baseline) + robustezza
  %(prog)s --client-id 1 --disable-defense

  # Test SOLO accuracy (no robustezza)
  %(prog)s --client-id 1 --no-robustness-test

  # ✅ Confronto automatico baseline vs robusto
  %(prog)s --client-id 1 --compare

CONFIGURAZIONE:
  - File config: attacks/defense_config.py
  - Epsilon default: 0.05
  - Max samples: 500
  - HSJ config: max_iter=10, max_eval=500 (veloce per FL)
        """
    )
    
    parser.add_argument(
        '--client-id',
        type=int,
        required=True,
        help='ID del client da testare (1-15)'
    )
    
    parser.add_argument(
        '--disable-defense',
        action='store_true',
        help='Disabilita difesa adversarial (test baseline)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.05,
        help='Epsilon per adversarial training (default: 0.05)'
    )
    
    parser.add_argument(
        '--no-robustness-test',
        action='store_true',
        help='Salta test robustness (solo accuracy)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Confronta automaticamente baseline vs robusto'
    )
    
    args = parser.parse_args()
    
    # Aggiorna configurazione se epsilon custom
    if args.epsilon != 0.05:
        update_defense_config({'EPSILON': args.epsilon})
    
    # Stampa configurazione
    print_defense_config()
    
    # ========== MODALITÀ CONFRONTO ==========
    if args.compare:
        print("\n" + "="*80)
        print("MODALITÀ CONFRONTO: BASELINE VS ROBUSTO")
        print("="*80)
        
        # Test baseline (NO defense)
        print("\n" + "🔴 "*20)
        print("TEST 1: MODELLO BASELINE (SENZA DIFESA)")
        print("🔴 "*20)
        
        results_baseline = test_defense_single_client(
            client_id=args.client_id,
            enable_defense=False,
            test_robustness=not args.no_robustness_test
        )
        
        # Test robusto (CON defense)
        print("\n" + "🟢 "*20)
        print("TEST 2: MODELLO ROBUSTO (CON DIFESA)")
        print("🟢 "*20)
        
        results_robust = test_defense_single_client(
            client_id=args.client_id,
            enable_defense=True,
            test_robustness=not args.no_robustness_test
        )
        
        # ========== CONFRONTO FINALE ==========
        print("\n" + "="*80)
        print("📊 CONFRONTO FINALE: BASELINE VS ROBUSTO")
        print("="*80)
        
        print(f"\n{'Metrica':<30} {'Baseline':<15} {'Robusto':<15} {'Δ (Robusto-Baseline)':<25}")
        print("-"*85)
        
        # Accuracy puliti
        acc_baseline = results_baseline['accuracy_clean']
        acc_robust = results_robust['accuracy_clean']
        print(f"{'Accuracy (dati puliti)':<30} {acc_baseline:<15.4f} {acc_robust:<15.4f} {acc_robust-acc_baseline:+.4f}")
        
        if not args.no_robustness_test:
            # Accuracy adversarial
            acc_adv_baseline = results_baseline.get('accuracy_adversarial', 0.0)
            acc_adv_robust = results_robust.get('accuracy_adversarial', 0.0)
            print(f"{'Accuracy (dati adversarial)':<30} {acc_adv_baseline:<15.4f} {acc_adv_robust:<15.4f} {acc_adv_robust-acc_adv_baseline:+.4f}")
            
            # ASR
            asr_baseline = results_baseline.get('asr', 1.0)
            asr_robust = results_robust.get('asr', 1.0)
            print(f"{'ASR (Attack Success Rate)':<30} {asr_baseline*100:<15.2f}% {asr_robust*100:<15.2f}% {(asr_robust-asr_baseline)*100:+.2f}%")
            
            # Robustness score
            rob_baseline = results_baseline.get('robustness_score', 0.0)
            rob_robust = results_robust.get('robustness_score', 0.0)
            print(f"{'Robustness score':<30} {rob_baseline:<15.4f} {rob_robust:<15.4f} {rob_robust-rob_baseline:+.4f}")
        
        print("-"*85)
        
        # Interpretazione
        print(f"\n💡 INTERPRETAZIONE:")
        
        if not args.no_robustness_test:
            improvement_asr = (asr_baseline - asr_robust) / asr_baseline * 100 if asr_baseline > 0 else 0
            
            if improvement_asr > 30:
                print(f"   ✅ DIFESA MOLTO EFFICACE: ASR ridotto del {improvement_asr:.1f}%")
            elif improvement_asr > 10:
                print(f"   ✅ DIFESA EFFICACE: ASR ridotto del {improvement_asr:.1f}%")
            elif improvement_asr > 0:
                print(f"   ⚠️ DIFESA PARZIALE: ASR ridotto del {improvement_asr:.1f}%")
            else:
                print(f"   ❌ DIFESA INEFFICACE: Nessun miglioramento robustezza")
            
            # Trade-off accuracy
            acc_loss = acc_baseline - acc_robust
            if acc_loss > 0.05:
                print(f"   ⚠️ Trade-off SIGNIFICATIVO: Perdita accuracy puliti = {acc_loss:.4f}")
            elif acc_loss > 0:
                print(f"   ✅ Trade-off ACCETTABILE: Perdita accuracy puliti = {acc_loss:.4f}")
            else:
                print(f"   ✅ NESSUN Trade-off: Accuracy puliti mantenuta/migliorata")
        
        print("="*80)
        
    else:
        # ========== MODALITÀ SINGOLA ==========
        enable_defense = not args.disable_defense
        
        results = test_defense_single_client(
            client_id=args.client_id,
            enable_defense=enable_defense,
            test_robustness=not args.no_robustness_test
        )
        
        # Stampa risultati
        print("\n" + "="*80)
        print("RISULTATI FINALI")
        print("="*80)
        print(f"Client: {results['client_id']}")
        print(f"Difesa: {'ABILITATA' if results['defense_enabled'] else 'DISABILITATA'}")
        print(f"Accuracy (dati puliti): {results['accuracy_clean']:.4f}")
        
        if not args.no_robustness_test:
            print(f"Accuracy (dati adversarial): {results.get('accuracy_adversarial', 0.0):.4f}")
            print(f"ASR: {results.get('asr', 0.0)*100:.2f}%")
            print(f"Robustness score: {results.get('robustness_score', 0.0):.4f}")
        
        print("="*80)


if __name__ == "__main__":
    main()