"""
attacks/test_adversarial_defense.py

Script di test per validare l'efficacia della difesa adversarial training.

WORKFLOW:
1. Addestra modello SENZA difesa (baseline)
2. Addestra modello CON difesa adversarial
3. Testa entrambi contro attacchi
4. Confronta robustezza

UTILIZZO:
    # Test con difesa abilitata
    python attacks/test_adversarial_defense.py --client-id 1

    # Test baseline (senza difesa)
    python attacks/test_adversarial_defense.py --client-id 1 --disable-defense

    # Test con epsilon custom
    python attacks/test_adversarial_defense.py --client-id 1 --epsilon 0.05

AUTORE: Carmine Cataldo
DATA: 2025-01-24
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

# Import moduli difesa
from attacks.defense_config import (
    DEFENSE_CONFIG,
    print_defense_config,
    update_defense_config,
    get_hsj_config_for_training
)
from attacks.defense_utils import (
    get_smartgrid_physical_constraints_advanced,
    apply_adaptive_constraints
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


def local_adversarial_training_test(model_instance, X_train, y_train, X_val, y_val, client_id):
    """
    Versione TEST di adversarial training locale.
    Identica alla versione in clientRF.py ma standalone.
    
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
        
        # STEP 5: Genera adversarial
        start_time = time.time()
        print(f"[Test Client {client_id}] 🔄 Generazione adversarial...")
        
        X_adv = hsj_local.generate(x=X_attack_sub)
        
        elapsed = time.time() - start_time
        print(f"[Test Client {client_id}] ✅ Generati in {elapsed:.1f}s")
        
        # STEP 6: Verifica output
        if X_adv is None or len(X_adv) == 0:
            print(f"[Test Client {client_id}] ⚠️ Generazione fallita")
            return model_instance, False
        
        # STEP 7: Vincoli fisici
        constraints = get_smartgrid_physical_constraints_advanced(X_train)
        
        X_adv_constrained = apply_adaptive_constraints(
            X_adv, X_attack_sub, constraints, DEFENSE_CONFIG['EPSILON']
        )
        
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


def test_defense_single_client(client_id, enable_defense=True):
    """
    Testa difesa su singolo client.
    
    Args:
        client_id: ID del client
        enable_defense: Se True, applica adversarial training
        
    Returns:
        Dictionary con risultati
    """
    print("\n" + "="*80)
    print(f"TEST DIFESA ADVERSARIAL - CLIENT {client_id}")
    print(f"Modalità: {'CON DIFESA' if enable_defense else 'SENZA DIFESA (Baseline)'}")
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
    
    # Addestra su dati puliti
    print(f"\n[Test] Training su dati puliti...")
    model.fit(X_train, y_train)
    
    acc_clean = model.score(X_val, y_val)
    print(f"[Test] Accuracy baseline: {acc_clean:.4f}")
    
    # Adversarial training (se abilitato)
    if enable_defense:
        print(f"\n[Test] Applicazione adversarial training...")
        
        model_robust, success = local_adversarial_training_test(
            model, X_train, y_train, X_val, y_val, client_id
        )
        
        if success:
            model = model_robust
            acc_robust = model.score(X_val, y_val)
            print(f"[Test] Accuracy dopo difesa: {acc_robust:.4f}")
        else:
            print(f"[Test] ⚠️ Adversarial training fallito")
    
    # Valutazione finale
    final_acc = model.score(X_val, y_val)
    
    results = {
        'client_id': client_id,
        'defense_enabled': enable_defense,
        'accuracy_clean': acc_clean,
        'accuracy_final': final_acc,
        'defense_success': enable_defense and success if enable_defense else False
    }
    
    print(f"\n[Test] ✅ Test completato")
    print(f"  Accuracy finale: {final_acc:.4f}")
    
    return results


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Test Adversarial Defense su Random Forest locale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:

  # Test con difesa abilitata (default)
  %(prog)s --client-id 1

  # Test baseline senza difesa
  %(prog)s --client-id 1 --disable-defense

  # Test con epsilon custom
  %(prog)s --client-id 1 --epsilon 0.05

CONFIGURAZIONE:
  - File config: attacks/defense_config.py
  - Epsilon default: 0.01
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
        default=0.01,
        help='Epsilon per adversarial training (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Aggiorna configurazione se epsilon custom
    if args.epsilon != 0.01:
        update_defense_config({'EPSILON': args.epsilon})
    
    # Stampa configurazione
    print_defense_config()
    
    # Esegui test
    enable_defense = not args.disable_defense
    
    results = test_defense_single_client(
        client_id=args.client_id,
        enable_defense=enable_defense
    )
    
    # Stampa risultati finali
    print("\n" + "="*80)
    print("RISULTATI FINALI")
    print("="*80)
    print(f"Client: {results['client_id']}")
    print(f"Difesa: {'ABILITATA' if results['defense_enabled'] else 'DISABILITATA'}")
    print(f"Accuracy clean: {results['accuracy_clean']:.4f}")
    print(f"Accuracy final: {results['accuracy_final']:.4f}")
    
    if results['defense_enabled']:
        delta = results['accuracy_final'] - results['accuracy_clean']
        print(f"Δ Accuracy: {delta:+.4f}")
        
        if delta >= 0:
            print(f"\n✅ Difesa EFFICACE: Accuracy mantenuta/migliorata")
        else:
            print(f"\n⚠️ Trade-off: Leggera perdita accuracy per robustezza")
    
    print("="*80)


if __name__ == "__main__":
    main()