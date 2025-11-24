"""
attacks/hsj_graybox_transfer.py

Attacco Gray-Box Transfer con HopSkipJump su modello surrogato Random Forest.

CORREZIONE: Rimosso parametro clip_values (non supportato in ART recenti).

SCENARIO:
Attaccante con conoscenza parziale: non ha accesso al modello federato globale,
ma può addestrare un modello surrogato su dati pubblici SmartGrid (client 7, 11).

STRATEGIA:

FASE 1: ADDESTRAMENTO SURROGATO
- Usa dati pubblici client 7 e 11
- Addestra Random Forest surrogato (50 alberi, parametri identici al target)
- Valuta performance surrogato

FASE 2: ATTACCO HSJ SUL SURROGATO
- Genera adversarial examples sul surrogato usando HSJ
- Obiettivo: Evadere surrogato (Attack → Natural)

FASE 3: TRANSFER AL TARGET FEDERATO
- Testa esempi adversarial generati sul surrogato contro target federato
- Misura transferability: quanti esempi si trasferiscono?

TRANSFERABILITY:
- Misura quanto sono trasferibili gli esempi adversarial
- Formula: transfer_rate = (evasioni su target) / (evasioni su surrogato)
- Se alta (>30%) → vulnerabilità condivisa tra modelli
- Se bassa (<10%) → diversità federata protegge

UTILIZZO:
    python attacks/hsj_graybox_transfer.py \
        --target-model-path federated/SmartGrid/models/federated_rf_global_20251121_024044.pkl \
        --surrogate-clients 7 11 \
        --max-iter 50 \
        --max-eval 5000 \
        --save-results

    opppure:
        python attacks/hsj_graybox_transfer.py \
        --target-model-path federated/SmartGrid/models/federated_rf_global_20251121_024044.pkl \
        --verbose \
        --save-results

AUTORE: Carmine Cataldo
DATA: 2025-01-23 (Aggiornato: 2025-01-24 - Rimosso clip_values)
"""

import numpy as np
import sys
import os
import argparse
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # ✅ AGGIUNTO: Progress bar al top degli import

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import ART
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier

# Import moduli custom
from attacks.utils import (
    load_federated_model,
    load_test_data_from_clients,
    apply_preprocessing_pipeline,
    get_smartgrid_physical_constraints,
    apply_physical_constraints,
    select_attack_samples,
    set_reproducibility_seeds
)
from attacks.evaluation import (
    evaluate_attack,
    print_attack_report,
    save_attack_results
)


def train_surrogate_model(client_ids=[7, 11], n_estimators=50, random_state=42):
    """
    Addestra modello Random Forest surrogato su dati pubblici.
    
    CONFIGURAZIONE SURROGATO:
    - N. alberi: 50 (più semplice del target con 100 alberi)
    - Parametri: identici al target (entropy, sqrt, balanced)
    - Dati: client 7 e 11 (pubblici, NO test set)
    
    Args:
        client_ids: Client per training surrogato (default: [7, 11])
        n_estimators: Numero alberi (default: 50)
        random_state: Seed (default: 42)
        
    Returns:
        surrogate_model: Random Forest surrogato addestrato
        surrogate_info: Dictionary con informazioni training
    """
    print("\n" + "="*80)
    print("ADDESTRAMENTO MODELLO SURROGATO RANDOM FOREST")
    print("="*80)
    print(f"Client training: {client_ids}")
    print(f"N. estimatori: {n_estimators}")
    print(f"Random state: {random_state}")
    print("="*80)
    
    set_reproducibility_seeds(random_state)
    
    # Carica dati client surrogato
    print(f"\n[Surrogate] Caricamento dati client {client_ids}...")
    X_surr_raw, y_surr, surr_info = load_test_data_from_clients(client_ids=client_ids)
    
    # Preprocessing (identico al target)
    print(f"\n[Surrogate] Applicazione preprocessing...")
    X_surr, _ = apply_preprocessing_pipeline(X_surr_raw, fit_on_data=X_surr_raw)
    
    print(f"[Surrogate] ✅ Dati preprocessati: {X_surr.shape}")
    print(f"[Surrogate] Distribuzione: {surr_info['attack_ratio']*100:.1f}% attacchi")
    
    # Addestra Random Forest surrogato
    # Configurazione IDENTICA al target per max transferability
    print(f"\n[Surrogate] Addestramento Random Forest surrogato...")
    
    rf_surrogate = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion='entropy',      # Identico al target
        max_features='sqrt',      # Identico al target
        class_weight='balanced',  # Identico al target
        random_state=random_state,
        n_jobs=-1
    )
    
    rf_surrogate.fit(X_surr, y_surr)
    
    # Valuta surrogato
    train_acc = rf_surrogate.score(X_surr, y_surr)
    
    print(f"[Surrogate] ✅ Surrogato addestrato:")
    print(f"  - N. alberi: {rf_surrogate.n_estimators}")
    print(f"  - N. feature: {rf_surrogate.n_features_in_}")
    print(f"  - Accuracy training: {train_acc:.4f}")
    
    surrogate_info = {
        'model': rf_surrogate,
        'client_ids': client_ids,
        'n_estimators': n_estimators,
        'train_samples': len(X_surr),
        'train_accuracy': train_acc,
        'attack_ratio': surr_info['attack_ratio']
    }
    
    return rf_surrogate, surrogate_info


def run_graybox_transfer_hsj_attack(
    target_model_path,
    surrogate_clients=[7, 11],
    test_clients=[1, 13],
    surrogate_estimators=50,
    max_iter=50,
    max_eval=5000,
    init_eval=100,
    norm=2,
    save_results=True,
    verbose=False
):
    """
    Esegue attacco Gray-Box Transfer con HSJ su surrogato Random Forest.
    
    WORKFLOW COMPLETO:
    
    FASE 1: ADDESTRAMENTO SURROGATO
    - Carica dati client 7, 11 (pubblici)
    - Addestra Random Forest surrogato (50 alberi)
    - Valuta performance
    
    FASE 2: CARICAMENTO TEST SET E TARGET
    - Carica modello target federato (per testing)
    - Carica test set (client 1, 13)
    - Preprocessing identico
    
    FASE 3: GENERAZIONE ADVERSARIAL SU SURROGATO
    - Wrap surrogato per ART
    - Configura HSJ su surrogato
    - Genera adversarial examples
    
    FASE 4: VALUTAZIONE SU SURROGATO
    - ASR sul surrogato (quanto è efficace HSJ)
    
    FASE 5: TRANSFER AL TARGET
    - Testa adversarial examples sul target federato
    - Calcola transferability
    
    Args:
        target_model_path: Path modello federato target
        surrogate_clients: Client per surrogato (default: [7, 11])
        test_clients: Client per test (default: [1, 13])
        surrogate_estimators: N. alberi surrogato (default: 50)
        max_iter: Max iterazioni HSJ (default: 50)
        max_eval: Max query (default: 5000)
        init_eval: Query init (default: 100)
        norm: Norma (default: 2)
        save_results: Salva risultati (default: True)
        verbose: Verbose (default: False)
        
    Returns:
        results: Dictionary con risultati completi
    """
    print("="*80)
    print("🟡 ATTACCO GRAY-BOX TRANSFER: HSJ SU SURROGATO → TRANSFER AL TARGET")
    print("="*80)
    print(f"Target model: {target_model_path}")
    print(f"Surrogate clients: {surrogate_clients}")
    print(f"Test clients: {test_clients}")
    print(f"Surrogate: {surrogate_estimators} alberi")
    print(f"HSJ config: max_iter={max_iter}, max_eval={max_eval}")
    print("="*80 + "\n")
    
    set_reproducibility_seeds(42)
    
    # ========== FASE 1: ADDESTRAMENTO SURROGATO ==========
    print("\n" + "="*80)
    print("FASE 1: ADDESTRAMENTO MODELLO SURROGATO")
    print("="*80)
    
    rf_surrogate, surrogate_info = train_surrogate_model(
        client_ids=surrogate_clients,
        n_estimators=surrogate_estimators,
        random_state=42
    )
    
    # ========== FASE 2: CARICAMENTO TEST SET E TARGET ==========
    print("\n" + "="*80)
    print("FASE 2: CARICAMENTO TEST SET E MODELLO TARGET")
    print("="*80)
    
    # Carica modello target federato
    print(f"\n[Gray-Box] Caricamento modello target federato...")
    rf_target = load_federated_model(target_model_path)
    
    # Carica test set
    print(f"\n[Gray-Box] Caricamento test set (client {test_clients})...")
    X_test_raw, y_test, test_info = load_test_data_from_clients(client_ids=test_clients)
    
    # Preprocessing
    print(f"\n[Gray-Box] Applicazione preprocessing...")
    X_test, _ = apply_preprocessing_pipeline(X_test_raw, fit_on_data=X_test_raw)
    
    # ✅ Verifica compatibilità ESPLICITA
    print(f"\n[Gray-Box] Verifica compatibilità dimensionale...")
    try:
        assert X_test.shape[1] == rf_target.n_features_in_, \
            f"Incompatibilità feature: test={X_test.shape[1]}, target={rf_target.n_features_in_}"
        assert X_test.shape[1] == rf_surrogate.n_features_in_, \
            f"Incompatibilità feature: test={X_test.shape[1]}, surrogate={rf_surrogate.n_features_in_}"
        print(f"[Gray-Box] ✅ Compatibilità verificata: target={rf_target.n_features_in_}, surrogate={rf_surrogate.n_features_in_}")
    except AssertionError as e:
        print(f"[Gray-Box] ❌ ERRORE: {e}")
        raise
    
    print(f"[Gray-Box] ✅ Test set preprocessato: {X_test.shape}")
    
    # Seleziona campioni Attack
    print(f"\n[Gray-Box] Selezione campioni Attack dal test set...")
    X_attacks_test, y_attacks_test, attack_indices = select_attack_samples(
        X_test, y_test, target_class=1
    )
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni Attack: {len(X_attacks_test)}")
    
    # ========== FASE 3: GENERAZIONE ADVERSARIAL SU SURROGATO ==========
    print("\n" + "="*80)
    print("FASE 3: GENERAZIONE ADVERSARIAL SU SURROGATO CON HSJ")
    print("="*80)
    
    # Wrap surrogato per ART
    print(f"\n[Gray-Box] Wrap surrogato per ART...")
    art_surrogate = SklearnClassifier(model=rf_surrogate)
    
    # ✅ Calcola percentili per logging (NON più usati da HopSkipJump)
    print(f"\n[Gray-Box] Calcolo range feature-wise con percentili robusti...")
    feature_min = np.percentile(X_test, 0.1, axis=0)
    feature_max = np.percentile(X_test, 99.9, axis=0)
    global_min = np.min(feature_min)
    global_max = np.max(feature_max)
    
    print(f"[Gray-Box] Range feature-wise: min={feature_min.min():.3f}, max={feature_max.max():.3f}")
    print(f"[Gray-Box] Range globale: [{global_min:.3f}, {global_max:.3f}]")
    print(f"[Gray-Box] 💡 NOTA: Clipping gestito automaticamente da SklearnClassifier")
    
    # Configura HSJ sul surrogato
    """
    CONFIGURAZIONE GRAY-BOX:
    
    max_iter: 50 (sufficiente per surrogato più semplice)
    max_eval: 5000 (budget medio)
    
    Il surrogato ha 50 alberi (vs 100 target) quindi:
    - Boundary decisionale più semplice
    - HSJ converge più velocemente
    
    ✅ clip_values RIMOSSO - gestito automaticamente
    """
    
    hsj_surrogate = HopSkipJump(
        classifier=art_surrogate,
        targeted=False,
        norm=norm,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        # ✅ clip_values RIMOSSO
        verbose=verbose
    )
    
    print(f"[Gray-Box] ✅ HSJ configurato su surrogato:")
    print(f"  - Max iter: {max_iter}")
    print(f"  - Max eval: {max_eval}")
    print(f"  - Clipping: Automatico")
    
    print(f"\n[Gray-Box] Generazione adversarial su SURROGATO...")
    print(f"  Tempo stimato: ~{len(X_attacks_test) * 1:.0f} secondi")
    
    try:
            
        X_adv_test = hsj_surrogate.generate(x=X_attacks_test)
        
        print(f"\n[Gray-Box] ✅ Generati {len(X_adv_test)} esempi adversarial")
        
    except Exception as e:
        print(f"\n[Gray-Box] ❌ Errore generazione: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== FASE 4: VALUTAZIONE SU SURROGATO ==========
    print("\n" + "="*80)
    print("FASE 4: VALUTAZIONE EFFICACIA SU SURROGATO")
    print("="*80)
    
    # Predizioni surrogato
    y_pred_surr_clean = rf_surrogate.predict(X_attacks_test)
    y_pred_surr_adv = rf_surrogate.predict(X_adv_test)
    
    # ASR sul surrogato
    evasion_surr_mask = (y_pred_surr_clean == 1) & (y_pred_surr_adv == 0)
    evasion_surr_rate = np.mean(evasion_surr_mask)
    evasions_surr = np.sum(evasion_surr_mask)
    
    print(f"\n[Gray-Box] 📊 RISULTATI SU SURROGATO:")
    print(f"  - ASR surrogato: {evasion_surr_rate*100:.2f}% ({evasions_surr}/{len(X_attacks_test)})")
    
    # ========== FASE 5: APPLICAZIONE VINCOLI FISICI ==========
    print("\n" + "="*80)
    print("FASE 5: APPLICAZIONE VINCOLI FISICI")
    print("="*80)
    
    constraints = get_smartgrid_physical_constraints(X_test)
    
    perturbation_before = X_adv_test - X_attacks_test
    l2_before = np.mean(np.linalg.norm(perturbation_before, axis=1))
    
    print(f"[Gray-Box] Prima vincoli: L2={l2_before:.6f}")
    
    X_adv_test_constrained = apply_physical_constraints(
        X_adv_test,
        X_attacks_test,
        constraints,
        max_perturbation_linf=None
    )
    
    perturbation_after = X_adv_test_constrained - X_attacks_test
    l2_after = np.mean(np.linalg.norm(perturbation_after, axis=1))
    
    print(f"[Gray-Box] Dopo vincoli: L2={l2_after:.6f}")
    
    # ========== FASE 6: TRANSFER AL TARGET FEDERATO ==========
    print("\n" + "="*80)
    print("FASE 6: TRANSFER AL MODELLO TARGET FEDERATO")
    print("="*80)
    
    # Predizioni target
    y_pred_target_clean = rf_target.predict(X_attacks_test)
    y_pred_target_adv = rf_target.predict(X_adv_test_constrained)
    
    # ASR sul target
    evasion_target_mask = (y_pred_target_clean == 1) & (y_pred_target_adv == 0)
    evasion_target_rate = np.mean(evasion_target_mask)
    evasions_target = np.sum(evasion_target_mask)
    
    # Transferability
    if evasions_surr > 0:
        transferability_rate = evasions_target / evasions_surr
    else:
        transferability_rate = 0.0
    
    print(f"\n[Gray-Box] 📊 RISULTATI TRANSFERABILITY:")
    print(f"  - ASR SURROGATO: {evasion_surr_rate*100:.2f}% ({evasions_surr} evasioni)")
    print(f"  - ASR TARGET:    {evasion_target_rate*100:.2f}% ({evasions_target} evasioni)")
    print(f"  - TRANSFERABILITY: {transferability_rate*100:.2f}% ({evasions_target}/{evasions_surr})")
    print(f"\n  Interpretazione:")
    print(f"  - {evasions_target} esempi generati su surrogato evadono anche il target")
    print(f"  - {transferability_rate*100:.2f}% degli esempi si trasferiscono con successo")
    
    # ========== FASE 7: RICOSTRUZIONE E VALUTAZIONE COMPLETA ==========
    X_adv_full = X_test.copy()
    X_adv_full[attack_indices] = X_adv_test_constrained
    
    metrics = evaluate_attack(
        rf_target,
        X_test,
        y_test,
        X_adv_full,
        attack_name="GrayBox_Transfer_HSJ"
    )
    
    # Aggiungi metriche transferability
    metrics['asr_surrogate'] = float(evasion_surr_rate)
    metrics['evasions_surrogate'] = int(evasions_surr)
    metrics['transferability_rate'] = float(transferability_rate)
    
    print_attack_report(metrics)
    
    # ========== FASE 8: SALVATAGGIO ==========
    if save_results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        save_attack_results(
            [metrics],
            X_test,
            {'graybox_transfer_hsj': X_adv_full},
            epsilons_tested=['Transfer_HSJ'],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== FASE 9: SUMMARY ==========
    print(f"\n{'='*80}")
    print(f"✅ ATTACCO GRAY-BOX TRANSFER COMPLETATO")
    print(f"{'='*80}")
    print(f"\n📊 RIASSUNTO FINALE:")
    print(f"\n1. SURROGATO:")
    print(f"   - Client: {surrogate_clients}")
    print(f"   - N. alberi: {surrogate_estimators}")
    print(f"   - Accuracy: {surrogate_info['train_accuracy']:.4f}")
    print(f"\n2. EFFICACIA SU SURROGATO:")
    print(f"   - ASR: {evasion_surr_rate*100:.2f}%")
    print(f"   - Evasioni: {evasions_surr}/{len(X_attacks_test)}")
    print(f"\n3. TRANSFER AL TARGET:")
    print(f"   - ASR: {evasion_target_rate*100:.2f}%")
    print(f"   - Evasioni: {evasions_target}/{len(X_attacks_test)}")
    print(f"   - Transferability: {transferability_rate*100:.2f}%")
    print(f"\n4. PERTURBAZIONI:")
    print(f"   - L2 medio: {metrics['l2_mean']:.6f}")
    print(f"   - L-inf medio: {metrics['linf_mean']:.6f}")
    print(f"   - Feature modificate: {metrics['l0_mean']:.2f}")
    print(f"={'='*80}\n")
    
    results = {
        'metrics': metrics,
        'X_adv': X_adv_full,
        'surrogate_info': surrogate_info,
        'transferability_rate': transferability_rate
    }
    
    return results


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Attacco Gray-Box Transfer HSJ su surrogato Random Forest"
    )
    
    parser.add_argument(
        '--target-model-path',
        type=str,
        required=True,
        help='Path modello target federato'
    )
    
    parser.add_argument(
        '--surrogate-clients',
        type=int,
        nargs='+',
        default=[7, 11],
        help='Client per surrogato (default: 7 11)'
    )
    
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client per test (default: 1 13)'
    )
    
    parser.add_argument(
        '--surrogate-estimators',
        type=int,
        default=50,
        help='N. alberi surrogato (default: 50)'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50,
        help='Max iterazioni HSJ (default: 50)'
    )
    
    parser.add_argument(
        '--max-eval',
        type=int,
        default=5000,
        help='Max query (default: 5000)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Salva risultati'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose'
    )
    
    args = parser.parse_args()
    
    results = run_graybox_transfer_hsj_attack(
        target_model_path=args.target_model_path,
        surrogate_clients=args.surrogate_clients,
        test_clients=args.test_clients,
        surrogate_estimators=args.surrogate_estimators,
        max_iter=args.max_iter,
        max_eval=args.max_eval,
        save_results=args.save_results or True,
        verbose=args.verbose
    )
    
    if results is None:
        sys.exit(1)
    else:
        print(f"\n✅ Transfer: {results['transferability_rate']*100:.2f}%")


if __name__ == "__main__":
    main()