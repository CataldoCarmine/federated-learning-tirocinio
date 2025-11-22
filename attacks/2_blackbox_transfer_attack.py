"""
attacks/2_blackbox_transfer_attack.py

Attacco Black-Box Transfer usando ZOO (Zeroth-Order Optimization) su modello surrogato.

STRATEGIA BLACK-BOX TRANSFER:
1. Addestra modello SURROGATO Random Forest su dati pubblici (client 2-12)
2. Genera adversarial examples sul SURROGATO usando ZOO (ART)
3. TRASFERISCE esempi adversarial al modello TARGET federato (client 1, 13)
4. Misura transferability: quanti esempi generati sul surrogato evadono anche il target?

PERCHÉ QUESTO APPROCCIO È BLACK-BOX:
- L'attaccante NON ha accesso al modello target federato
- L'attaccante HA solo dati pubblici SmartGrid
- L'attaccante addestra un surrogato che "imita" il target
- Gli adversarial examples sono generati SOLO sul surrogato
- Zero query al modello target (scenario realistico)

PERCHÉ ZOO SU RANDOM FOREST SURROGATO:
- ZOO non richiede gradienti (perfetto per Random Forest)
- ZOO usa solo predict_proba() del surrogato
- ZOO ottimizza perturbazioni in modo iterativo
- Transferability attesa: 15-35% (vs white-box 0.19-1.04%)

UTILIZZO:
    python attacks/2_blackbox_transfer_attack.py \
        --target-model-path models/federated_rf_global_20251121_024044.pkl \
        --surrogate-estimators 50 \
        --zoo-max-iter 100 \
        --test-clients 1 13 \
        --save-results

AUTORE: Carmine Cataldo
DATA: 2025-01-21
"""

import numpy as np
import sys
import os
import argparse
from typing import Tuple, Dict

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import ART
from art.attacks.evasion import ZooAttack
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
from attacks.surrogate_training import train_surrogate_model


def run_blackbox_transfer_attack(
    target_model_path,
    test_clients=[1, 13],
    surrogate_estimators=50,
    surrogate_clients=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    zoo_max_iter=100,
    zoo_learning_rate=0.01,
    zoo_binary_search_steps=10,
    zoo_initial_const=0.01,
    save_results=True,
    verbose=False
):
    """
    Esegue attacco Black-Box Transfer con ZOO su surrogato Random Forest.
    
    WORKFLOW COMPLETO:
    
    FASE 1: ADDESTRAMENTO SURROGATO
    - Carica dati pubblici (client 2-12)
    - Addestra Random Forest surrogato (50 alberi)
    - Valuta performance surrogato
    
    FASE 2: GENERAZIONE ADVERSARIAL SUL SURROGATO
    - Wrap surrogato per ART
    - Configura ZOO attack
    - Genera adversarial examples usando ZOO sul surrogato
    
    FASE 3: TRANSFER AL TARGET FEDERATO
    - Carica modello target federato
    - Testa esempi adversarial generati sul surrogato
    - Misura transferability (ASR sul target)
    
    FASE 4: VALUTAZIONE E CONFRONTO
    - ASR sul surrogato (quanto è efficace ZOO)
    - ASR sul target (transferability)
    - Metriche perturbazione
    
    Args:
        target_model_path: Path al modello Random Forest federato target (.pkl)
        test_clients: Client per dataset test (default: [1, 13])
        surrogate_estimators: Numero alberi surrogato (default: 50)
        surrogate_clients: Client per training surrogato (default: 2-12)
        zoo_max_iter: Max iterazioni ZOO (default: 100)
        zoo_learning_rate: Learning rate ZOO (default: 0.01)
        zoo_binary_search_steps: Binary search steps ZOO (default: 10)
        zoo_initial_const: Costante iniziale ZOO (default: 0.01)
        save_results: Se True, salva risultati (default: True)
        verbose: Se True, stampa dettagli (default: False)
        
    Returns:
        results: Dictionary con risultati completi dell'attacco
    """
    print("="*80)
    print("🔴 ATTACCO BLACK-BOX: TRANSFER ATTACK con ZOO su SURROGATO")
    print("="*80)
    print(f"Target model: {target_model_path}")
    print(f"Test clients: {test_clients}")
    print(f"Surrogato: {surrogate_estimators} alberi, training su client {surrogate_clients}")
    print(f"ZOO config: max_iter={zoo_max_iter}, lr={zoo_learning_rate}")
    print("="*80 + "\n")
    
    set_reproducibility_seeds(42)
    
    # ========== FASE 1: ADDESTRAMENTO SURROGATO ==========
    print("\n" + "="*80)
    print("FASE 1: ADDESTRAMENTO MODELLO SURROGATO")
    print("="*80)
    
    surrogate_model, surrogate_info = train_surrogate_model(
        client_ids=surrogate_clients,
        n_estimators=surrogate_estimators,
        max_depth=None,
        random_state=42,
        save_model=save_results,
        model_dir=os.path.join(os.path.dirname(__file__), 'models')
    )
    
    print(f"\n[Transfer Attack] ✅ Surrogato addestrato:")
    print(f"  - Accuracy training: {surrogate_info['train_accuracy']:.4f}")
    print(f"  - Accuracy validation: {surrogate_info['val_accuracy']:.4f}")
    print(f"  - N. alberi: {surrogate_info['n_estimators']}")
    
    # ========== FASE 2: CARICA DATI TEST E TARGET ==========
    print("\n" + "="*80)
    print("FASE 2: CARICAMENTO DATI TEST E MODELLO TARGET")
    print("="*80)
    
    # Carica modello target federato
    print(f"\n[Transfer Attack] Caricamento modello target federato...")
    target_model = load_federated_model(target_model_path)
    
    # Carica dati test
    print(f"\n[Transfer Attack] Caricamento dati test...")
    X_test_raw, y_test, test_info = load_test_data_from_clients(client_ids=test_clients)
    
    # Preprocessing (identico a surrogato e target)
    print(f"\n[Transfer Attack] Applicazione preprocessing...")
    X_test, _ = apply_preprocessing_pipeline(X_test_raw, fit_on_data=X_test_raw)
    
    # Verifica compatibilità
    if X_test.shape[1] != target_model.n_features_in_:
        raise ValueError(
            f"❌ Incompatibilità feature: test={X_test.shape[1]}, target={target_model.n_features_in_}"
        )
    if X_test.shape[1] != surrogate_model.n_features_in_:
        raise ValueError(
            f"❌ Incompatibilità feature: test={X_test.shape[1]}, surrogato={surrogate_model.n_features_in_}"
        )
    
    print(f"[Transfer Attack] ✅ Preprocessing completato: {X_test.shape}")
    
    # Seleziona solo campioni di attacco
    print(f"\n[Transfer Attack] Selezione campioni di attacco...")
    X_attacks_only, y_attacks_only, attack_indices = select_attack_samples(
        X_test, y_test, target_class=1
    )
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni di attacco: {len(X_attacks_only)}")
    print(f"  - Campioni naturali: {(y_test == 0).sum()}")
    
    # ========== FASE 3: GENERAZIONE ADVERSARIAL CON ZOO SUL SURROGATO ==========
    print("\n" + "="*80)
    print("FASE 3: GENERAZIONE ADVERSARIAL EXAMPLES con ZOO su SURROGATO")
    print("="*80)
    
    # Wrap surrogato per ART
    print(f"\n[Transfer Attack] Configurazione ZOO su surrogato...")
    art_surrogate = SklearnClassifier(model=surrogate_model)
    
    # Configura ZOO attack
    """
    ZOO (Zeroth-Order Optimization):
    - NON usa gradienti (perfetto per Random Forest)
    - Approssima gradiente con differenze finite
    - Ottimizza perturbazione iterativamente
    - Usa solo predict_proba() del surrogato
    
    Parametri:
    - max_iter: Numero iterazioni ottimizzazione (100 = buon compromesso)
    - learning_rate: Step size per aggiornamenti (0.01 standard)
    - binary_search_steps: Ricerca binaria per costante C (10 standard)
    - initial_const: Costante iniziale per loss (0.01 standard)
    - confidence: Threshold confidenza per successo (0.0 = qualsiasi flip)
    - targeted: False = evasion attack (Attack → Natural)
    - batch_size: 1 (processa un campione alla volta)
    """
    
    zoo_attack = ZooAttack(
        classifier=art_surrogate,
        confidence=0.0,
        targeted=False,
        learning_rate=zoo_learning_rate,
        max_iter=zoo_max_iter,
        binary_search_steps=zoo_binary_search_steps,
        initial_const=zoo_initial_const,
        abort_early=True,
        use_resize=False,
        use_importance=True,
        nb_parallel=1,
        batch_size=1,
        variable_h=0.01
    )
    
    print(f"[Transfer Attack] ✅ ZOO configurato:")
    print(f"  - Max iterations: {zoo_max_iter}")
    print(f"  - Learning rate: {zoo_learning_rate}")
    print(f"  - Binary search steps: {zoo_binary_search_steps}")
    print(f"  - Initial const: {zoo_initial_const}")
    
    # Genera adversarial examples sul SURROGATO
    print(f"\n[Transfer Attack] Generazione adversarial examples con ZOO sul SURROGATO...")
    print(f"  ⚠️ ATTENZIONE: ZOO è LENTO (iterativo)")
    print(f"  Tempo stimato: ~{len(X_attacks_only) * 2:.0f} secondi per {len(X_attacks_only)} campioni")
    
    try:
        X_attacks_adv_surrogate = zoo_attack.generate(x=X_attacks_only)
        print(f"[Transfer Attack] ✅ Generati {len(X_attacks_adv_surrogate)} esempi adversarial sul SURROGATO")
    except Exception as e:
        print(f"[Transfer Attack] ❌ Errore generazione ZOO: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== FASE 4: VALUTAZIONE SUL SURROGATO ==========
    print("\n" + "="*80)
    print("FASE 4: VALUTAZIONE EFFICACIA ZOO sul SURROGATO")
    print("="*80)
    
    # Predizioni sul surrogato
    surrogate_pred_original = surrogate_model.predict(X_attacks_only)
    surrogate_pred_adversarial = surrogate_model.predict(X_attacks_adv_surrogate)
    
    # ASR sul surrogato
    surrogate_evasions = np.sum(
        (surrogate_pred_original == 1) &
        (surrogate_pred_adversarial == 0)
    )
    surrogate_asr = surrogate_evasions / len(X_attacks_only)
    
    print(f"\n[Transfer Attack] Performance ZOO sul SURROGATO:")
    print(f"  - ASR sul surrogato: {surrogate_asr*100:.2f}% ({surrogate_evasions}/{len(X_attacks_only)})")
    print(f"  - Questo misura quanto è efficace ZOO sul modello surrogato")
    
    # ========== FASE 5: APPLICAZIONE VINCOLI FISICI ==========
    print("\n" + "="*80)
    print("FASE 5: APPLICAZIONE VINCOLI FISICI SMARTGRID")
    print("="*80)
    
    constraints = get_smartgrid_physical_constraints(X_test)
    
    # Calcola perturbazione prima dei vincoli
    perturbation_before = X_attacks_adv_surrogate - X_attacks_only
    l2_before = np.mean(np.linalg.norm(perturbation_before, axis=1))
    linf_before = np.mean(np.max(np.abs(perturbation_before), axis=1))
    
    print(f"[Transfer Attack] Prima vincoli: L2={l2_before:.6f}, L-inf={linf_before:.6f}")
    
    # Applica vincoli
    X_attacks_adv_constrained = apply_physical_constraints(
        X_attacks_adv_surrogate,
        X_attacks_only,
        constraints,
        max_perturbation_linf=None  # ZOO già ottimizza
    )
    
    # Calcola perturbazione dopo vincoli
    perturbation_after = X_attacks_adv_constrained - X_attacks_only
    l2_after = np.mean(np.linalg.norm(perturbation_after, axis=1))
    linf_after = np.mean(np.max(np.abs(perturbation_after), axis=1))
    
    print(f"[Transfer Attack] Dopo vincoli: L2={l2_after:.6f}, L-inf={linf_after:.6f}")
    print(f"[Transfer Attack] Impatto vincoli: ΔL2={l2_after-l2_before:.6f}, ΔL-inf={linf_after-linf_before:.6f}")
    
    # ========== FASE 6: TRANSFER AL TARGET FEDERATO ==========
    print("\n" + "="*80)
    print("FASE 6: TRANSFER AL MODELLO TARGET FEDERATO")
    print("="*80)
    
    # Ricostruisci dataset completo per valutazione
    X_adv_full = X_test.copy()
    X_adv_full[attack_indices] = X_attacks_adv_constrained
    
    # Valuta sul TARGET federato
    print(f"\n[Transfer Attack] Valutazione esempi adversarial sul TARGET FEDERATO...")
    
    target_metrics = evaluate_attack(
        target_model,
        X_test,
        y_test,
        X_adv_full,
        attack_name="BlackBox_Transfer_ZOO"
    )
    
    # ========== FASE 7: VALUTAZIONE TRANSFERABILITY ==========
    print("\n" + "="*80)
    print("FASE 7: ANALISI TRANSFERABILITY")
    print("="*80)
    
    # Predizioni sul target
    target_pred_original = target_model.predict(X_attacks_only)
    target_pred_adversarial = target_model.predict(X_attacks_adv_constrained)
    
    # ASR sul target
    target_evasions = np.sum(
        (target_pred_original == 1) &
        (target_pred_adversarial == 0)
    )
    target_asr = target_evasions / len(X_attacks_only)
    
    # Transferability rate
    if surrogate_evasions > 0:
        transferability_rate = target_evasions / surrogate_evasions
    else:
        transferability_rate = 0.0
    
    print(f"\n[Transfer Attack] 📊 RISULTATI TRANSFERABILITY:")
    print(f"  - ASR sul SURROGATO: {surrogate_asr*100:.2f}% ({surrogate_evasions} evasioni)")
    print(f"  - ASR sul TARGET:    {target_asr*100:.2f}% ({target_evasions} evasioni)")
    print(f"  - TRANSFERABILITY:   {transferability_rate*100:.2f}% ({target_evasions}/{surrogate_evasions})")
    print(f"\n  Interpretazione:")
    print(f"  - {target_evasions} esempi generati sul surrogato evadono anche il target")
    print(f"  - {transferability_rate*100:.2f}% degli esempi si trasferiscono con successo")
    
    # ========== FASE 8: REPORT DETTAGLIATO ==========
    print("\n" + "="*80)
    print("FASE 8: REPORT FINALE")
    print("="*80)
    
    print_attack_report(target_metrics)
    
    # ========== FASE 9: SALVATAGGIO RISULTATI ==========
    if save_results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        # Salva metriche con metadata transfer
        target_metrics['surrogate_asr'] = float(surrogate_asr)
        target_metrics['surrogate_evasions'] = int(surrogate_evasions)
        target_metrics['transferability_rate'] = float(transferability_rate)
        target_metrics['surrogate_estimators'] = int(surrogate_estimators)
        target_metrics['zoo_max_iter'] = int(zoo_max_iter)
        
        save_attack_results(
            [target_metrics],
            X_test,
            {'transfer_zoo': X_adv_full},
            epsilons_tested=['Transfer_ZOO'],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== FASE 10: SUMMARY FINALE ==========
    print(f"\n{'='*80}")
    print(f"✅ ATTACCO BLACK-BOX TRANSFER COMPLETATO")
    print(f"{'='*80}")
    print(f"\n📊 RIASSUNTO FINALE:")
    print(f"\n1. MODELLO SURROGATO:")
    print(f"   - Alberi: {surrogate_estimators}")
    print(f"   - Accuracy: {surrogate_info['val_accuracy']:.4f}")
    print(f"\n2. EFFICACIA ZOO SUL SURROGATO:")
    print(f"   - ASR: {surrogate_asr*100:.2f}%")
    print(f"   - Evasioni: {surrogate_evasions}/{len(X_attacks_only)}")
    print(f"\n3. TRANSFER AL TARGET FEDERATO:")
    print(f"   - ASR: {target_asr*100:.2f}%")
    print(f"   - Evasioni: {target_evasions}/{len(X_attacks_only)}")
    print(f"   - Transferability: {transferability_rate*100:.2f}%")
    print(f"\n4. PERTURBAZIONI:")
    print(f"   - L2 medio: {target_metrics['l2_mean']:.6f}")
    print(f"   - L-inf medio: {target_metrics['linf_mean']:.6f}")
    print(f"   - Feature modificate: {target_metrics['l0_mean']:.2f}")
    print(f"\n5. CONFRONTO CON WHITE-BOX:")
    print(f"   - White-Box Monte Carlo: ASR 0.19%")
    print(f"   - White-Box HopSkipJump: ASR 1.04%")
    print(f"   - Black-Box Transfer: ASR {target_asr*100:.2f}%")
    print(f"={'='*80}\n")
    
    # Return completo
    results = {
        'target_metrics': target_metrics,
        'surrogate_info': surrogate_info,
        'surrogate_asr': surrogate_asr,
        'target_asr': target_asr,
        'transferability_rate': transferability_rate,
        'X_adv': X_adv_full
    }
    
    return results


def main():
    """Funzione principale per esecuzione da linea di comando."""
    parser = argparse.ArgumentParser(
        description="Attacco Black-Box Transfer con ZOO su surrogato Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:

  # Esecuzione base
  %(prog)s --target-model-path models/federated_rf_global_20251121_024044.pkl

  # Con parametri personalizzati
  %(prog)s --target-model-path models/model.pkl --surrogate-estimators 100 --zoo-max-iter 200

  # Modalità verbose
  %(prog)s --target-model-path models/model.pkl --verbose

PARAMETRI RACCOMANDATI:
  - surrogate-estimators: 50 (più veloce) o 100 (più accurato)
  - zoo-max-iter: 100 (compromesso) o 200 (migliore)
        """
    )
    
    parser.add_argument(
        '--target-model-path',
        type=str,
        required=True,
        help='Path al modello Random Forest federato target (.pkl)'
    )
    
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client da usare per test (default: 1 13)'
    )
    
    parser.add_argument(
        '--surrogate-estimators',
        type=int,
        default=50,
        help='Numero alberi surrogato (default: 50)'
    )
    
    parser.add_argument(
        '--surrogate-clients',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help='Client per training surrogato (default: 2-12)'
    )
    
    parser.add_argument(
        '--zoo-max-iter',
        type=int,
        default=100,
        help='Max iterazioni ZOO (default: 100)'
    )
    
    parser.add_argument(
        '--zoo-learning-rate',
        type=float,
        default=0.01,
        help='Learning rate ZOO (default: 0.01)'
    )
    
    parser.add_argument(
        '--zoo-binary-search-steps',
        type=int,
        default=10,
        help='Binary search steps ZOO (default: 10)'
    )
    
    parser.add_argument(
        '--zoo-initial-const',
        type=float,
        default=0.01,
        help='Costante iniziale ZOO (default: 0.01)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Salva risultati in file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Stampa informazioni dettagliate'
    )
    
    args = parser.parse_args()
    
    # Esegui attacco
    results = run_blackbox_transfer_attack(
        target_model_path=args.target_model_path,
        test_clients=args.test_clients,
        surrogate_estimators=args.surrogate_estimators,
        surrogate_clients=args.surrogate_clients,
        zoo_max_iter=args.zoo_max_iter,
        zoo_learning_rate=args.zoo_learning_rate,
        zoo_binary_search_steps=args.zoo_binary_search_steps,
        zoo_initial_const=args.zoo_initial_const,
        save_results=args.save_results or True,
        verbose=args.verbose
    )
    
    if results is None:
        print("\n❌ Attacco fallito. Controlla gli errori sopra.")
        sys.exit(1)
    else:
        print(f"\n✅ Attacco completato con successo!")
        print(f"   ASR finale sul target: {results['target_asr']*100:.2f}%")
        print(f"   Transferability: {results['transferability_rate']*100:.2f}%")


if __name__ == "__main__":
    main()