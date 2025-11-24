"""
attacks/hsj_whitebox.py

Attacco White-Box con HopSkipJump sul Random Forest federato globale.

MODIFICHE RISPETTO ALLA VERSIONE PRECEDENTE:
1. ✅ Clip values feature-wise con percentili robusti (0.1-99.9)
2. ✅ Verifica compatibilità esplicita con assert
3. ✅ Logging migliorato con statistiche dettagliate
4. ✅ Gestione errori robusta con fallback

SCENARIO:
Attaccante interno (o scenario worst-case) con accesso completo al modello
Random Forest globale salvato: struttura alberi, parametri, dataset test.

STRATEGIA:
1. Carica modello federato salvato dal server
2. Carica dati test (client 1, 13)
3. Applica preprocessing identico al federato
4. Configura HopSkipJump con budget generoso (white-box)
5. Genera adversarial examples
6. Applica vincoli fisici SmartGrid
7. Valuta efficacia (ASR, metriche perturbazione)

HOPSKIPJUMP (HSJ):
- Algoritmo decision-based (usa solo predizioni binarie)
- Non richiede gradienti (perfetto per Random Forest)
- Esplora boundary decisionale in modo iterativo
- Minimizza perturbazione mantenendo evasione

PARAMETRI WHITE-BOX:
- max_iter: 100 (convergenza accurata)
- max_eval: 10000 (budget generoso)
- norm: L2 (minimizza distanza euclidea)

UTILIZZO:
    python attacks/hsj_whitebox.py \
        --model-path models/federated_rf_global_20251121_024044.pkl \
        --max-iter 100 \
        --max-eval 10000 \
        --save-results

AUTORE: Carmine Cataldo
DATA: 2025-01-23 (Aggiornato)
"""

import numpy as np
import sys
import os
import argparse
from typing import Tuple, Dict

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


def run_whitebox_hsj_attack(
    model_path,
    test_clients=[1, 13],
    max_iter=100,
    max_eval=10000,
    init_eval=100,
    norm=2,
    save_results=True,
    verbose=False
):
    """
    Esegue attacco White-Box con HopSkipJump sul Random Forest federato globale.
    
    WORKFLOW COMPLETO:
    
    FASE 1: CARICAMENTO MODELLO E DATI
    - Carica modello Random Forest federato salvato
    - Carica dataset test (client 1, 13)
    - Applica preprocessing identico al federato
    
    FASE 2: CONFIGURAZIONE HOPSKIPJUMP WHITE-BOX
    - Wrap modello per ART
    - Configura HSJ con parametri generosi (worst-case)
    - Calcola vincoli fisici SmartGrid
    
    FASE 3: GENERAZIONE ADVERSARIAL EXAMPLES
    - Seleziona campioni Attack
    - Genera perturbazioni con HSJ
    - Applica vincoli fisici
    
    FASE 4: VALUTAZIONE
    - Calcola ASR (Attack Success Rate)
    - Metriche perturbazione (L2, L-inf, L0)
    - Report dettagliato
    
    Args:
        model_path: Path al modello Random Forest federato (.pkl)
        test_clients: Client per test set (default: [1, 13])
        max_iter: Max iterazioni HSJ (default: 100)
        max_eval: Max query al modello (default: 10000)
        init_eval: Query per inizializzazione (default: 100)
        norm: Norma da minimizzare (default: 2 = L2)
        save_results: Se True, salva risultati (default: True)
        verbose: Se True, stampa dettagli (default: False)
        
    Returns:
        results: Dictionary con risultati completi
    """
    print("="*80)
    print("🔴 ATTACCO WHITE-BOX: HOPSKIPJUMP SUL MODELLO GLOBALE")
    print("="*80)
    print(f"Modello target: {model_path}")
    print(f"Test clients: {test_clients}")
    print(f"Configurazione HSJ (white-box worst-case):")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Max evaluations (query): {max_eval}")
    print(f"  - Init evaluations: {init_eval}")
    print(f"  - Norm: L{norm}")
    print("="*80 + "\n")
    
    set_reproducibility_seeds(42)
    
    # ========== FASE 1: CARICAMENTO MODELLO E DATI ==========
    print("\n" + "="*80)
    print("FASE 1: CARICAMENTO MODELLO TARGET FEDERATO E DATI TEST")
    print("="*80)
    
    # Carica modello federato globale
    print(f"\n[White-Box HSJ] Caricamento modello federato globale...")
    model = load_federated_model(model_path)
    
    # Carica dati test
    print(f"\n[White-Box HSJ] Caricamento dataset test...")
    X_test_raw, y_test, test_info = load_test_data_from_clients(client_ids=test_clients)
    
    # Preprocessing (identico al federato)
    print(f"\n[White-Box HSJ] Applicazione preprocessing...")
    X_test, _ = apply_preprocessing_pipeline(X_test_raw, fit_on_data=X_test_raw)
    
    # ✅ MODIFICA 1: Verifica compatibilità ESPLICITA con assert
    print(f"\n[White-Box HSJ] Verifica compatibilità dimensionale...")
    try:
        assert X_test.shape[1] == model.n_features_in_, \
            f"Incompatibilità feature: test={X_test.shape[1]}, modello={model.n_features_in_}"
        print(f"[White-Box HSJ] ✅ Compatibilità verificata: {X_test.shape[1]} feature")
    except AssertionError as e:
        print(f"[White-Box HSJ] ❌ ERRORE: {e}")
        raise
    
    print(f"[White-Box HSJ] ✅ Preprocessing completato: {X_test.shape}")
    
    # Seleziona solo campioni Attack
    print(f"\n[White-Box HSJ] Selezione campioni di attacco...")
    X_attacks_only, y_attacks_only, attack_indices = select_attack_samples(
        X_test, y_test, target_class=1
    )
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni di attacco: {len(X_attacks_only)}")
    print(f"  - Campioni naturali: {(y_test == 0).sum()}")
    
    # ========== FASE 2: CONFIGURAZIONE HOPSKIPJUMP WHITE-BOX ==========
    print("\n" + "="*80)
    print("FASE 2: CONFIGURAZIONE HOPSKIPJUMP WHITE-BOX")
    print("="*80)
    
    # Wrap modello per ART
    print(f"\n[White-Box HSJ] Wrap modello Random Forest per ART...")
    art_classifier = SklearnClassifier(model=model)
    
    # ✅ MODIFICA 2: Clip values FEATURE-WISE con percentili robusti
    print(f"\n[White-Box HSJ] Calcolo clip values feature-wise con percentili robusti...")
    feature_min = np.percentile(X_test, 0.1, axis=0)  # Percentile 0.1% (robusto)
    feature_max = np.percentile(X_test, 99.9, axis=0)  # Percentile 99.9% (robusto)
    
    # Converti in range globale per compatibilità ART
    # (ART accetta sia (scalar, scalar) che (array, array))
    global_min = np.min(feature_min)
    global_max = np.max(feature_max)
    clip_values = (global_min, global_max)
    
    print(f"[White-Box HSJ] ✅ Modello wrapped per ART")
    print(f"[White-Box HSJ] Range feature-wise: min={feature_min.min():.3f}, max={feature_max.max():.3f}")
    print(f"[White-Box HSJ] Range globale usato: [{global_min:.3f}, {global_max:.3f}]")
    
    # Configura HopSkipJump
    """
    CONFIGURAZIONE WHITE-BOX (scenario worst-case):
    
    max_iter: 100
    - Convergenza accurata per perturbazioni minime
    - In white-box abbiamo tempo e risorse illimitate
    
    max_eval: 10000
    - Budget query generoso (white-box)
    - Random Forest è veloce → possiamo permetterci molte query
    
    norm: L2
    - Minimizza distanza euclidea (standard)
    - Alternative: np.inf per L-inf
    
    targeted: False
    - Evasion attack non-targeted (Attack → Natural)
    """
    
    hsj_attack = HopSkipJump(
        classifier=art_classifier,
        targeted=False,           # Evasion non-targeted
        norm=norm,                # Minimizza L2
        max_iter=max_iter,        # Iterazioni HSJ
        max_eval=max_eval,        # Budget query generoso
        init_eval=init_eval,      # Query inizializzazione
        init_size=100,            # Batch size iniziale
        clip_values=clip_values,  # Vincoli globali
        verbose=verbose
    )
    
    print(f"\n[White-Box HSJ] ✅ HopSkipJump configurato:")
    print(f"  - Max iterations: {max_iter} (convergenza accurata)")
    print(f"  - Max evaluations: {max_eval} (budget generoso white-box)")
    print(f"  - Norm: L{norm} (minimizza distanza euclidea)")
    print(f"  - Clip values: [{clip_values[0]:.3f}, {clip_values[1]:.3f}]")
    
    # ========== FASE 3: GENERAZIONE ADVERSARIAL EXAMPLES ==========
    print("\n" + "="*80)
    print("FASE 3: GENERAZIONE ADVERSARIAL EXAMPLES CON HSJ")
    print("="*80)
    
    print(f"\n[White-Box HSJ] Generazione adversarial examples...")
    print(f"  ⚠️ HSJ è iterativo (query-intensive)")
    print(f"  Tempo stimato: ~{len(X_attacks_only) * 1.5:.0f} secondi per {len(X_attacks_only)} campioni")
    
    try:
        X_attacks_adv = hsj_attack.generate(x=X_attacks_only)
        print(f"[White-Box HSJ] ✅ Generati {len(X_attacks_adv)} esempi adversarial")
    except Exception as e:
        print(f"[White-Box HSJ] ❌ Errore generazione: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== FASE 4: APPLICAZIONE VINCOLI FISICI ==========
    print("\n" + "="*80)
    print("FASE 4: APPLICAZIONE VINCOLI FISICI SMARTGRID")
    print("="*80)
    
    constraints = get_smartgrid_physical_constraints(X_test)
    
    # Calcola perturbazione prima vincoli
    perturbation_before = X_attacks_adv - X_attacks_only
    l2_before = np.mean(np.linalg.norm(perturbation_before, axis=1))
    linf_before = np.mean(np.max(np.abs(perturbation_before), axis=1))
    
    print(f"[White-Box HSJ] Prima vincoli: L2={l2_before:.6f}, L-inf={linf_before:.6f}")
    
    # Applica vincoli
    X_attacks_adv_constrained = apply_physical_constraints(
        X_attacks_adv,
        X_attacks_only,
        constraints,
        max_perturbation_linf=None  # HSJ già ottimizza
    )
    
    # Calcola perturbazione dopo vincoli
    perturbation_after = X_attacks_adv_constrained - X_attacks_only
    l2_after = np.mean(np.linalg.norm(perturbation_after, axis=1))
    linf_after = np.mean(np.max(np.abs(perturbation_after), axis=1))
    
    print(f"[White-Box HSJ] Dopo vincoli: L2={l2_after:.6f}, L-inf={linf_after:.6f}")
    print(f"[White-Box HSJ] Impatto vincoli: ΔL2={l2_after-l2_before:.6f}, ΔL-inf={linf_after-linf_before:.6f}")
    
    # ========== FASE 5: RICOSTRUZIONE DATASET COMPLETO ==========
    X_adv_full = X_test.copy()
    X_adv_full[attack_indices] = X_attacks_adv_constrained
    
    # ========== FASE 6: VALUTAZIONE ==========
    print("\n" + "="*80)
    print("FASE 5: VALUTAZIONE EFFICACIA ATTACCO WHITE-BOX")
    print("="*80)
    
    metrics = evaluate_attack(
        model,
        X_test,
        y_test,
        X_adv_full,
        attack_name="WhiteBox_HSJ_Global"
    )
    
    print_attack_report(metrics)
    
    # ========== FASE 7: SALVATAGGIO RISULTATI ==========
    if save_results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        save_attack_results(
            [metrics],
            X_test,
            {'whitebox_hsj': X_adv_full},
            epsilons_tested=[f'WhiteBox_HSJ_L{norm}'],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== FASE 8: SUMMARY FINALE ==========
    print(f"\n{'='*80}")
    print(f"✅ ATTACCO WHITE-BOX HSJ COMPLETATO")
    print(f"{'='*80}")
    print(f"\n📊 RIASSUNTO FINALE:")
    print(f"\n1. MODELLO TARGET:")
    print(f"   - Tipo: Random Forest Federato Globale")
    print(f"   - N. alberi: {len(model.estimators_)}")
    print(f"   - N. feature: {model.n_features_in_}")
    print(f"\n2. EFFICACIA ATTACCO:")
    print(f"   - ASR: {metrics['asr']*100:.2f}%")
    print(f"   - Evasioni: {metrics['successful_evasions']}/{metrics['total_attacks']}")
    print(f"   - Accuracy drop: {metrics['accuracy_drop']*100:.2f}%")
    print(f"\n3. PERTURBAZIONI:")
    print(f"   - L2 medio: {metrics['l2_mean']:.6f}")
    print(f"   - L-inf medio: {metrics['linf_mean']:.6f}")
    print(f"   - Feature modificate: {metrics['l0_mean']:.2f}")
    print(f"\n4. QUERY UTILIZZATE:")
    print(f"   - Budget max: {max_eval} query/campione")
    print(f"   - Query totali (stimate): ~{max_eval * len(X_attacks_only)}")
    print(f"={'='*80}\n")
    
    results = {
        'metrics': metrics,
        'X_adv': X_adv_full,
        'attack_name': f'WhiteBox_HSJ_L{norm}',
        'config': {
            'max_iter': max_iter,
            'max_eval': max_eval,
            'norm': f'L{norm}'
        }
    }
    
    return results


def main():
    """Funzione principale per esecuzione da linea di comando."""
    parser = argparse.ArgumentParser(
        description="Attacco White-Box HopSkipJump su Random Forest federato",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:

  # Esecuzione base
  %(prog)s --model-path models/federated_rf_global_20251121_024044.pkl

  # Configurazione custom
  %(prog)s --model-path models/model.pkl --max-iter 200 --max-eval 20000

  # Minimizza L-inf
  %(prog)s --model-path models/model.pkl --norm inf

PARAMETRI RACCOMANDATI WHITE-BOX:
  - max-iter: 100 (convergenza accurata)
  - max-eval: 10000 (budget generoso)
  - norm: 2 (L2)
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path al modello Random Forest federato (.pkl)'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Max iterazioni HSJ (default: 100)'
    )
    
    parser.add_argument(
        '--max-eval',
        type=int,
        default=10000,
        help='Max query al modello (default: 10000)'
    )
    
    parser.add_argument(
        '--init-eval',
        type=int,
        default=100,
        help='Query per inizializzazione (default: 100)'
    )
    
    parser.add_argument(
        '--norm',
        type=str,
        choices=['2', 'inf'],
        default='2',
        help='Norma da minimizzare (default: 2)'
    )
    
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client per test (default: 1 13)'
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
    
    # Converti norm
    norm_value = 2 if args.norm == '2' else np.inf
    
    # Esegui attacco
    results = run_whitebox_hsj_attack(
        model_path=args.model_path,
        test_clients=args.test_clients,
        max_iter=args.max_iter,
        max_eval=args.max_eval,
        init_eval=args.init_eval,
        norm=norm_value,
        save_results=args.save_results or True,
        verbose=args.verbose
    )
    
    if results is None:
        print("\n❌ Attacco fallito.")
        sys.exit(1)
    else:
        print(f"\n✅ Attacco completato!")
        print(f"   ASR: {results['metrics']['asr']*100:.2f}%")


if __name__ == "__main__":
    main()