"""
attacks/1_whitebox_hopskipjump_attack.py

Attacco White-Box usando HopSkipJump per Random Forest federato SmartGrid.

DESCRIZIONE:
HopSkipJump è un attacco decision-based (black-box) che usa solo le predizioni
del modello (nessun gradiente richiesto). Perfetto per Random Forest!

In modalità WHITE-BOX:
- Abbiamo accesso completo al modello salvato
- Conosciamo l'architettura (Random Forest con 100 alberi)
- Ottimizziamo i parametri basandoci sulla struttura del modello

ALGORITMO HOPSKIPJUMP:
1. Inizializzazione: Parte da un esempio adversarial casuale
2. Boundary walk: Si muove lungo il confine decisionale
3. Step towards original: Si avvicina all'esempio originale mantenendo evasione
4. Iterazione: Ripete fino a convergenza o max iterations

RIFERIMENTI:
- Paper: Chen et al. (2020) - "HopSkipJump: A Query-Efficient Decision-Based Attack"
- ART Documentation: https://adversarial-robustness-toolbox.readthedocs.io/

UTILIZZO:
    python attacks/1_whitebox_hopskipjump_attack.py \
        --model-path models/federated_rf_global_20251121_024044.pkl \
        --max-iter 50 \
        --max-eval 5000 \
        --test-clients 1 13 \
        --save-results

PARAMETRI CHIAVE:
    --max-iter: Numero massimo di iterazioni HopSkipJump (default: 50)
    --max-eval: Numero massimo di query al modello (default: 5000)
    --init-eval: Query per inizializzazione (default: 100)
    --norm: Norma da minimizzare ('2' o 'inf', default: '2')

OUTPUT:
    - Report con ASR, metriche perturbazione, confusion matrix
    - File di testo con risultati salvati in attacks/results/
    - Esempi adversarial generati

AUTORE: Carmine Cataldo
DATA: 2025-01-21
"""

import numpy as np
import sys
import os
import argparse
from typing import Tuple, List

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import librerie ART
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


def run_whitebox_hopskipjump_attack(
    model_path,
    test_clients=[1, 13],
    max_iter=50,
    max_eval=5000,
    init_eval=100,
    norm=2,
    save_results=True,
    verbose=False
):
    """
    Esegue attacco White-Box usando HopSkipJump su Random Forest federato.
    
    WORKFLOW:
    1. Carica modello Random Forest federato salvato
    2. Carica dati di test (client 1 e 13)
    3. Applica preprocessing identico al federato
    4. Configura HopSkipJump con parametri ottimizzati per Random Forest
    5. Genera esempi adversarial
    6. Applica vincoli fisici SmartGrid
    7. Valuta efficacia (ASR, metriche)
    8. Salva report completo
    
    SPIEGAZIONE PARAMETRI HOPSKIPJUMP:
    
    max_iter: Numero massimo di iterazioni dell'algoritmo
        - Più alto = perturbazioni più piccole ma più lento
        - Raccomandato per Random Forest: 50 (buon compromesso)
    
    max_eval: Numero massimo di query al modello
        - HopSkipJump usa molte query per esplorare il boundary
        - Raccomandato: 5000 (permette convergenza)
    
    init_eval: Query per trovare inizializzazione valida
        - Numero di tentativi per trovare un punto adversarial iniziale
        - Raccomandato: 100
    
    norm: Norma da minimizzare (2 = L2, np.inf = L-inf)
        - L2: Distanza euclidea (standard)
        - L-inf: Massimo cambiamento su singola feature
    
    Args:
        model_path: Path al modello Random Forest federato (.pkl)
        test_clients: Lista di client da usare per test (default: [1, 13])
        max_iter: Max iterazioni HopSkipJump (default: 50)
        max_eval: Max query al modello (default: 5000)
        init_eval: Query per inizializzazione (default: 100)
        norm: Norma da minimizzare (default: 2)
        save_results: Se True, salva risultati in file (default: True)
        verbose: Se True, stampa informazioni dettagliate (default: False)
        
    Returns:
        results: Dictionary con risultati dell'attacco
        
    Example:
        >>> results = run_whitebox_hopskipjump_attack(
        ...     model_path='models/federated_rf_global_20251121_024044.pkl',
        ...     max_iter=50,
        ...     max_eval=5000
        ... )
        >>> print(f"ASR: {results['metrics']['asr']*100:.2f}%")
    """
    print("="*80)
    print("🔴 ATTACCO WHITE-BOX: HOPSKIPJUMP (Decision-Based)")
    print("="*80)
    print(f"Modello: {model_path}")
    print(f"Test clients: {test_clients}")
    print(f"Configurazione HopSkipJump:")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Max evaluations: {max_eval}")
    print(f"  - Init evaluations: {init_eval}")
    print(f"  - Norm: L{norm}")
    print("="*80 + "\n")
    
    # Imposta seed per riproducibilità
    set_reproducibility_seeds(42)
    
    # ========== STEP 1: CARICA MODELLO FEDERATO ==========
    print("\n[STEP 1] Caricamento modello Random Forest federato...")
    model = load_federated_model(model_path)
    
    # ========== STEP 2: CARICA DATI DI TEST ==========
    print("\n[STEP 2] Caricamento dati di test...")
    X_test_raw, y_test, test_info = load_test_data_from_clients(
        client_ids=test_clients
    )
    
    # ========== STEP 3: PREPROCESSING ==========
    print("\n[STEP 3] Applicazione preprocessing...")
    X_test, _ = apply_preprocessing_pipeline(X_test_raw, fit_on_data=X_test_raw)
    
    # Verifica compatibilità dimensioni
    if X_test.shape[1] != model.n_features_in_:
        raise ValueError(
            f"❌ Incompatibilità feature: test={X_test.shape[1]}, modello={model.n_features_in_}"
        )
    
    print(f"✅ Preprocessing completato: {X_test.shape}")
    
    # ========== STEP 4: WRAP MODELLO PER ART ==========
    print("\n[STEP 4] Configurazione HopSkipJump per Random Forest...")
    
    # Wrap Random Forest per compatibilità ART
    art_classifier = SklearnClassifier(model=model)
    print(f"✅ Modello wrapped per ART")
    
    # ========== STEP 5: OTTIENI VINCOLI FISICI ==========
    constraints = get_smartgrid_physical_constraints(X_test)
    
    # ========== STEP 6: SELEZIONA SOLO CAMPIONI DI ATTACCO ==========
    print("\n[STEP 5] Selezione campioni di attacco...")
    X_attacks_only, y_attacks_only, attack_indices = select_attack_samples(
        X_test, y_test, target_class=1
    )
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni di attacco: {len(X_attacks_only)}")
    print(f"  - Campioni naturali: {(y_test == 0).sum()}")
    
    # ========== STEP 7: CONFIGURA HOPSKIPJUMP ==========
    print(f"\n[STEP 6] Creazione attacco HopSkipJump ottimizzato per Random Forest...")
    
    """
    SPIEGAZIONE CONFIGURAZIONE HOPSKIPJUMP PER RANDOM FOREST:
    
    Random Forest è un modello ENSEMBLE:
    - 100 alberi decisionali
    - Predizione = voto di maggioranza
    - Boundary decisionale = complesso e non-lineare
    
    Parametri ottimizzati:
    - targeted=False: Evasion attack (Attack → Natural)
    - norm=2: Minimizza distanza L2 (standard)
    - max_iter=50: Sufficiente per convergenza su Random Forest
    - max_eval=5000: Permette esplorazione adeguata del boundary
    - init_eval=100: Tentativi sufficienti per inizializzazione
    - init_size=100: Dimensione batch per ricerca iniziale
    """
    
    attack = HopSkipJump(
        classifier=art_classifier,
        targeted=False,          # Evasion attack non-targeted
        norm=norm,               # Norma da minimizzare (L2 o L-inf)
        max_iter=max_iter,       # Iterazioni massime
        max_eval=max_eval,       # Query massime al modello
        init_eval=init_eval,     # Query per inizializzazione
        init_size=100,           # Dimensione batch iniziale
        verbose=verbose          # Stampa progresso
    )
    
    print(f"✅ HopSkipJump configurato")
    print(f"   Questo attacco è OTTIMALE per Random Forest perché:")
    print(f"   - NON richiede gradienti (Random Forest non differenziabile)")
    print(f"   - Usa solo predizioni (decision-based)")
    print(f"   - Esplora il boundary in modo intelligente")
    print(f"   - Converge verso perturbazioni minime")
    
    # ========== STEP 8: GENERA ADVERSARIAL EXAMPLES ==========
    print(f"\n[STEP 7] Generazione adversarial examples con HopSkipJump...")
    print(f"   ⚠️ ATTENZIONE: HopSkipJump è LENTO (usa molte query)")
    print(f"   Tempo stimato: ~{len(X_attacks_only) * 0.5:.0f} secondi per {len(X_attacks_only)} campioni")
    
    try:
        # Genera adversarial examples
        X_attacks_adv = attack.generate(x=X_attacks_only)
        print(f"✅ Generati {len(X_attacks_adv)} esempi adversarial")
        
    except Exception as e:
        print(f"❌ Errore generazione adversarial examples: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== STEP 9: APPLICA VINCOLI FISICI SMARTGRID ==========
    print(f"\n[STEP 8] Applicazione vincoli fisici SmartGrid...")
    
    """
    VINCOLI FISICI SMARTGRID:
    
    SmartGrid rappresenta misurazioni fisiche (voltaggio, corrente, frequenza).
    Per garantire che gli esempi adversarial siano REALISTICI:
    
    1. Range fisico: Ogni feature deve essere tra min e max osservati nel dataset
    2. Perturbazione limitata: La modifica non deve essere eccessiva
    
    HopSkipJump già minimizza la perturbazione, ma applichiamo vincoli
    per garantire che rimanga nei range fisici plausibili.
    """
    
    # Calcola perturbazione media prima dei vincoli
    perturbation_before = X_attacks_adv - X_attacks_only
    l2_before = np.mean(np.linalg.norm(perturbation_before, axis=1))
    linf_before = np.mean(np.max(np.abs(perturbation_before), axis=1))
    
    print(f"   Prima dei vincoli: L2={l2_before:.6f}, L-inf={linf_before:.6f}")
    
    # Applica vincoli fisici
    X_attacks_adv_constrained = apply_physical_constraints(
        X_attacks_adv,
        X_attacks_only,
        constraints,
        max_perturbation_linf=None  # HopSkipJump già ottimizza la perturbazione
    )
    
    # Calcola perturbazione media dopo i vincoli
    perturbation_after = X_attacks_adv_constrained - X_attacks_only
    l2_after = np.mean(np.linalg.norm(perturbation_after, axis=1))
    linf_after = np.mean(np.max(np.abs(perturbation_after), axis=1))
    
    print(f"   Dopo vincoli: L2={l2_after:.6f}, L-inf={linf_after:.6f}")
    print(f"   Impatto vincoli: ΔL2={l2_after-l2_before:.6f}, ΔL-inf={linf_after-linf_before:.6f}")
    
    # ========== STEP 10: RICOSTRUISCI DATASET COMPLETO ==========
    # (attacchi adversarial + campioni naturali non perturbati)
    X_adv_full = X_test.copy()
    X_adv_full[attack_indices] = X_attacks_adv_constrained
    
    # ========== STEP 11: VALUTA EFFICACIA ATTACCO ==========
    print(f"\n[STEP 9] Valutazione efficacia attacco HopSkipJump...")
    
    metrics = evaluate_attack(
        model,
        X_test,  # Dataset originale completo
        y_test,
        X_adv_full,  # Dataset adversarial completo
        attack_name=f"WhiteBox_HopSkipJump_L{norm}"
    )
    
    # ========== STEP 12: STAMPA REPORT DETTAGLIATO ==========
    print_attack_report(metrics)
    
    # ========== STEP 13: SALVA RISULTATI ==========
    if save_results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        # Salva con metadata HopSkipJump
        metadata = {
            'attack_type': 'WhiteBox_HopSkipJump',
            'max_iter': max_iter,
            'max_eval': max_eval,
            'init_eval': init_eval,
            'norm': f'L{norm}',
            'model_type': 'Random Forest Federato',
            'n_trees': len(model.estimators_),
            'n_features': model.n_features_in_
        }
        
        save_attack_results(
            [metrics],
            X_test,
            {'hopskipjump': X_adv_full},
            epsilons_tested=[f'HopSkipJump_L{norm}'],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== STEP 14: SUMMARY FINALE ==========
    print(f"\n{'='*80}")
    print(f"✅ ATTACCO WHITE-BOX HOPSKIPJUMP COMPLETATO")
    print(f"{'='*80}")
    print(f"\nRISULTATI FINALI:\n")
    print(f"Attack Success Rate (ASR): {metrics['asr']*100:.2f}%")
    print(f"Evasioni riuscite: {metrics['successful_evasions']}/{metrics['total_attacks']}")
    print(f"Accuracy drop: {metrics['accuracy_drop']*100:.2f}%")
    print(f"\nPerturbazioni:")
    print(f"  L2 medio: {metrics['l2_mean']:.6f}")
    print(f"  L-inf medio: {metrics['linf_mean']:.6f}")
    print(f"  Feature modificate (L0): {metrics['l0_mean']:.2f}")
    print(f"\nQuery utilizzate: ~{max_eval * len(X_attacks_only)} (max)")
    print(f"Tempo esecuzione: stimato ~{len(X_attacks_only) * 0.5:.0f} secondi")
    print(f"{'='*80}\n")
    
    # Ritorna risultati completi
    results = {
        'X_adv': X_adv_full,
        'metrics': metrics,
        'metadata': metadata,
        'attack_name': f'WhiteBox_HopSkipJump_L{norm}'
    }
    
    return results


def main():
    """
    Funzione principale per esecuzione da linea di comando.
    
    ESEMPI D'USO:
    
    # Esecuzione base
    python attacks/1_whitebox_hopskipjump_attack.py \
        --model-path models/federated_rf_global_20251121_024044.pkl
    
    # Con parametri personalizzati
    python attacks/1_whitebox_hopskipjump_attack.py \
        --model-path models/federated_rf_global_20251121_024044.pkl \
        --max-iter 100 \
        --max-eval 10000 \
        --norm inf
    
    # Modalità verbose
    python attacks/1_whitebox_hopskipjump_attack.py \
        --model-path models/federated_rf_global_20251121_024044.pkl \
        --verbose
    """
    parser = argparse.ArgumentParser(
        description="Attacco White-Box HopSkipJump su Random Forest federato SmartGrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:

  # Esecuzione base
  %(prog)s --model-path models/federated_rf_global_20251121_024044.pkl

  # Attacco più aggressivo (più iterazioni)
  %(prog)s --model-path models/model.pkl --max-iter 100 --max-eval 10000

  # Minimizza L-inf invece di L2
  %(prog)s --model-path models/model.pkl --norm inf

PARAMETRI RACCOMANDATI PER RANDOM FOREST:
  - max-iter: 50 (buon compromesso velocità/efficacia)
  - max-eval: 5000 (permette convergenza)
  - norm: 2 (L2, standard)
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path al modello Random Forest federato (.pkl). Es: models/federated_rf_global_20251121_024044.pkl'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50,
        help='Numero massimo di iterazioni HopSkipJump (default: 50). Più alto = perturbazioni più piccole ma più lento.'
    )
    
    parser.add_argument(
        '--max-eval',
        type=int,
        default=5000,
        help='Numero massimo di query al modello (default: 5000). Controlla il budget computazionale.'
    )
    
    parser.add_argument(
        '--init-eval',
        type=int,
        default=100,
        help='Query per trovare inizializzazione (default: 100).'
    )
    
    parser.add_argument(
        '--norm',
        type=str,
        choices=['2', 'inf'],
        default='2',
        help='Norma da minimizzare (default: 2). Opzioni: 2 (L2), inf (L-inf)'
    )
    
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client da usare per test (default: 1 13). Es: --test-clients 1 13'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Salva risultati in file (attivo di default se non specificato)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Stampa informazioni dettagliate durante esecuzione (utile per debug)'
    )
    
    args = parser.parse_args()
    
    # Converti norm da stringa a numero
    norm_value = 2 if args.norm == '2' else np.inf
    
    # Esegui attacco
    results = run_whitebox_hopskipjump_attack(
        model_path=args.model_path,
        test_clients=args.test_clients,
        max_iter=args.max_iter,
        max_eval=args.max_eval,
        init_eval=args.init_eval,
        norm=norm_value,
        save_results=args.save_results or True,  # Default True
        verbose=args.verbose
    )
    
    if results is None:
        print("\n❌ Attacco fallito. Controlla gli errori sopra.")
        sys.exit(1)
    else:
        print(f"\n✅ Attacco completato con successo!")
        print(f"   ASR finale: {results['metrics']['asr']*100:.2f}%")
        print(f"   L2 medio: {results['metrics']['l2_mean']:.6f}")


if __name__ == "__main__":
    main()