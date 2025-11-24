"""
attacks/1_whitebox_decision_tree_attack_ART.py

Attacco White-Box Decision Tree per Random Forest usando ART.

STRATEGIA:
Usa DecisionTreeAttack di ART sui singoli alberi del Random Forest,
aggregando intelligentemente le perturbazioni.

APPROCCIO:
1. Identifica alberi che votano "Attack"
2. Per ogni albero, calcola perturbazione con DecisionTreeAttack di ART
3. Seleziona N alberi più facili da flippare (norma minima)
4. Aggrega perturbazioni usando mediana
5. Applica perturbazione aggregata

AUTORE: Carmine Cataldo
DATA: 2025-01-21
"""

import numpy as np
import sys
import os
import argparse
from typing import List, Tuple

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import librerie ART - ✅ CORREZIONE: Nome corretto è SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification import SklearnClassifier  # ✅ CORREZIONE: Nome corretto

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


def generate_perturbation_for_single_tree(tree, X_sample, epsilon, verbose=False):
    """
    Genera una perturbazione per un singolo albero usando DecisionTreeAttack di ART.
    
    SPIEGAZIONE:
    DecisionTreeAttack di ART analizza la struttura interna di un albero decisionale
    e calcola la perturbazione minima necessaria per cambiare la predizione.
    
    Processo:
    1. Wrap dell'albero con SklearnClassifier per compatibilità ART
    2. Creazione attacco DecisionTreeAttack con epsilon specificato
    3. Generazione perturbazione che cambia la predizione dell'albero
    4. Verifica se l'albero è stato effettivamente "flippato"
    
    Args:
        tree: Singolo DecisionTreeClassifier dal Random Forest
        X_sample: Campione originale (1D numpy array, shape: [n_features])
        epsilon: Budget perturbazione (offset parameter di DecisionTreeAttack)
        verbose: Se True, stampa informazioni dettagliate
        
    Returns:
        perturbation: Vettore perturbazione (1D numpy array, shape: [n_features])
        success: Boolean, True se l'albero viene flippato con successo
        
    Example:
        >>> tree = random_forest.estimators_[0]
        >>> X_sample = X_test[0]
        >>> perturbation, success = generate_perturbation_for_single_tree(tree, X_sample, epsilon=0.01)
        >>> if success:
        >>>     print(f"Albero flippato con perturbazione L2={np.linalg.norm(perturbation):.6f}")
    """
    try:
        # ✅ CORREZIONE: Wrap del singolo albero con SklearnClassifier
        # SklearnClassifier è il wrapper generico di ART per modelli scikit-learn
        art_tree = SklearnClassifier(model=tree)
        
        # Crea attacco DecisionTree di ART
        # offset: controlla la magnitudine della perturbazione
        attack = DecisionTreeAttack(
            classifier=art_tree,
            offset=epsilon
        )
        
        # Genera adversarial example
        # IMPORTANTE: DecisionTreeAttack.generate() richiede input 2D (batch)
        # Convertiamo X_sample da 1D a 2D usando np.array([X_sample])
        X_adv = attack.generate(x=np.array([X_sample]))  # Input: (1, n_features)
        
        # Calcola perturbazione: differenza tra adversarial e originale
        perturbation = X_adv[0] - X_sample  # Estrai da batch: (n_features,)
        
        # Verifica se l'albero è stato flippato
        original_pred = tree.predict([X_sample])[0]  # Predizione originale
        new_pred = tree.predict(X_adv)[0]            # Predizione adversarial
        success = (original_pred != new_pred)        # Success se diverso
        
        if verbose:
            print(f"    Perturbazione generata: L2={np.linalg.norm(perturbation):.6f}, Flipped={success}")
            print(f"    Predizione: {original_pred} → {new_pred}")
        
        return perturbation, success
        
    except Exception as e:
        if verbose:
            print(f"    ⚠️ Errore generazione perturbazione: {e}")
        return None, False


def attack_random_forest_with_art(model, X_sample, epsilon, target_class=0, verbose=False):
    """
    Attacca un Random Forest usando DecisionTreeAttack di ART sui singoli alberi.
    
    STRATEGIA DETTAGLIATA:
    
    Random Forest predizione = VOTO DI MAGGIORANZA di N alberi
    
    Per evadere il sistema (cambiare predizione da Attack → Natural):
    1. Conta i voti correnti: quanti alberi votano "Attack" vs "Natural"
    2. Calcola quanti alberi dobbiamo "flippare" per ottenere maggioranza "Natural"
    3. Per ogni albero che vota "Attack", genera perturbazione con ART
    4. Seleziona gli N alberi più facili da flippare (perturbazione minima)
    5. Aggrega le N perturbazioni usando mediana (robusto agli outlier)
    6. Applica perturbazione aggregata al campione
    
    ESEMPIO PRATICO:
    Random Forest con 100 alberi:
    - 65 alberi votano "Attack" (1)
    - 35 alberi votano "Natural" (0)
    - Maggioranza: Attack
    
    Per cambiare a Natural:
    - Serve maggioranza ≥ 51 voti "Natural"
    - Attualmente: 35 voti "Natural"
    - Alberi da flippare: 51 - 35 = 16
    
    Strategia:
    - Genera perturbazione per tutti i 65 alberi "Attack"
    - Ordina per norma L2 crescente
    - Seleziona i 16 alberi più facili da flippare
    - Aggrega le 16 perturbazioni con mediana
    
    Args:
        model: RandomForestClassifier completo
        X_sample: Campione originale (1D numpy array)
        epsilon: Budget perturbazione
        target_class: Classe target (default: 0 = Natural)
        verbose: Se True, stampa informazioni dettagliate
        
    Returns:
        X_adv: Campione adversarial
        success: Boolean, True se l'attacco ha successo (predizione cambiata)
        
    Example:
        >>> X_adv, success = attack_random_forest_with_art(rf_model, X_test[0], epsilon=0.01)
        >>> if success:
        >>>     print("Attacco riuscito!")
    """
    if verbose:
        print(f"\n[ART Attack] Generazione perturbazione per campione...")
    
    # STEP 1: Predizione corrente Random Forest
    current_pred = model.predict([X_sample])[0]
    
    if current_pred == target_class:
        # Già classificato come target, non serve attaccare
        if verbose:
            print(f"  Campione già classificato come classe {target_class}, skip")
        return X_sample, True
    
    # STEP 2: Identifica voti di ogni albero
    # Ogni albero vota 0 (Natural) o 1 (Attack)
    tree_votes = np.array([tree.predict([X_sample])[0] for tree in model.estimators_])
    
    # Conta voti
    attack_votes = int(np.sum(tree_votes == 1))  # Alberi che votano "Attack"
    natural_votes = len(tree_votes) - attack_votes
    
    # STEP 3: Calcola quanti alberi serve flippare
    # Per ottenere maggioranza Natural serve: n_trees // 2 + 1 voti
    majority_threshold = len(tree_votes) // 2 + 1
    trees_to_flip = majority_threshold - natural_votes
    
    if trees_to_flip <= 0:
        # Già maggioranza Natural (non dovrebbe accadere)
        if verbose:
            print(f"  Già maggioranza Natural, skip")
        return X_sample, True
    
    if verbose:
        print(f"  Voti correnti: {attack_votes} Attack, {natural_votes} Natural")
        print(f"  Maggioranza richiesta: {majority_threshold} voti")
        print(f"  Alberi da flippare: {trees_to_flip}")
    
    # STEP 4: Per ogni albero che vota "Attack", genera perturbazione con ART
    tree_perturbations = []
    
    for i in range(len(tree_votes)):
        if tree_votes[i] == 1:  # Albero vota "Attack"
            tree = model.estimators_[i]
            
            if verbose:
                print(f"  Generazione perturbazione albero {i+1}...")
            
            perturbation, success = generate_perturbation_for_single_tree(
                tree, X_sample, epsilon, verbose=verbose
            )
            
            if perturbation is not None and success:
                # Calcola norma L2 per ordinamento
                norm_L2 = np.linalg.norm(perturbation)
                tree_perturbations.append((i, perturbation, norm_L2))
    
    if not tree_perturbations:
        # Nessuna perturbazione generata con successo
        if verbose:
            print(f"  ⚠️ Nessuna perturbazione valida generata")
        return X_sample, False
    
    # STEP 5: Ordina per norma crescente (più facili da flippare)
    tree_perturbations.sort(key=lambda x: x[2])  # Ordina per x[2] = norm_L2
    
    if verbose:
        print(f"  Perturbazioni valide generate: {len(tree_perturbations)}")
        print(f"  Range norme L2: [{tree_perturbations[0][2]:.6f}, {tree_perturbations[-1][2]:.6f}]")
    
    # STEP 6: Seleziona i N alberi più facili
    if len(tree_perturbations) >= trees_to_flip:
        selected_perturbations = tree_perturbations[:trees_to_flip]
    else:
        # Usa tutte le perturbazioni disponibili se non ne abbiamo abbastanza
        selected_perturbations = tree_perturbations
        if verbose:
            print(f"  ⚠️ Solo {len(tree_perturbations)} perturbazioni disponibili (richiesti {trees_to_flip})")
    
    # STEP 7: Aggrega usando MEDIANA (robusto agli outlier)
    # Mediana: valore centrale, meno sensibile a perturbazioni estreme
    perturbations_array = np.array([pert for _, pert, _ in selected_perturbations])
    aggregated_perturbation = np.median(perturbations_array, axis=0)
    
    # Clippa entro epsilon per rispettare budget
    aggregated_perturbation = np.clip(aggregated_perturbation, -epsilon, epsilon)
    
    # STEP 8: Applica perturbazione
    X_adv = X_sample + aggregated_perturbation
    
    # STEP 9: Verifica successo
    new_pred = model.predict([X_adv])[0]
    success = (new_pred == target_class)
    
    if verbose:
        print(f"  Perturbazione aggregata: L2={np.linalg.norm(aggregated_perturbation):.6f}")
        print(f"  Predizione originale: {current_pred}, Predizione adversarial: {new_pred}")
        print(f"  Successo: {success}")
    
    return X_adv, success


def run_whitebox_art_attack(
    model_path,
    test_clients=[1, 13],
    epsilons=[0.001, 0.005, 0.01, 0.05],
    save_results=True,
    verbose=False
):
    """
    Esegue attacco White-Box usando DecisionTreeAttack di ART sui singoli alberi.
    
    WORKFLOW COMPLETO:
    1. Carica modello Random Forest federato salvato
    2. Carica dati di test (client 1 e 13)
    3. Applica preprocessing identico al federato
    4. Per ogni epsilon:
       a. Per ogni campione di attacco:
          - Genera perturbazione con attack_random_forest_with_art()
       b. Applica vincoli fisici SmartGrid
       c. Valuta efficacia (ASR, L2, L0, metriche)
    5. Salva report completo
    
    Args:
        model_path: Path al modello Random Forest federato (.pkl)
        test_clients: Lista di client da usare per test (default: [1, 13])
        epsilons: Lista di epsilon da testare
        save_results: Se True, salva risultati in file
        verbose: Se True, stampa informazioni dettagliate
        
    Returns:
        results: Dictionary con risultati per ogni epsilon
    """
    print("="*80)
    print("🔴 ATTACCO WHITE-BOX: DECISION TREE ATTACK (ART su singoli alberi)")
    print("="*80)
    print(f"Modello: {model_path}")
    print(f"Test clients: {test_clients}")
    print(f"Epsilon da testare: {epsilons}")
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
    
    # ========== STEP 4: OTTIENI VINCOLI FISICI ==========
    constraints = get_smartgrid_physical_constraints(X_test)
    
    # ========== STEP 5: SELEZIONA SOLO CAMPIONI DI ATTACCO ==========
    print("\n[STEP 4] Selezione campioni di attacco...")
    X_attacks_only, y_attacks_only, attack_indices = select_attack_samples(
        X_test, y_test, target_class=1
    )
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni di attacco: {len(X_attacks_only)}")
    print(f"  - Campioni naturali: {(y_test == 0).sum()}")
    
    # ========== STEP 6: ESEGUI ATTACCO PER OGNI EPSILON ==========
    results = {}
    
    for i, epsilon in enumerate(epsilons):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}/{len(epsilons)}: EPSILON = {epsilon}")
        print(f"{'='*80}")
        
        # ===== 6.1: Genera adversarial examples =====
        print(f"\n[Generazione] Adversarial examples con ART DecisionTreeAttack (epsilon={epsilon})...")
        
        X_attacks_adv = []
        success_count = 0
        
        for j, X_sample in enumerate(X_attacks_only):
            # Stampa progresso ogni 100 campioni
            if verbose or (j > 0 and j % 100 == 0):
                print(f"  Processando campione {j+1}/{len(X_attacks_only)}...")
            
            # Genera perturbazione con ART
            X_adv, success = attack_random_forest_with_art(
                model, X_sample, epsilon, target_class=0, 
                verbose=(verbose and j < 3)  # Verbose solo per primi 3 campioni
            )
            
            X_attacks_adv.append(X_adv)
            if success:
                success_count += 1
        
        X_attacks_adv = np.array(X_attacks_adv)
        
        print(f"✅ Generati {len(X_attacks_adv)} esempi adversarial")
        print(f"  Successi individuali: {success_count}/{len(X_attacks_adv)} ({success_count/len(X_attacks_adv)*100:.2f}%)")
        
        # ===== 6.2: Applica vincoli fisici SmartGrid =====
        print(f"\n[Vincoli] Applicazione vincoli fisici SmartGrid...")
        
        X_attacks_adv_constrained = apply_physical_constraints(
            X_attacks_adv,
            X_attacks_only,
            constraints,
            max_perturbation_linf=epsilon * 10  # Margine tolleranza
        )
        
        # ===== 6.3: Ricostruisci dataset completo =====
        # Dataset completo = campioni natural (invariati) + campioni attack (adversarial)
        X_adv_full = X_test.copy()
        X_adv_full[attack_indices] = X_attacks_adv_constrained
        
        # ===== 6.4: Valuta efficacia attacco =====
        print(f"\n[Valutazione] Calcolo metriche per epsilon={epsilon}...")
        
        metrics = evaluate_attack(
            model,
            X_test,      # Dataset originale completo
            y_test,
            X_adv_full,  # Dataset adversarial completo
            attack_name=f"WhiteBox_ART_DecisionTree_eps_{epsilon}"
        )
        
        # ===== 6.5: Stampa report =====
        if verbose or i < 2:  # Report dettagliato per primi 2 epsilon
            print_attack_report(metrics)
        else:
            # Report compatto per epsilon successivi
            print(f"\n✅ Epsilon {epsilon}: ASR = {metrics['asr']*100:.2f}%, L2 = {metrics['l2_mean']:.6f}")
        
        # ===== 6.6: Salva risultati =====
        results[epsilon] = {
            'X_adv': X_adv_full,
            'metrics': metrics
        }
    
    # ========== STEP 7: SALVA RISULTATI FINALI ==========
    if save_results and results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        metrics_list = [results[eps]['metrics'] for eps in epsilons if eps in results]
        X_adv_dict = {eps: results[eps]['X_adv'] for eps in epsilons if eps in results}
        
        save_attack_results(
            metrics_list,
            X_test,
            X_adv_dict,
            epsilons_tested=[eps for eps in epsilons if eps in results],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== STEP 8: SUMMARY FINALE ==========
    if results:
        print(f"\n{'='*80}")
        print(f"✅ ATTACCO WHITE-BOX ART COMPLETATO")
        print(f"{'='*80}")
        print(f"\nRIEPILOGO RISULTATI:\n")
        print(f"{'Epsilon':<12} {'ASR (%)':<12} {'L2 medio':<15} {'Feature mod.':<15}")
        print(f"{'-'*60}")
        
        for eps in epsilons:
            if eps in results:
                m = results[eps]['metrics']
                print(f"{eps:<12} {m['asr']*100:<12.2f} {m['l2_mean']:<15.6f} {m['l0_mean']:<15.2f}")
        
        print(f"{'='*80}\n")
    else:
        print(f"\n❌ NESSUN RISULTATO DISPONIBILE")
    
    return results


def main():
    """
    Funzione principale per esecuzione da linea di comando.
    
    Gestisce parsing argomenti e invoca l'attacco white-box ART.
    """
    parser = argparse.ArgumentParser(
        description="Attacco White-Box usando ART DecisionTreeAttack su Random Forest"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path al modello Random Forest federato (.pkl)'
    )
    
    parser.add_argument(
        '--epsilons',
        type=float,
        nargs='+',
        default=[0.001, 0.005, 0.01, 0.05],
        help='Lista di epsilon da testare'
    )
    
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client da usare per test'
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
    run_whitebox_art_attack(
        model_path=args.model_path,
        test_clients=args.test_clients,
        epsilons=args.epsilons,
        save_results=args.save_results or True,  # Default True
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()