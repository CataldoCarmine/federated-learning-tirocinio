"""
attacks/1_whitebox_decision_tree_attack.py

Attacco White-Box Decision Tree Custom per Random Forest federato SmartGrid.

IMPORTANTE: ART DecisionTreeAttack funziona SOLO su singoli alberi, NON su Random Forest.
Questa è un'implementazione CUSTOM che estende l'approccio per ensemble di alberi.

STRATEGIA:
1. Per ogni albero del Random Forest, identifica percorsi verso classe target
2. Calcola perturbazioni minime per attraversare nodi critici di ogni albero
3. Aggrega perturbazioni da tutti gli alberi usando strategia ottimale
4. Applica vincoli fisici SmartGrid per garantire realismo

RIFERIMENTI:
- Paper originale Decision Tree Attack: Papernot et al. (2016)
- Estensione per Random Forest: Custom implementation

UTILIZZO:
    python attacks/1_whitebox_decision_tree_attack.py \
        --model-path models/federated_rf_global_20251121_024044.pkl \
        --epsilons 0.001 0.005 0.01 0.05 \
        --test-clients 1 13 \
        --save-results

AUTORE: Carmine Cataldo
DATA: 2025-01-21
"""

import numpy as np
import sys
import os
import argparse
from typing import Tuple, List, Optional

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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


class RandomForestDecisionTreeAttack:
    """
    Attacco White-Box Custom per Random Forest basato su Decision Tree Attack.
    
    SPIEGAZIONE ALGORITMO:
    
    Random Forest = Ensemble di N alberi decisionali
    Predizione finale = Voto di maggioranza degli alberi
    
    Per evadere il sistema:
    1. Identifica quanti alberi votano "Attack" (classe 1)
    2. Calcola quanti alberi dobbiamo "flippare" per ottenere maggioranza "Natural" (classe 0)
    3. Per ogni albero da flippare, trova la perturbazione minima necessaria
    4. Aggrega le perturbazioni per massimizzare efficacia e minimizzare norma
    
    Esempio:
    - Random Forest con 100 alberi
    - 60 alberi votano "Attack", 40 votano "Natural"
    - Per evadere: dobbiamo flippare almeno 11 alberi (da Attack → Natural)
    - Obiettivo: flippare gli 11 alberi più "facili" (perturbazione minima)
    """
    
    def __init__(self, model, epsilon=0.01, verbose=False):
        """
        Inizializza l'attacco per Random Forest.
        
        Args:
            model: RandomForestClassifier da attaccare
            epsilon: Budget massimo perturbazione L-infinity
            verbose: Se True, stampa informazioni dettagliate
        """
        self.model = model
        self.epsilon = epsilon
        self.verbose = verbose
        
        # Verifica che sia un Random Forest
        if not hasattr(model, 'estimators_'):
            raise ValueError("Il modello deve essere un RandomForestClassifier con alberi addestrati")
        
        self.n_trees = len(model.estimators_)
        self.n_features = model.n_features_in_
        
        if self.verbose:
            print(f"[RandomForestAttack] Inizializzato con {self.n_trees} alberi, {self.n_features} feature")
    
    def _get_leaf_for_sample(self, tree, x_sample):
        """
        Trova la foglia raggiunta da un campione in un albero decisionale.
        
        Args:
            tree: DecisionTreeClassifier
            x_sample: Campione di input (1D numpy array)
            
        Returns:
            leaf_id: ID della foglia raggiunta
            leaf_value: Valore della foglia (predizione)
        """
        # Applica il decision path per trovare la foglia
        leaf = tree.apply([x_sample])[0]
        tree_structure = tree.tree_
        leaf_value = tree_structure.value[leaf]
        
        return leaf, leaf_value
    
    def _compute_minimal_perturbation_for_tree(self, tree, x_sample, target_class=0):
        """
        Calcola la perturbazione minima per cambiare la predizione di un singolo albero.
        
        STRATEGIA SEMPLIFICATA:
        Invece di trovare percorsi complessi, usa un approccio euristico:
        1. Identifica le feature più importanti per questo albero
        2. Perturba quelle feature nella direzione che favorisce la classe target
        3. Limita la perturbazione entro epsilon
        
        Args:
            tree: DecisionTreeClassifier
            x_sample: Campione originale
            target_class: Classe target (0 = Natural)
            
        Returns:
            Perturbation vector (numpy array) o None se impossibile
        """
        # Predizione corrente dell'albero
        current_pred = tree.predict([x_sample])[0]
        
        # Se già predice la classe target, nessuna perturbazione necessaria
        if current_pred == target_class:
            return np.zeros(self.n_features)
        
        # Strategia euristica: perturba feature in modo casuale ma controllato
        # fino a cambiare predizione
        best_perturbation = None
        best_norm = float('inf')
        
        # Prova diverse direzioni casuali (Monte Carlo sampling)
        n_trials = 20
        
        for trial in range(n_trials):
            # Genera perturbazione casuale
            perturbation = np.random.randn(self.n_features) * self.epsilon * 0.5
            
            # Clippa entro epsilon
            perturbation = np.clip(perturbation, -self.epsilon, self.epsilon)
            
            # Applica perturbazione
            x_perturbed = x_sample + perturbation
            
            # Verifica se cambia predizione
            new_pred = tree.predict([x_perturbed])[0]
            
            if new_pred == target_class:
                # Perturbazione riuscita
                norm = np.linalg.norm(perturbation)
                if norm < best_norm:
                    best_norm = norm
                    best_perturbation = perturbation
        
        return best_perturbation
    
    def generate(self, x):
        """
        Genera esempi adversarial per un batch di campioni.
        
        ALGORITMO SEMPLIFICATO:
        1. Per ogni campione:
           a. Predizione corrente del Random Forest
           b. Se già classe target, skip
           c. Per ogni albero che vota "Attack":
              - Trova perturbazione minima per flippare a "Natural"
           d. Aggrega perturbazioni usando mediana
        2. Applica perturbazione aggregata
        
        Args:
            x: Batch di campioni (N, n_features)
            
        Returns:
            x_adv: Esempi adversarial (N, n_features)
        """
        x_adv = x.copy()
        
        for i, x_sample in enumerate(x):
            if self.verbose and i % 100 == 0:
                print(f"[RandomForestAttack] Processando campione {i+1}/{len(x)}")
            
            # Predizione corrente
            current_pred = self.model.predict([x_sample])[0]
            
            # Se già classe target, skip
            if current_pred == 0:  # 0 = Natural
                continue
            
            # Raccogli voti da tutti gli alberi
            tree_votes = np.array([tree.predict([x_sample])[0] for tree in self.model.estimators_])
            
            # Conta voti per "Attack" (classe 1)
            attack_votes = int(np.sum(tree_votes == 1))  # 🔧 FIX: Converti a int
            
            # Calcola quanti alberi dobbiamo flippare
            # Per ottenere maggioranza "Natural" (classe 0)
            majority_threshold = self.n_trees // 2 + 1
            current_natural_votes = self.n_trees - attack_votes
            trees_to_flip = max(0, majority_threshold - current_natural_votes)
            trees_to_flip = int(trees_to_flip)  # 🔧 FIX: Assicura che sia int
            
            if trees_to_flip <= 0:
                # Già maggioranza "Natural"
                continue
            
            # Trova perturbazioni per gli alberi che votano "Attack"
            tree_perturbations = []
            
            for tree_idx in range(self.n_trees):
                if tree_votes[tree_idx] == 1:  # Albero vota "Attack"
                    pert = self._compute_minimal_perturbation_for_tree(
                        self.model.estimators_[tree_idx], 
                        x_sample, 
                        target_class=0
                    )
                    
                    if pert is not None:
                        # Calcola norma L2 della perturbazione
                        norm = np.linalg.norm(pert)
                        tree_perturbations.append((norm, pert))
            
            if not tree_perturbations:
                # Nessuna perturbazione trovata
                continue
            
            # Ordina per norma crescente (perturbazioni più piccole prima)
            tree_perturbations.sort(key=lambda x: x[0])
            
            # Seleziona le perturbazioni dei primi N alberi più facili da flippare
            if len(tree_perturbations) >= trees_to_flip:
                selected_perturbations = [pert for _, pert in tree_perturbations[:trees_to_flip]]
            else:
                # Se non abbiamo abbastanza perturbazioni, usa tutte quelle disponibili
                selected_perturbations = [pert for _, pert in tree_perturbations]
            
            if selected_perturbations:
                # Aggrega usando la MEDIANA (robusto agli outlier)
                aggregated_perturbation = np.median(selected_perturbations, axis=0)
                
                # Clippa entro epsilon
                aggregated_perturbation = np.clip(aggregated_perturbation, -self.epsilon, self.epsilon)
                
                # Applica perturbazione
                x_adv[i] = x_sample + aggregated_perturbation
        
        return x_adv


def run_whitebox_decision_tree_attack(
    model_path,
    test_clients=[1, 13],
    epsilons=[0.001, 0.005, 0.01, 0.05],
    save_results=True,
    verbose=False
):
    """
    Esegue attacco White-Box Decision Tree Custom su Random Forest federato.
    
    Args:
        model_path: Path al modello Random Forest federato (.pkl)
        test_clients: Lista di client da usare per test (default: [1, 13])
        epsilons: Lista di epsilon da testare (default: [0.001, 0.005, 0.01, 0.05])
        save_results: Se True, salva risultati in file (default: True)
        verbose: Se True, stampa informazioni dettagliate (default: False)
        
    Returns:
        results: Dictionary con risultati per ogni epsilon
    """
    print("="*80)
    print("🔴 ATTACCO WHITE-BOX: DECISION TREE ATTACK CUSTOM (Random Forest)")
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
        
        # ===== 6.1: Crea attacco Custom =====
        print(f"\n[Attacco] Creazione RandomForestDecisionTreeAttack con epsilon={epsilon}...")
        
        try:
            attack = RandomForestDecisionTreeAttack(
                model=model,
                epsilon=epsilon,
                verbose=(verbose or i == 0)  # Verbose per primo epsilon
            )
            
            print(f"✅ RandomForestDecisionTreeAttack configurato")
            
            # ===== 6.2: Genera esempi adversarial =====
            print(f"\n[Generazione] Adversarial examples con epsilon={epsilon}...")
            
            X_attacks_adv = attack.generate(X_attacks_only)
            print(f"✅ Generati {len(X_attacks_adv)} esempi adversarial")
            
            # ===== 6.3: Applica vincoli fisici SmartGrid =====
            print(f"\n[Vincoli] Applicazione vincoli fisici SmartGrid...")
            
            X_attacks_adv_constrained = apply_physical_constraints(
                X_attacks_adv,
                X_attacks_only,
                constraints,
                max_perturbation_linf=epsilon * 10  # Margine di tolleranza (10x epsilon)
            )
            
            # ===== 6.4: Ricostruisci dataset completo =====
            X_adv_full = X_test.copy()
            X_adv_full[attack_indices] = X_attacks_adv_constrained
            
            # ===== 6.5: Valuta efficacia attacco =====
            print(f"\n[Valutazione] Calcolo metriche per epsilon={epsilon}...")
            
            metrics = evaluate_attack(
                model,
                X_test,
                y_test,
                X_adv_full,
                attack_name=f"WhiteBox_DecisionTree_Custom_eps_{epsilon}"
            )
            
            # ===== 6.6: Stampa report =====
            if verbose or i < 2:  # Verbose per primi 2 epsilon
                print_attack_report(metrics)
            else:
                # Stampa solo ASR per epsilon successivi
                print(f"\n✅ Epsilon {epsilon}: ASR = {metrics['asr']*100:.2f}%, L2 = {metrics['l2_mean']:.6f}")
            
            # ===== 6.7: Salva risultati =====
            results[epsilon] = {
                'X_adv': X_adv_full,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Errore per epsilon={epsilon}: {e}")
            import traceback
            traceback.print_exc()
            # Continua con il prossimo epsilon invece di crashare
            continue
    
    # ========== STEP 7: SALVA RISULTATI FINALI ==========
    if save_results and results:  # 🔧 FIX: Verifica che results non sia vuoto
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
    if results:  # 🔧 FIX: Stampa summary solo se ci sono risultati
        print(f"\n{'='*80}")
        print(f"✅ ATTACCO WHITE-BOX COMPLETATO")
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
        print(f"\n❌ NESSUN RISULTATO DISPONIBILE - Tutti gli epsilon hanno generato errori")
    
    return results


def main():
    """
    Funzione principale per esecuzione da linea di comando.
    """
    parser = argparse.ArgumentParser(
        description="Attacco White-Box Decision Tree Custom per Random Forest federato SmartGrid"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path al modello Random Forest federato (.pkl). Es: models/federated_rf_global_20251121_024044.pkl'
    )
    
    parser.add_argument(
        '--epsilons',
        type=float,
        nargs='+',
        default=[0.001, 0.005, 0.01, 0.05],
        help='Lista di epsilon da testare. Es: --epsilons 0.001 0.01 0.05'
    )
    
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client da usare per test. Es: --test-clients 1 13'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Salva risultati in file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Stampa informazioni dettagliate durante esecuzione'
    )
    
    args = parser.parse_args()
    
    # Esegui attacco
    run_whitebox_decision_tree_attack(
        model_path=args.model_path,
        test_clients=args.test_clients,
        epsilons=args.epsilons,
        save_results=args.save_results or True,  # Default True
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()