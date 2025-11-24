"""
attacks/1_whitebox_decision_tree_attack.py

Attacco White-Box Decision Tree MIGLIORATO per Random Forest federato SmartGrid.

MIGLIORAMENTI IMPLEMENTATI:
1. Aumento trial: 20 → 200 (esplorazione più profonda)
2. Zeroth-Order Optimization (ZOO) per gradiente approssimato
3. Early stopping intelligente (converge quando trova soluzione)
4. Perturbazione SELETTIVA (solo feature importanti)
5. Direzione guidata da analisi distribuzione classi
6. Adaptive epsilon per singole feature

STRATEGIA MIGLIORATA:
- Usa analisi dati per identificare feature critiche
- Perturba SOLO feature importanti (non tutte le 128)
- Usa gradiente approssimato (ZOO) per direzione ottimale
- Early stopping quando trova perturbazione valida
- Adaptive sampling basato su successi precedenti

RIFERIMENTI:
- Paper originale: Papernot et al. (2016) - Decision Tree Attack
- ZOO: Chen et al. (2017) - Zeroth Order Optimization

UTILIZZO:
    python attacks/1_whitebox_decision_tree_attack.py \
        --model-path models/federated_rf_global_20251121_024044.pkl \
        --epsilons 0.01 0.05 \
        --enable-zoo \
        --enable-analysis \
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
from attacks.attacks_beta.data_analysis import (
    analyze_feature_importance,
    analyze_class_distributions,
    find_critical_features_for_evasion,
    run_complete_data_analysis
)


class ImprovedRandomForestDecisionTreeAttack:
    """
    Attacco White-Box MIGLIORATO per Random Forest.
    
    MIGLIORAMENTI RISPETTO ALLA VERSIONE BASE:
    
    1. TRIAL AUMENTATI: 20 → 200
       - Maggiore probabilità di trovare perturbazioni efficaci
       - Esplorazione più completa dello spazio
    
    2. ZEROTH-ORDER OPTIMIZATION (ZOO):
       - Approssima gradiente senza derivate
       - Direzione intelligente invece di casuale
       - Formula: grad ≈ (f(x+δ) - f(x-δ)) / (2δ)
    
    3. EARLY STOPPING:
       - Ferma quando trova perturbazione che funziona
       - Risparmia computazione inutile
       - Converge più velocemente
    
    4. PERTURBAZIONE SELETTIVA:
       - Perturba SOLO feature importanti/critiche
       - Ignora feature irrilevanti
       - Riduce spazio ricerca da 128D a ~20D
    
    5. DIREZIONE GUIDATA:
       - Usa analisi distribuzione per direzione iniziale
       - Attack → Natural: sposta verso media Natural
       - Aumenta probabilità successo
    
    6. ADAPTIVE EPSILON:
       - Epsilon diverso per feature diverse
       - Feature importanti: epsilon più grande
       - Feature irrilevanti: epsilon piccolo
    """
    
    def __init__(self, model, epsilon=0.01, enable_zoo=True, enable_analysis=True, 
                 n_trials=200, verbose=False):
        """
        Inizializza l'attacco migliorato per Random Forest.
        
        Args:
            model: RandomForestClassifier da attaccare
            epsilon: Budget massimo perturbazione L-infinity
            enable_zoo: Se True, usa Zeroth-Order Optimization
            enable_analysis: Se True, usa analisi dati per guidare perturbazioni
            n_trials: Numero di trial Monte Carlo (default: 200)
            verbose: Se True, stampa informazioni dettagliate
        """
        self.model = model
        self.epsilon = epsilon
        self.enable_zoo = enable_zoo
        self.enable_analysis = enable_analysis
        self.n_trials = n_trials
        self.verbose = verbose
        
        # Verifica che sia un Random Forest
        if not hasattr(model, 'estimators_'):
            raise ValueError("Il modello deve essere un RandomForestClassifier con alberi addestrati")
        
        self.n_trees = len(model.estimators_)
        self.n_features = model.n_features_in_
        
        # Inizializza analisi dati (se abilitata)
        self.important_features = None
        self.feature_directions = None
        self.adaptive_epsilon = None
        
        if self.verbose:
            print(f"[ImprovedRFAttack] Inizializzato con {self.n_trees} alberi, {self.n_features} feature")
            print(f"[ImprovedRFAttack] ZOO: {'ENABLED' if enable_zoo else 'DISABLED'}")
            print(f"[ImprovedRFAttack] Data Analysis: {'ENABLED' if enable_analysis else 'DISABLED'}")
            print(f"[ImprovedRFAttack] N. trials: {n_trials}")
    
    def setup_guided_attack(self, X_sample_batch, y_sample_batch):
        """
        Configura attacco guidato da analisi dati.
        
        Analizza batch di campioni per identificare:
        - Feature più importanti da perturbare
        - Direzione ottimale per ogni feature
        - Epsilon adaptive per ogni feature
        
        Args:
            X_sample_batch: Batch campioni per analisi (shape: [N, features])
            y_sample_batch: Etichette corrispondenti
        """
        if not self.enable_analysis:
            return
        
        print(f"\n[ImprovedRFAttack] === CONFIGURAZIONE ATTACCO GUIDATO ===")
        
        # Analisi feature importance
        importance_info = analyze_feature_importance(
            self.model, X_sample_batch, y_sample_batch, 
            top_n=20, save_report=False
        )
        
        # Seleziona top 20 feature importanti
        self.important_features = importance_info['top_features_indices'][:20]
        
        print(f"[ImprovedRFAttack] Feature selezionate: {len(self.important_features)} su {self.n_features}")
        print(f"[ImprovedRFAttack] Top 5 feature: {self.important_features[:5]}")
        
        # Analisi distribuzione per direzione
        X_attack = X_sample_batch[y_sample_batch == 1]
        X_natural = X_sample_batch[y_sample_batch == 0]
        
        if len(X_natural) > 0:
            # Calcola direzione per ogni feature importante
            self.feature_directions = {}
            
            for feat_idx in self.important_features:
                mean_attack = X_attack[:, feat_idx].mean()
                mean_natural = X_natural[:, feat_idx].mean()
                
                # Direzione: verso media Natural
                direction = +1 if mean_natural > mean_attack else -1
                self.feature_directions[feat_idx] = direction
            
            print(f"[ImprovedRFAttack] Direzioni calcolate per {len(self.feature_directions)} feature")
        
        # Adaptive epsilon (proporzionale a feature importance)
        self.adaptive_epsilon = np.ones(self.n_features) * (self.epsilon * 0.1)  # Default basso
        
        for i, feat_idx in enumerate(self.important_features):
            # Feature più importanti → epsilon più grande
            importance_weight = 1.0 - (i / len(self.important_features))  # 1.0 → 0.0
            self.adaptive_epsilon[feat_idx] = self.epsilon * (0.5 + 0.5 * importance_weight)
        
        print(f"[ImprovedRFAttack] Adaptive epsilon configurato")
        print(f"[ImprovedRFAttack] Range epsilon: [{self.adaptive_epsilon.min():.4f}, {self.adaptive_epsilon.max():.4f}]")
        print(f"[ImprovedRFAttack] === CONFIGURAZIONE COMPLETATA ===\n")
    
    def _compute_zoo_gradient(self, tree, x_sample, target_class=0, delta=0.001):
        """
        Calcola gradiente approssimato usando Zeroth-Order Optimization (ZOO).
        
        SPIEGAZIONE ZOO:
        
        Problema: Random Forest non ha gradiente (non differenziabile)
        Soluzione: Approssimiamo gradiente con differenze finite
        
        Formula:
        grad[i] ≈ (f(x + δ*e_i) - f(x - δ*e_i)) / (2δ)
        
        dove:
        - e_i = vettore con 1 in posizione i, 0 altrove
        - f(x) = probabilità predizione classe target
        - δ = small perturbation (es. 0.001)
        
        Interpretazione:
        - grad[i] > 0: aumentare feature i favorisce target class
        - grad[i] < 0: diminuire feature i favorisce target class
        
        Args:
            tree: DecisionTreeClassifier singolo
            x_sample: Campione originale
            target_class: Classe target (0 = Natural)
            delta: Dimensione perturbazione per diff finite
            
        Returns:
            gradient: Gradiente approssimato (numpy array)
        """
        gradient = np.zeros(self.n_features)
        
        # Predizione originale
        try:
            # Usa predict_proba se disponibile (più smooth)
            if hasattr(tree, 'predict_proba'):
                prob_orig = tree.predict_proba([x_sample])[0][target_class]
            else:
                # Fallback: usa predict binario
                prob_orig = 1.0 if tree.predict([x_sample])[0] == target_class else 0.0
        except:
            prob_orig = 0.5  # Default neutrale
        
        # Calcola gradiente per ogni feature
        # Usa solo feature importanti se disponibili
        features_to_test = self.important_features if self.important_features is not None else range(self.n_features)
        
        for feat_idx in features_to_test:
            # Perturbazione positiva
            x_plus = x_sample.copy()
            x_plus[feat_idx] += delta
            
            try:
                if hasattr(tree, 'predict_proba'):
                    prob_plus = tree.predict_proba([x_plus])[0][target_class]
                else:
                    prob_plus = 1.0 if tree.predict([x_plus])[0] == target_class else 0.0
            except:
                prob_plus = prob_orig
            
            # Perturbazione negativa
            x_minus = x_sample.copy()
            x_minus[feat_idx] -= delta
            
            try:
                if hasattr(tree, 'predict_proba'):
                    prob_minus = tree.predict_proba([x_minus])[0][target_class]
                else:
                    prob_minus = 1.0 if tree.predict([x_minus])[0] == target_class else 0.0
            except:
                prob_minus = prob_orig
            
            # Gradiente approssimato
            gradient[feat_idx] = (prob_plus - prob_minus) / (2 * delta)
        
        return gradient
    
    def _compute_minimal_perturbation_for_tree(self, tree, x_sample, target_class=0):
        """
        Calcola perturbazione minima MIGLIORATA per singolo albero.
        
        MIGLIORAMENTI:
        1. Più trial (200 invece di 20)
        2. ZOO per direzione iniziale intelligente
        3. Early stopping quando trova soluzione
        4. Perturbazione SELETTIVA (solo feature importanti)
        5. Adaptive epsilon per feature diverse
        
        Args:
            tree: DecisionTreeClassifier
            x_sample: Campione originale
            target_class: Classe target (0 = Natural)
            
        Returns:
            Perturbation vector (numpy array) o None se fallisce
        """
        # Predizione corrente
        current_pred = tree.predict([x_sample])[0]
        
        # Se già classe target, zero perturbazione
        if current_pred == target_class:
            return np.zeros(self.n_features)
        
        # Usa ZOO per direzione iniziale (se abilitato)
        if self.enable_zoo:
            gradient = self._compute_zoo_gradient(tree, x_sample, target_class)
        else:
            gradient = None
        
        best_perturbation = None
        best_norm = float('inf')
        
        # MONTE CARLO MIGLIORATO con early stopping
        for trial in range(self.n_trials):
            # Genera perturbazione
            if self.enable_zoo and gradient is not None:
                # ZOO-guided: usa gradiente come direzione base
                if trial < self.n_trials // 2:
                    # Prima metà: usa gradiente + rumore
                    perturbation = gradient * self.epsilon * 0.5
                    perturbation += np.random.randn(self.n_features) * self.epsilon * 0.3
                else:
                    # Seconda metà: esplorazione casuale
                    perturbation = np.random.randn(self.n_features) * self.epsilon * 0.5
            else:
                # Random Monte Carlo standard
                perturbation = np.random.randn(self.n_features) * self.epsilon * 0.5
            
            # Usa adaptive epsilon se disponibile
            if self.adaptive_epsilon is not None:
                perturbation = np.clip(perturbation, -self.adaptive_epsilon, self.adaptive_epsilon)
            else:
                perturbation = np.clip(perturbation, -self.epsilon, self.epsilon)
            
            # Applica direzioni guidate (se disponibili)
            if self.feature_directions is not None:
                for feat_idx, direction in self.feature_directions.items():
                    # Forza direzione ottimale per feature critiche
                    if perturbation[feat_idx] * direction < 0:  # Direzione opposta
                        perturbation[feat_idx] *= -1  # Inverte direzione
            
            # Perturbazione SELETTIVA: azzera feature non importanti
            if self.important_features is not None:
                mask = np.zeros(self.n_features, dtype=bool)
                mask[self.important_features] = True
                perturbation[~mask] = 0  # Azzera feature non selezionate
            
            # Applica perturbazione
            x_perturbed = x_sample + perturbation
            
            # Verifica se cambia predizione
            new_pred = tree.predict([x_perturbed])[0]
            
            if new_pred == target_class:
                # Successo! Calcola norma
                norm = np.linalg.norm(perturbation)
                
                if norm < best_norm:
                    best_norm = norm
                    best_perturbation = perturbation
                    
                    # EARLY STOPPING: Se trova perturbazione molto piccola, ferma
                    if norm < self.epsilon * 0.1:
                        break
        
        return best_perturbation
    
    def generate(self, x):
        """
        Genera esempi adversarial MIGLIORATI.
        
        MIGLIORAMENTI:
        - Setup attacco guidato su batch
        - Perturbazioni più efficaci
        - Early stopping globale
        
        Args:
            x: Batch di campioni (N, n_features)
            
        Returns:
            x_adv: Esempi adversarial (N, n_features)
        """
        x_adv = x.copy()
        
        # Setup attacco guidato (una volta per batch)
        if self.enable_analysis and self.important_features is None:
            # Usa primi 1000 campioni per analisi rapida
            sample_size = min(1000, len(x))
            X_analysis = x[:sample_size]
            y_analysis = self.model.predict(X_analysis)  # Usa predizioni come proxy
            
            self.setup_guided_attack(X_analysis, y_analysis)
        
        # Processa ogni campione
        successful_evasions = 0
        
        for i, x_sample in enumerate(x):
            if self.verbose and i % 100 == 0:
                print(f"[ImprovedRFAttack] Campione {i+1}/{len(x)}, Successi: {successful_evasions}")
            
            # Predizione corrente
            current_pred = self.model.predict([x_sample])[0]
            
            # Se già classe target, skip
            if current_pred == 0:  # 0 = Natural
                continue
            
            # Raccogli voti alberi
            tree_votes = np.array([tree.predict([x_sample])[0] for tree in self.model.estimators_])
            attack_votes = int(np.sum(tree_votes == 1))
            
            # Calcola alberi da flippare
            majority_threshold = self.n_trees // 2 + 1
            current_natural_votes = self.n_trees - attack_votes
            trees_to_flip = max(0, majority_threshold - current_natural_votes)
            trees_to_flip = int(trees_to_flip)
            
            if trees_to_flip <= 0:
                continue
            
            # Trova perturbazioni per alberi Attack
            tree_perturbations = []
            
            for tree_idx in range(self.n_trees):
                if tree_votes[tree_idx] == 1:  # Albero vota Attack
                    pert = self._compute_minimal_perturbation_for_tree(
                        self.model.estimators_[tree_idx], 
                        x_sample, 
                        target_class=0
                    )
                    
                    if pert is not None:
                        norm = np.linalg.norm(pert)
                        tree_perturbations.append((norm, pert))
            
            if not tree_perturbations:
                continue
            
            # Ordina e seleziona
            tree_perturbations.sort(key=lambda x: x[0])
            
            if len(tree_perturbations) >= trees_to_flip:
                selected_perturbations = [pert for _, pert in tree_perturbations[:trees_to_flip]]
            else:
                selected_perturbations = [pert for _, pert in tree_perturbations]
            
            if selected_perturbations:
                # Aggrega con mediana
                aggregated_perturbation = np.median(selected_perturbations, axis=0)
                aggregated_perturbation = np.clip(aggregated_perturbation, -self.epsilon, self.epsilon)
                
                # Applica
                x_adv[i] = x_sample + aggregated_perturbation
                
                # Verifica successo
                new_pred = self.model.predict([x_adv[i]])[0]
                if new_pred == 0:
                    successful_evasions += 1
        
        print(f"\n[ImprovedRFAttack] Generazione completata: {successful_evasions}/{len(x)} evasioni riuscite")
        
        return x_adv


def run_whitebox_decision_tree_attack(
    model_path,
    test_clients=[1, 13],
    epsilons=[0.001, 0.005, 0.01, 0.05],
    enable_zoo=True,
    enable_analysis=True,
    n_trials=200,
    save_results=True,
    verbose=False
):
    """
    Esegue attacco White-Box MIGLIORATO su Random Forest federato.
    
    Args:
        model_path: Path al modello Random Forest federato (.pkl)
        test_clients: Lista di client da usare per test
        epsilons: Lista di epsilon da testare
        enable_zoo: Se True, abilita Zeroth-Order Optimization
        enable_analysis: Se True, abilita analisi dati guidata
        n_trials: Numero di trial Monte Carlo
        save_results: Se True, salva risultati in file
        verbose: Se True, stampa informazioni dettagliate
        
    Returns:
        results: Dictionary con risultati per ogni epsilon
    """
    print("="*80)
    print("🔴 ATTACCO WHITE-BOX MIGLIORATO: DECISION TREE ATTACK (Monte Carlo + ZOO)")
    print("="*80)
    print(f"Modello: {model_path}")
    print(f"Test clients: {test_clients}")
    print(f"Epsilon da testare: {epsilons}")
    print(f"ZOO: {'ENABLED' if enable_zoo else 'DISABLED'}")
    print(f"Data Analysis: {'ENABLED' if enable_analysis else 'DISABLED'}")
    print(f"Monte Carlo trials: {n_trials}")
    print("="*80 + "\n")
    
    # Imposta seed
    set_reproducibility_seeds(42)
    
    # ========== STEP 1: CARICA MODELLO ==========
    print("\n[STEP 1] Caricamento modello Random Forest federato...")
    model = load_federated_model(model_path)
    
    # ========== STEP 2: CARICA DATI TEST ==========
    print("\n[STEP 2] Caricamento dati di test...")
    X_test_raw, y_test, test_info = load_test_data_from_clients(client_ids=test_clients)
    
    # ========== STEP 3: PREPROCESSING ==========
    print("\n[STEP 3] Applicazione preprocessing...")
    X_test, _ = apply_preprocessing_pipeline(X_test_raw, fit_on_data=X_test_raw)
    
    if X_test.shape[1] != model.n_features_in_:
        raise ValueError(f"❌ Incompatibilità feature: test={X_test.shape[1]}, modello={model.n_features_in_}")
    
    print(f"✅ Preprocessing completato: {X_test.shape}")
    
    # ========== STEP 4: VINCOLI FISICI ==========
    constraints = get_smartgrid_physical_constraints(X_test)
    
    # ========== STEP 5: SELEZIONA CAMPIONI ATTACCO ==========
    print("\n[STEP 4] Selezione campioni di attacco...")
    X_attacks_only, y_attacks_only, attack_indices = select_attack_samples(X_test, y_test, target_class=1)
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni di attacco: {len(X_attacks_only)}")
    print(f"  - Campioni naturali: {(y_test == 0).sum()}")
    
    # ========== STEP 6: ESEGUI ATTACCO PER OGNI EPSILON ==========
    results = {}
    
    for i, epsilon in enumerate(epsilons):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}/{len(epsilons)}: EPSILON = {epsilon}")
        print(f"{'='*80}")
        
        try:
            # Crea attacco migliorato
            attack = ImprovedRandomForestDecisionTreeAttack(
                model=model,
                epsilon=epsilon,
                enable_zoo=enable_zoo,
                enable_analysis=enable_analysis,
                n_trials=n_trials,
                verbose=(verbose or i == 0)
            )
            
            # Genera adversarial examples
            print(f"\n[Generazione] Adversarial examples con epsilon={epsilon}...")
            X_attacks_adv = attack.generate(X_attacks_only)
            print(f"✅ Generati {len(X_attacks_adv)} esempi adversarial")
            
            # Applica vincoli fisici
            print(f"\n[Vincoli] Applicazione vincoli fisici SmartGrid...")
            X_attacks_adv_constrained = apply_physical_constraints(
                X_attacks_adv, X_attacks_only, constraints, max_perturbation_linf=epsilon * 10
            )
            
            # Ricostruisci dataset completo
            X_adv_full = X_test.copy()
            X_adv_full[attack_indices] = X_attacks_adv_constrained
            
            # Valuta efficacia
            print(f"\n[Valutazione] Calcolo metriche per epsilon={epsilon}...")
            metrics = evaluate_attack(
                model, X_test, y_test, X_adv_full,
                attack_name=f"WhiteBox_Improved_DT_eps_{epsilon}"
            )
            
            # Stampa report
            if verbose or i < 2:
                print_attack_report(metrics)
            else:
                print(f"\n✅ Epsilon {epsilon}: ASR = {metrics['asr']*100:.2f}%, L2 = {metrics['l2_mean']:.6f}")
            
            # Salva risultati
            results[epsilon] = {'X_adv': X_adv_full, 'metrics': metrics}
            
        except Exception as e:
            print(f"❌ Errore per epsilon={epsilon}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== STEP 7: SALVA RISULTATI ==========
    if save_results and results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        metrics_list = [results[eps]['metrics'] for eps in epsilons if eps in results]
        X_adv_dict = {eps: results[eps]['X_adv'] for eps in epsilons if eps in results}
        
        save_attack_results(
            metrics_list, X_test, X_adv_dict,
            epsilons_tested=[eps for eps in epsilons if eps in results],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== STEP 8: SUMMARY ==========
    if results:
        print(f"\n{'='*80}")
        print(f"✅ ATTACCO WHITE-BOX MIGLIORATO COMPLETATO")
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
    """Funzione principale per esecuzione da linea di comando."""
    parser = argparse.ArgumentParser(
        description="Attacco White-Box MIGLIORATO per Random Forest federato SmartGrid"
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path al modello Random Forest federato (.pkl)')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05],
                       help='Lista di epsilon da testare')
    parser.add_argument('--test-clients', type=int, nargs='+', default=[1, 13],
                       help='Client da usare per test')
    parser.add_argument('--enable-zoo', action='store_true',
                       help='Abilita Zeroth-Order Optimization')
    parser.add_argument('--enable-analysis', action='store_true',
                       help='Abilita analisi dati guidata')
    parser.add_argument('--n-trials', type=int, default=200,
                       help='Numero trial Monte Carlo (default: 200)')
    parser.add_argument('--save-results', action='store_true',
                       help='Salva risultati in file')
    parser.add_argument('--verbose', action='store_true',
                       help='Stampa informazioni dettagliate')
    
    args = parser.parse_args()
    
    run_whitebox_decision_tree_attack(
        model_path=args.model_path,
        test_clients=args.test_clients,
        epsilons=args.epsilons,
        enable_zoo=args.enable_zoo,
        enable_analysis=args.enable_analysis,
        n_trials=args.n_trials,
        save_results=args.save_results or True,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()