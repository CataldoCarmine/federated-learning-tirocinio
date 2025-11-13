"""
attacks/whitebox_decision_tree_attack.py

Implementazione dell'attacco White-Box Decision Tree Attack per Random Forest.

Questo attacco sfrutta l'accesso completo alla struttura interna del Random Forest
per generare adversarial examples ottimizzati. Utilizza la libreria ART (Adversarial
Robustness Toolbox) per implementare l'attacco specifico per tree-based models.

SCENARIO:
- Attaccante: Ha accesso completo al modello Random Forest addestrato
- Obiettivo: Far classificare campioni di attacco come traffico naturale
- Metodo: Decision Tree Attack che manipola minimamente le feature per
          attraversare le soglie decisionali degli alberi

CARATTERISTICHE:
- White-box: Richiede conoscenza completa del modello
- Specifico per Random Forest (non funziona su DNN)
- Perturbazioni minime ottimizzate per attraversare decision boundaries
- Supporta test multipli di epsilon per analisi comparativa

Autore: Cataldo Carmine
Progetto: Federated Learning SmartGrid IDS - Adversarial Attacks
"""

import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import ART components
try:
    from art.estimators.classification import ScikitlearnClassifier
    from art.attacks.evasion import DecisionTreeAttack
    ART_AVAILABLE = True
except ImportError:
    print("⚠️ ART (Adversarial Robustness Toolbox) non installato!")
    print("Installa con: pip install adversarial-robustness-toolbox")
    ART_AVAILABLE = False

# Import utility dal modulo attacks
from .utils import (
    set_reproducibility_seeds,
    load_test_data_from_clients,
    apply_preprocessing_pipeline,
    get_smartgrid_physical_constraints,
    apply_physical_constraints
)
from .evaluation import (
    evaluate_attack,
    print_attack_report,
    save_attack_results
)


class WhiteBoxDecisionTreeAttack:
    """
    Attacco White-Box usando Decision Tree Attack di ART.
    
    Questo attacco è specificamente progettato per Random Forest e sfrutta
    la conoscenza della struttura interna degli alberi per generare
    adversarial examples con perturbazioni minime.
    
    COME FUNZIONA:
    1. Accede alle soglie decisionali di ogni albero nel Random Forest
    2. Identifica le feature critiche che influenzano maggiormente la classificazione
    3. Calcola perturbazioni minime necessarie per attraversare le soglie
    4. Genera adversarial examples ottimizzati
    
    PARAMETRI PRINCIPALI:
    - epsilon (offset): Quanto "oltre" la soglia vogliamo spostare il valore
                       (valori più piccoli = perturbazioni minori ma più fragili)
    
    Attributes:
        model: Random Forest target (già addestrato)
        X_test: Dati di test
        y_test: Etichette di test
        attack_name: Nome identificativo dell'attacco
        art_classifier: Wrapper ART del modello per compatibilità
    """
    
    def __init__(self, model, X_test, y_test, attack_name="WhiteBox_DecisionTree"):
        """
        Inizializza l'attacco White-Box Decision Tree.
        
        Args:
            model: Random Forest target (deve essere già addestrato)
            X_test: Dati di test (numpy array)
            y_test: Etichette di test (numpy array)
            attack_name: Nome dell'attacco per logging e salvataggio risultati
            
        Raises:
            ValueError: Se il modello non è addestrato o non è un Random Forest
            ImportError: Se ART non è disponibile
        """
        if not ART_AVAILABLE:
            raise ImportError(
                "ART (Adversarial Robustness Toolbox) è richiesto per questo attacco. "
                "Installa con: pip install adversarial-robustness-toolbox"
            )
        
        # Verifica che il modello sia un Random Forest addestrato
        if not isinstance(model, RandomForestClassifier):
            raise ValueError("Il modello deve essere un RandomForestClassifier")
        
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
            raise ValueError("Il modello Random Forest deve essere addestrato prima dell'attacco!")
        
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.attack_name = attack_name
        
        print(f"\n{'='*60}")
        print(f"INIZIALIZZAZIONE ATTACCO: {attack_name}")
        print(f"{'='*60}")
        print(f"Modello target: Random Forest con {len(model.estimators_)} alberi")
        print(f"Test set: {len(X_test)} campioni, {X_test.shape[1]} feature")
        print(f"Distribuzione test set: {y_test.sum()} attacchi, {(y_test==0).sum()} naturali")
        
        # Calcola range feature per clipping
        self.feature_min = X_test.min(axis=0)
        self.feature_max = X_test.max(axis=0)
        
        # Wrap del modello per ART
        try:
            self.art_classifier = ScikitlearnClassifier(
                model=model,
                clip_values=(self.feature_min, self.feature_max)
            )
            print(f"✅ Modello wrappato per ART con clip_values")
        except Exception as e:
            print(f"❌ Errore nel wrapping del modello per ART: {e}")
            raise
        
        print(f"✅ Attacco White-Box inizializzato correttamente")
        print(f"{'='*60}\n")
    
    def generate_adversarial_examples(self, epsilon=0.001, verbose=True):
        """
        Genera adversarial examples usando Decision Tree Attack.
        
        FUNZIONAMENTO:
        1. Crea l'attacco DecisionTreeAttack di ART
        2. L'attacco analizza la struttura degli alberi nel Random Forest
        3. Identifica le soglie decisionali critiche
        4. Genera perturbazioni minime per attraversare le soglie
        5. Applica vincoli fisici per garantire validità
        
        Args:
            epsilon: Offset per attraversare le soglie decisionali
                    - Valori piccoli (0.001-0.01): perturbazioni minime ma fragili
                    - Valori medi (0.01-0.1): buon compromesso
                    - Valori grandi (>0.1): perturbazioni evidenti ma robuste
            verbose: Se True, stampa informazioni durante l'attacco
            
        Returns:
            X_adv: Adversarial examples generati (numpy array)
            
        Note:
            - L'attacco funziona SOLO su campioni di attacco (y=1)
            - I campioni naturali (y=0) non vengono modificati
            - Le perturbazioni rispettano i vincoli fisici del dominio SmartGrid
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"GENERAZIONE ADVERSARIAL EXAMPLES")
            print(f"{'='*60}")
            print(f"Epsilon (offset): {epsilon}")
        
        # Seleziona SOLO campioni di attacco
        # (non ha senso far "evadere" campioni già classificati come naturali)
        attack_mask = (self.y_test == 1)
        X_attacks_only = self.X_test[attack_mask]
        
        if len(X_attacks_only) == 0:
            print(f"⚠️ Nessun campione di attacco nel test set!")
            return self.X_test.copy()
        
        if verbose:
            print(f"Campioni di attacco da perturbare: {len(X_attacks_only)}")
            print(f"Campioni naturali (non modificati): {(~attack_mask).sum()}")
        
        try:
            if verbose:
                print(f"\n[Decision Tree Attack] Creazione attacco ART...")
            
            # Crea l'attacco Decision Tree
            # offset: quanto "oltre" la soglia vogliamo andare
            attack = DecisionTreeAttack(
                classifier=self.art_classifier,
                offset=epsilon  # Parametro chiave: controlla magnitude perturbazione
            )
            
            if verbose:
                print(f"[Decision Tree Attack] Attacco creato con offset={epsilon}")
                print(f"[Decision Tree Attack] Generazione adversarial examples in corso...")
            
            # Genera adversarial examples SOLO per campioni di attacco
            X_attacks_adv = attack.generate(x=X_attacks_only)
            
            if verbose:
                print(f"[Decision Tree Attack] ✅ Generazione completata")
            
            # Ricostruisci array completo: naturali invariati, attacchi perturbati
            X_adv_full = self.X_test.copy()
            X_adv_full[attack_mask] = X_attacks_adv
            
            # Calcola statistiche perturbazione (solo su campioni attaccati)
            perturbation = X_attacks_adv - X_attacks_only
            l_inf = np.abs(perturbation).max()
            l2 = np.sqrt((perturbation ** 2).sum(axis=1)).mean()
            l0 = (perturbation != 0).sum(axis=1).mean()
            
            if verbose:
                print(f"\n📊 STATISTICHE PERTURBAZIONE (prima dei vincoli):")
                print(f"  - L-infinity (max): {l_inf:.6f}")
                print(f"  - L2 (media): {l2:.6f}")
                print(f"  - L0 (feature modificate medie): {l0:.1f}")
            
            # Applica vincoli fisici SmartGrid
            if verbose:
                print(f"\n[Vincoli Fisici] Applicazione vincoli fisici SmartGrid...")
            
            # Calcola vincoli fisici dal dataset
            constraints = get_smartgrid_physical_constraints(self.X_test)
            
            # Applica vincoli SOLO ai campioni attaccati
            X_attacks_adv_constrained = apply_physical_constraints(
                X_attacks_adv,
                X_attacks_only,
                constraints,
                max_perturbation_linf=epsilon * 10  # Margine per vincoli fisici
            )
            
            # Ricostruisci array finale
            X_adv_final = self.X_test.copy()
            X_adv_final[attack_mask] = X_attacks_adv_constrained
            
            # Statistiche finali
            perturbation_final = X_attacks_adv_constrained - X_attacks_only
            l_inf_final = np.abs(perturbation_final).max()
            l2_final = np.sqrt((perturbation_final ** 2).sum(axis=1)).mean()
            l0_final = (perturbation_final != 0).sum(axis=1).mean()
            
            if verbose:
                print(f"\n📊 STATISTICHE PERTURBAZIONE (dopo vincoli):")
                print(f"  - L-infinity (max): {l_inf_final:.6f}")
                print(f"  - L2 (media): {l2_final:.6f}")
                print(f"  - L0 (feature modificate medie): {l0_final:.1f}")
                print(f"\n✅ Adversarial examples generati e validati")
                print(f"{'='*60}\n")
            
            return X_adv_final
            
        except Exception as e:
            print(f"\n❌ ERRORE durante generazione adversarial examples: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run(self, epsilons=[0.001, 0.005, 0.01, 0.05], save_results=True):
        """
        Esegue l'attacco con multipli valori di epsilon e valuta i risultati.
        
        Questa funzione automatizza il processo completo di:
        1. Generazione adversarial examples per ogni epsilon
        2. Valutazione dell'efficacia dell'attacco
        3. Confronto tra diversi epsilon
        4. Salvataggio risultati e grafici
        
        Args:
            epsilons: Lista di valori epsilon da testare
                     - Default: [0.001, 0.005, 0.01, 0.05]
                     - Suggerimento: inizia con valori piccoli
            save_results: Se True, salva risultati e grafici
            
        Returns:
            dict: Dizionario con {epsilon: (X_adv, metrics)} per ogni epsilon testato
            
        Example:
            >>> attack = WhiteBoxDecisionTreeAttack(model, X_test, y_test)
            >>> results = attack.run(epsilons=[0.001, 0.01, 0.1])
            >>> # Analizza i risultati per trovare l'epsilon ottimale
        """
        print(f"\n{'='*80}")
        print(f"🚀 ESECUZIONE ATTACCO WHITE-BOX: {self.attack_name}")
        print(f"{'='*80}")
        print(f"Epsilon da testare: {epsilons}")
        print(f"Campioni test: {len(self.X_test)}")
        print(f"Modello target: Random Forest con {len(self.model.estimators_)} alberi")
        print(f"{'='*80}\n")
        
        results = {}
        all_metrics = []
        all_X_adv = {}
        
        for i, epsilon in enumerate(epsilons):
            print(f"\n{'*'*60}")
            print(f"TEST {i+1}/{len(epsilons)}: EPSILON = {epsilon}")
            print(f"{'*'*60}")
            
            try:
                # Genera adversarial examples
                X_adv = self.generate_adversarial_examples(epsilon=epsilon, verbose=True)
                
                # Valuta efficacia
                print(f"\n[Valutazione] Calcolo metriche per epsilon={epsilon}...")
                metrics = evaluate_attack(
                    self.model,
                    self.X_test,
                    self.y_test,
                    X_adv,
                    attack_name=f"{self.attack_name}_eps_{epsilon}"
                )
                
                # Stampa report
                print_attack_report(metrics)
                
                # Salva per confronto
                results[epsilon] = (X_adv, metrics)
                all_metrics.append(metrics)
                all_X_adv[epsilon] = X_adv
                
            except Exception as e:
                print(f"\n❌ ERRORE con epsilon={epsilon}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Stampa confronto finale
        if len(all_metrics) > 1:
            self._print_comparison(all_metrics, epsilons)
        
        # Salva risultati
        if save_results and all_metrics:
            print(f"\n{'='*60}")
            print(f"SALVATAGGIO RISULTATI")
            print(f"{'='*60}")
            
            save_attack_results(
                all_metrics,
                self.X_test,
                all_X_adv,
                epsilons_tested=epsilons,
                save_dir="results/attacks"
            )
        
        print(f"\n{'='*80}")
        print(f"✅ ATTACCO COMPLETATO")
        print(f"{'='*80}")
        print(f"Epsilon testati: {len(results)}/{len(epsilons)}")
        print(f"{'='*80}\n")
        
        return results
    
    def _print_comparison(self, all_metrics, epsilons):
        """
        Stampa tabella comparativa per diversi epsilon.
        
        Args:
            all_metrics: Lista di dizionari metriche
            epsilons: Lista epsilon testati
        """
        print(f"\n{'='*80}")
        print(f"CONFRONTO EPSILON")
        print(f"{'='*80}\n")
        
        # Header
        print(f"{'Epsilon':<12} {'ASR':<10} {'ASR_A→N':<12} {'L-inf':<12} {'L2':<12} {'Acc_After':<12}")
        print(f"{'-'*80}")
        
        # Righe
        for i, (metrics, eps) in enumerate(zip(all_metrics, epsilons)):
            print(f"{eps:<12.6f} "
                  f"{metrics['asr']:<10.4f} "
                  f"{metrics['asr_attack_to_natural']:<12.4f} "
                  f"{metrics['l_inf']:<12.6f} "
                  f"{metrics['l2']:<12.6f} "
                  f"{metrics['accuracy_after']:<12.4f}")
        
        print(f"{'='*80}\n")
        
        # Identifica epsilon ottimale (massimo ASR attack→natural con minima perturbazione)
        best_idx = max(
            range(len(all_metrics)),
            key=lambda i: (all_metrics[i]['asr_attack_to_natural'], -all_metrics[i]['l2'])
        )
        best_eps = epsilons[best_idx]
        best_asr = all_metrics[best_idx]['asr_attack_to_natural']
        
        print(f"🏆 EPSILON OTTIMALE: {best_eps}")
        print(f"   - ASR (Attack→Natural): {best_asr:.2%}")
        print(f"   - L2: {all_metrics[best_idx]['l2']:.6f}")
        print(f"   - L-inf: {all_metrics[best_idx]['l_inf']:.6f}")
        print(f"{'='*80}\n")


# ============== FUNZIONI HELPER PER USO STANDALONE ==============

def load_and_train_random_forest(train_clients=[2,3,4,5,6,7,8,9,10,11,12], 
                                  preprocessing_config=None,
                                  rf_config=None):
    """
    Carica dati e addestra Random Forest per l'attacco.
    
    Questa funzione è un helper per addestrare rapidamente un Random Forest
    su cui testare l'attacco, usando la configurazione del progetto.
    
    Args:
        train_clients: Lista ID client per training
        preprocessing_config: Dict con configurazione preprocessing
        rf_config: Dict con configurazione Random Forest
        
    Returns:
        tuple: (model, X_train, y_train, X_val, y_val, preprocessing_objects)
    """
    from .utils import apply_preprocessing_pipeline, load_test_data_from_clients
    
    print(f"\n{'='*60}")
    print(f"CARICAMENTO E TRAINING RANDOM FOREST")
    print(f"{'='*60}")
    
    # Carica dati training
    print(f"Caricamento dati da client {train_clients}...")
    X_train, y_train, train_info = load_test_data_from_clients(
        train_clients, 
        data_dir="data/SmartGrid"
    )
    
    # Split train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    print(f"Training set: {len(X_train_split)} campioni")
    print(f"Validation set: {len(X_val)} campioni")
    
    # Preprocessing (identico a clientRF.py)
    print(f"\nApplicazione preprocessing...")
    X_train_preprocessed, preprocessing_objects = apply_preprocessing_pipeline(
        X_train_split,
        fit_on_data=X_train_split  # Fit su training
    )
    X_val_preprocessed, _ = apply_preprocessing_pipeline(
        X_val,
        fit_on_data=None  # Usa oggetti già fittati
    )
    
    print(f"Dati preprocessati: {X_train_preprocessed.shape}")
    
    # Configurazione Random Forest (default: come in clientRF.py)
    if rf_config is None:
        rf_config = {
            'n_estimators': 65,
            'criterion': 'entropy',
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Addestramento
    print(f"\nAddestramento Random Forest...")
    print(f"Configurazione: {rf_config}")
    
    model = RandomForestClassifier(**rf_config)
    model.fit(X_train_preprocessed, y_train_split)
    
    # Valutazione rapida
    train_acc = model.score(X_train_preprocessed, y_train_split)
    val_acc = model.score(X_val_preprocessed, y_val)
    
    print(f"\n✅ Random Forest addestrato")
    print(f"   - Accuracy training: {train_acc:.4f}")
    print(f"   - Accuracy validation: {val_acc:.4f}")
    print(f"   - Alberi: {len(model.estimators_)}")
    print(f"{'='*60}\n")
    
    return model, X_train_preprocessed, y_train_split, X_val_preprocessed, y_val, preprocessing_objects


# ============== MAIN PER TESTING STANDALONE ==============

if __name__ == "__main__":
    """
    Script di test standalone per l'attacco White-Box Decision Tree.
    
    Questo script può essere eseguito direttamente per testare l'attacco
    senza dover integrare tutto il framework federato.
    
    Usage:
        python -m attacks.whitebox_decision_tree_attack
    """
    print(f"\n{'#'*80}")
    print(f"# TEST STANDALONE: WHITE-BOX DECISION TREE ATTACK")
    print(f"{'#'*80}\n")
    
    # Imposta semi per riproducibilità
    set_reproducibility_seeds(42)
    
    try:
        # 1. Carica dati test (client 1 e 13 come da configurazione)
        print(f"STEP 1: Caricamento test set...")
        X_test, y_test, test_info = load_test_data_from_clients(
            client_ids=[1, 13],
            data_dir="data/SmartGrid"
        )
        
        # Preprocessing test set
        print(f"\nSTEP 2: Preprocessing test set...")
        X_test_preprocessed, _ = apply_preprocessing_pipeline(
            X_test,
            fit_on_data=X_test  # Fit su test per semplicità in standalone
        )
        
        # 2. Addestra Random Forest (o carica se disponibile)
        print(f"\nSTEP 3: Training Random Forest...")
        model, X_train, y_train, X_val, y_val, _ = load_and_train_random_forest(
            train_clients=[2,3,4,5,6,7,8,9,10,11,12]
        )
        
        # 3. Crea attacco
        print(f"\nSTEP 4: Creazione attacco White-Box...")
        attack = WhiteBoxDecisionTreeAttack(
            model=model,
            X_test=X_test_preprocessed,
            y_test=y_test,
            attack_name="WhiteBox_DecisionTree_Standalone"
        )
        
        # 4. Esegui attacco con multipli epsilon
        print(f"\nSTEP 5: Esecuzione attacco...")
        results = attack.run(
            epsilons=[0.001, 0.005, 0.01, 0.05],
            save_results=True
        )
        
        print(f"\n{'#'*80}")
        print(f"# TEST COMPLETATO")
        print(f"{'#'*80}\n")
        print(f"Risultati disponibili in: results/attacks/")
        print(f"Epsilon testati: {list(results.keys())}")
        
    except Exception as e:
        print(f"\n{'#'*80}")
        print(f"# ERRORE DURANTE IL TEST")
        print(f"{'#'*80}\n")
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)