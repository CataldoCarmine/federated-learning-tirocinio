#!/usr/bin/env python3
"""
run_whitebox_attack.py

Script esecutivo per l'attacco White-Box Decision Tree Attack sul Random Forest
federato SmartGrid.

Questo script orchestra l'intero processo di:
1. Caricamento del modello Random Forest (federato o centralizzato)
2. Caricamento e preprocessing del test set
3. Esecuzione dell'attacco con multipli epsilon
4. Generazione report e grafici

PREREQUISITI:
- Random Forest addestrato (usa centralizedRF.py o la versione federata)
- ART (Adversarial Robustness Toolbox) installato
- Dataset SmartGrid in data/SmartGrid/

USAGE:
    # Opzione 1: Usa Random Forest già salvato
    python run_whitebox_attack.py --model-path models/federated_rf.pkl
    
    # Opzione 2: Addestra Random Forest on-the-fly
    python run_whitebox_attack.py --train-on-fly
    
    # Opzione 3: Specifica epsilon personalizzati
    python run_whitebox_attack.py --epsilons 0.001 0.01 0.1

Autore: Cataldo Carmine
Progetto: Federated Learning SmartGrid IDS - Adversarial Attacks
"""

import argparse
import os
import sys
import pickle
import numpy as np
from datetime import datetime

# Aggiungi parent directory al path per import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import componenti attacco
from attacks.whitebox_decision_tree_attack import (
    WhiteBoxDecisionTreeAttack,
    load_and_train_random_forest
)
from attacks.utils import (
    set_reproducibility_seeds,
    load_test_data_from_clients,
    apply_preprocessing_pipeline
)

# Import modello centralizzato se disponibile
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'centralized'))
    from centralizedRF import (
        create_smartgrid_random_forest_model,
        load_centralized_smartgrid_data,
        split_train_validation_test,
        centralized_preprocessing
    )
    CENTRALIZED_RF_AVAILABLE = True
except ImportError:
    CENTRALIZED_RF_AVAILABLE = False
    print("⚠️ centralizedRF.py non trovato - usare --train-on-fly o --model-path")


def load_model_from_path(model_path):
    """
    Carica un modello Random Forest salvato da file.
    
    Args:
        model_path: Percorso al file .pkl del modello
        
    Returns:
        RandomForestClassifier: Modello caricato
        
    Raises:
        FileNotFoundError: Se il file non esiste
        ValueError: Se il file non contiene un Random Forest valido
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File modello non trovato: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"CARICAMENTO MODELLO DA FILE")
    print(f"{'='*60}")
    print(f"Percorso: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Verifica che sia un Random Forest
        from sklearn.ensemble import RandomForestClassifier
        if not isinstance(model, RandomForestClassifier):
            raise ValueError(f"Il file non contiene un RandomForestClassifier, trovato: {type(model)}")
        
        # Verifica che sia addestrato
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
            raise ValueError("Il Random Forest nel file non è addestrato!")
        
        print(f"✅ Modello caricato con successo")
        print(f"   - Tipo: RandomForestClassifier")
        print(f"   - Alberi: {len(model.estimators_)}")
        print(f"   - Feature: {model.n_features_in_}")
        print(f"{'='*60}\n")
        
        return model
        
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {e}")
        raise


def train_centralized_rf():
    """
    Addestra un Random Forest centralizzato usando centralizedRF.py.
    
    Returns:
        tuple: (model, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not CENTRALIZED_RF_AVAILABLE:
        raise ImportError(
            "centralizedRF.py non disponibile. "
            "Usa --model-path o --train-on-fly invece."
        )
    
    print(f"\n{'='*60}")
    print(f"ADDESTRAMENTO RANDOM FOREST CENTRALIZZATO")
    print(f"{'='*60}")
    
    # Carica dati
    X, y, dataset_info = load_centralized_smartgrid_data()
    
    # Split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_validation_test(
        X, y,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )
    
    # Preprocessing
    X_train_final, X_val_final, X_test_final = centralized_preprocessing(
        X_train_raw, X_val_raw, X_test_raw
    )
    
    # Crea e addestra modello
    input_features = X_train_final.shape[1]
    model = create_smartgrid_random_forest_model(input_features)
    
    print(f"\nAddestramento in corso...")
    model.fit(X_train_final, y_train)
    
    # Valutazione rapida
    train_acc = model.score(X_train_final, y_train)
    val_acc = model.score(X_val_final, y_val)
    test_acc = model.score(X_test_final, y_test)
    
    print(f"\n✅ Random Forest centralizzato addestrato")
    print(f"   - Accuracy training: {train_acc:.4f}")
    print(f"   - Accuracy validation: {val_acc:.4f}")
    print(f"   - Accuracy test: {test_acc:.4f}")
    print(f"{'='*60}\n")
    
    return model, X_train_final, y_train, X_val_final, y_val, X_test_final, y_test


def main(args):
    """
    Funzione principale per eseguire l'attacco White-Box.
    
    Args:
        args: Argomenti da argparse
    """
    print(f"\n{'#'*80}")
    print(f"# WHITE-BOX DECISION TREE ATTACK - SMARTGRID IDS")
    print(f"{'#'*80}")
    print(f"# Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Configurazione:")
    print(f"#   - Epsilon: {args.epsilons}")
    print(f"#   - Test clients: {args.test_clients}")
    print(f"#   - Save results: {args.save_results}")
    print(f"{'#'*80}\n")
    
    # Imposta seed per riproducibilità
    set_reproducibility_seeds(args.seed)
    
    # ============== STEP 1: CARICAMENTO MODELLO ==============
    
    if args.model_path:
        # Opzione 1: Carica modello da file
        model = load_model_from_path(args.model_path)
        X_test, y_test, _ = load_test_data_from_clients(
            args.test_clients,
            data_dir=args.data_dir
        )
        X_test, _ = apply_preprocessing_pipeline(X_test, fit_on_data=X_test)
        
    elif args.train_on_fly:
        # Opzione 2: Addestra Random Forest on-the-fly
        print(f"Modalità: Addestramento on-the-fly")
        model, _, _, _, _, _ = load_and_train_random_forest(
            train_clients=args.train_clients
        )
        X_test, y_test, _ = load_test_data_from_clients(
            args.test_clients,
            data_dir=args.data_dir
        )
        X_test, _ = apply_preprocessing_pipeline(X_test, fit_on_data=X_test)
        
    elif args.train_centralized:
        # Opzione 3: Addestra Random Forest centralizzato
        print(f"Modalità: Random Forest centralizzato")
        model, _, _, _, _, X_test, y_test = train_centralized_rf()
        
    else:
        print(f"\n❌ ERRORE: Devi specificare una delle seguenti opzioni:")
        print(f"   --model-path PATH    : Carica modello da file")
        print(f"   --train-on-fly       : Addestra modello on-the-fly")
        print(f"   --train-centralized  : Usa centralizedRF.py")
        sys.exit(1)
    
    # ============== STEP 2: VERIFICA DATI ==============
    
    print(f"\n{'='*60}")
    print(f"VERIFICA DATI")
    print(f"{'='*60}")
    print(f"Modello:")
    print(f"  - Tipo: {type(model).__name__}")
    print(f"  - Alberi: {len(model.estimators_)}")
    print(f"  - Feature: {model.n_features_in_}")
    print(f"\nTest Set:")
    print(f"  - Campioni: {len(X_test)}")
    print(f"  - Feature: {X_test.shape[1]}")
    print(f"  - Attacchi: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print(f"  - Naturali: {(y_test==0).sum()} ({(1-y_test.mean())*100:.1f}%)")
    
    # Verifica compatibilità
    if X_test.shape[1] != model.n_features_in_:
        print(f"\n❌ ERRORE: Incompatibilità dimensioni!")
        print(f"   Modello richiede {model.n_features_in_} feature")
        print(f"   Test set ha {X_test.shape[1]} feature")
        sys.exit(1)
    
    print(f"✅ Verifica completata")
    print(f"{'='*60}\n")
    
    # ============== STEP 3: CREAZIONE ATTACCO ==============
    
    print(f"STEP 3: Creazione attacco White-Box...")
    attack = WhiteBoxDecisionTreeAttack(
        model=model,
        X_test=X_test,
        y_test=y_test,
        attack_name="WhiteBox_DecisionTree_SmartGrid"
    )
    
    # ============== STEP 4: ESECUZIONE ATTACCO ==============
    
    print(f"\nSTEP 4: Esecuzione attacco con {len(args.epsilons)} epsilon...")
    results = attack.run(
        epsilons=args.epsilons,
        save_results=args.save_results
    )
    
    # ============== STEP 5: RIEPILOGO FINALE ==============
    
    print(f"\n{'#'*80}")
    print(f"# ATTACCO COMPLETATO CON SUCCESSO")
    print(f"{'#'*80}")
    print(f"# Epsilon testati: {len(results)}")
    print(f"# Risultati salvati: {args.save_results}")
    if args.save_results:
        print(f"# Directory output: results/attacks/")
    print(f"{'#'*80}\n")
    
    # Mostra best epsilon
    if results:
        best_eps = max(
            results.keys(),
            key=lambda e: results[e][1]['asr_attack_to_natural']
        )
        best_asr = results[best_eps][1]['asr_attack_to_natural']
        print(f"🏆 MIGLIOR EPSILON: {best_eps}")
        print(f"   ASR (Attack→Natural): {best_asr:.2%}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="White-Box Decision Tree Attack su Random Forest SmartGrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  # Usa modello salvato
  python run_whitebox_attack.py --model-path models/rf_model.pkl
  
  # Addestra on-the-fly
  python run_whitebox_attack.py --train-on-fly
  
  # Usa centralizedRF.py
  python run_whitebox_attack.py --train-centralized
  
  # Epsilon personalizzati
  python run_whitebox_attack.py --train-on-fly --epsilons 0.001 0.01 0.1
        """
    )
    
    # Opzioni modello
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--model-path',
        type=str,
        help='Percorso al file .pkl del Random Forest addestrato'
    )
    model_group.add_argument(
        '--train-on-fly',
        action='store_true',
        help='Addestra Random Forest on-the-fly per test rapido'
    )
    model_group.add_argument(
        '--train-centralized',
        action='store_true',
        help='Usa centralizedRF.py per addestrare modello'
    )
    
    # Configurazione attacco
    parser.add_argument(
        '--epsilons',
        type=float,
        nargs='+',
        default=[0.001, 0.005, 0.01, 0.05],
        help='Lista di epsilon da testare (default: 0.001 0.005 0.01 0.05)'
    )
    
    # Configurazione dati
    parser.add_argument(
        '--test-clients',
        type=int,
        nargs='+',
        default=[1, 13],
        help='Client IDs per test set (default: 1 13)'
    )
    parser.add_argument(
        '--train-clients',
        type=int,
        nargs='+',
        default=[2,3,4,5,6,7,8,9,10,11,12],
        help='Client IDs per training (solo con --train-on-fly)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/SmartGrid',
        help='Directory dataset SmartGrid (default: data/SmartGrid)'
    )
    
    # Opzioni output
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Salva risultati e grafici (default: True)'
    )
    parser.add_argument(
        '--no-save',
        dest='save_results',
        action='store_false',
        help='Non salvare risultati'
    )
    
    # Riproducibilità
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed per riproducibilità (default: 42)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Attacco interrotto dall'utente")
        sys.exit(0)
    except Exception as e:
        print(f"\n{'#'*80}")
        print(f"# ERRORE FATALE")
        print(f"{'#'*80}")
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)