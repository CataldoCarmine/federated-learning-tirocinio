#!/usr/bin/env python3
"""
Script per eseguire l'ottimizzazione iperparametri con Optuna.
Uso semplificato dell'optimizer.

UTILIZZO:
    python scripts/run_optuna_optimization.py [numero_trial]
    
ESEMPIO:
    python scripts/run_optuna_optimization.py 100
"""

import sys
import os

# Aggiungi path del progetto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    """Esegue l'ottimizzazione con Optuna."""
    
    # Numero di trial (default 50, personalizzabile)
    n_trials = 50
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            print("❌ Numero trial deve essere un intero")
            print("Utilizzo: python scripts/run_optuna_optimization.py [numero_trial]")
            sys.exit(1)
    
    print(f"🚀 Avvio ottimizzazione con {n_trials} trial")
    
    try:
        # Importa e usa l'optimizer
        from hyperparameter_optimization.optuna_optimizer import SmartGridOptunaOptimizer
        
        # Verifica directory dati
        data_dir = "data/SmartGrid"
        if not os.path.exists(data_dir):
            print(f"❌ Directory {data_dir} non trovata!")
            print("Struttura attesa:")
            print("  data/SmartGrid/")
            print("  ├── data1.csv")
            print("  ├── data2.csv")
            print("  └── ...")
            sys.exit(1)
        
        # Crea e esegui optimizer
        optimizer = SmartGridOptunaOptimizer(data_dir)
        best_params = optimizer.optimize(n_trials)
        
        # Stampa istruzioni per applicazione manuale
        optimizer.print_manual_instructions()
        
        print(f"\n✅ Ottimizzazione completata!")
        print(f"📁 Risultati salvati in: hyperparameter_optimization/results/")
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        print("Verifica che il file hyperparameter_optimization/optuna_optimizer.py esista")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()