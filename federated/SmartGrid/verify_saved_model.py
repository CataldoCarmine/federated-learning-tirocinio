"""
verify_saved_model.py

Script per verificare che il modello Random Forest federato salvato sia caricabile e funzionante.

Uso:
    python3 verify_saved_model.py models/federated_rf_global_YYYYMMDD_HHMMSS.pkl
"""

import sys
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def verify_model(model_path):
    """
    Verifica che un modello Random Forest federato salvato sia valido e funzionante.
    
    Args:
        model_path: Path al file .pkl del modello
        
    Returns:
        True se il modello è valido, False altrimenti
    """
    print(f"{'='*80}")
    print(f"VERIFICA MODELLO RANDOM FOREST FEDERATO")
    print(f"{'='*80}")
    print(f"Path modello: {model_path}\n")
    
    # STEP 1: Verifica esistenza file
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: File non trovato: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"✅ File trovato")
    print(f"   Dimensione: {file_size / (1024*1024):.2f} MB ({file_size:,} bytes)\n")
    
    # STEP 2: Caricamento modello
    try:
        print(f"🔄 Caricamento modello con joblib...")
        model = joblib.load(model_path)
        print(f"✅ Modello caricato con successo\n")
    except Exception as e:
        print(f"❌ ERRORE nel caricamento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # STEP 3: Verifica tipo modello
    print(f"📋 VERIFICA TIPO:")
    print(f"   Tipo Python: {type(model)}")
    print(f"   Classe: {model.__class__.__name__}")
    
    if not isinstance(model, RandomForestClassifier):
        print(f"❌ ERRORE: Il modello non è un RandomForestClassifier!")
        print(f"   Tipo ricevuto: {type(model)}")
        return False
    
    print(f"✅ Tipo corretto: RandomForestClassifier\n")
    
    # STEP 4: Verifica attributi obbligatori
    print(f"🔍 VERIFICA ATTRIBUTI OBBLIGATORI:")
    
    required_attrs = {
        'estimators_': 'Lista alberi',
        'n_features_in_': 'Numero feature',
        'classes_': 'Classi',
        'n_estimators': 'Numero estimatori configurato',
        'criterion': 'Criterio splitting'
    }
    
    missing_attrs = []
    for attr, description in required_attrs.items():
        has_attr = hasattr(model, attr)
        status = "✅" if has_attr else "❌"
        print(f"   {status} {description} ({attr})")
        if not has_attr:
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"\n❌ ERRORE: Attributi mancanti: {missing_attrs}")
        return False
    
    print(f"\n✅ Tutti gli attributi obbligatori presenti\n")
    
    # STEP 5: Verifica dettagli modello
    print(f"📊 DETTAGLI MODELLO:")
    print(f"   - Numero alberi (estimators_): {len(model.estimators_)}")
    print(f"   - Numero alberi (config): {model.n_estimators}")
    print(f"   - Numero feature: {model.n_features_in_}")
    print(f"   - Classi: {model.classes_}")
    print(f"   - N. classi: {model.n_classes_ if hasattr(model, 'n_classes_') else 'N/A'}")
    print(f"   - Criterio: {model.criterion}")
    print(f"   - Max features: {model.max_features}")
    print(f"   - Bootstrap: {model.bootstrap}")
    
    # Verifica coerenza
    if len(model.estimators_) != model.n_estimators:
        print(f"\n⚠️ WARNING: Numero alberi non coerente!")
        print(f"   estimators_ ha {len(model.estimators_)} alberi")
        print(f"   ma n_estimators è configurato a {model.n_estimators}")
    
    print()
    
    # STEP 6: Verifica alberi individuali
    print(f"🌳 VERIFICA ALBERI INDIVIDUALI:")
    
    if len(model.estimators_) == 0:
        print(f"❌ ERRORE: Nessun albero trovato nel modello!")
        return False
    
    first_tree = model.estimators_[0]
    last_tree = model.estimators_[-1]
    
    print(f"   Primo albero:")
    print(f"      - Tipo: {type(first_tree).__name__}")
    print(f"      - Ha metodo predict: {hasattr(first_tree, 'predict')}")
    print(f"      - Ha attributo tree_: {hasattr(first_tree, 'tree_')}")
    
    print(f"   Ultimo albero:")
    print(f"      - Tipo: {type(last_tree).__name__}")
    print(f"      - Ha metodo predict: {hasattr(last_tree, 'predict')}")
    
    # Verifica che tutti gli alberi abbiano gli attributi necessari
    invalid_trees = []
    for i, tree in enumerate(model.estimators_):
        if not (hasattr(tree, 'predict') and hasattr(tree, 'tree_')):
            invalid_trees.append(i)
    
    if invalid_trees:
        print(f"\n❌ ERRORE: {len(invalid_trees)} alberi non validi trovati!")
        print(f"   Indici alberi problematici: {invalid_trees[:10]}...")  # Mostra primi 10
        return False
    
    print(f"\n✅ Tutti i {len(model.estimators_)} alberi sono validi\n")
    
    # STEP 7: Test predizione
    print(f"🔬 TEST PREDIZIONE SU DATI FITTIZI:")
    
    n_features = model.n_features_in_
    n_samples = 10
    
    try:
        # Genera dati casuali con la giusta dimensionalità
        np.random.seed(42)
        X_test = np.random.random((n_samples, n_features))
        
        print(f"   Generati {n_samples} campioni con {n_features} feature")
        print(f"   Shape input: {X_test.shape}")
        
        # Test predict
        predictions = model.predict(X_test)
        print(f"\n   ✅ Predizioni:")
        print(f"      - Shape: {predictions.shape}")
        print(f"      - Valori: {predictions}")
        print(f"      - Classi uniche predette: {np.unique(predictions)}")
        
        # Test predict_proba
        probabilities = model.predict_proba(X_test)
        print(f"\n   ✅ Probabilità:")
        print(f"      - Shape: {probabilities.shape}")
        print(f"      - Prime 3 probabilità (Classe 0 | Classe 1):")
        
        for i in range(min(3, len(probabilities))):
            prob_0 = probabilities[i][0]
            prob_1 = probabilities[i][1]
            pred_class = predictions[i]
            print(f"         Campione {i+1}: {prob_0:.4f} | {prob_1:.4f}  → Predetto: {pred_class}")
        
        # Verifica somma probabilità = 1
        prob_sums = probabilities.sum(axis=1)
        all_sum_to_one = np.allclose(prob_sums, 1.0)
        
        if all_sum_to_one:
            print(f"\n   ✅ Le probabilità sommano correttamente a 1.0")
        else:
            print(f"\n   ⚠️ WARNING: Alcune probabilità non sommano a 1.0")
            print(f"      Range somme: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante test predizione: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # STEP 8: Riepilogo finale
    print(f"\n{'='*80}")
    print(f"✅ VERIFICA COMPLETATA CON SUCCESSO!")
    print(f"{'='*80}")
    print(f"\nIl modello Random Forest federato è:")
    print(f"   ✅ Caricabile correttamente")
    print(f"   ✅ Strutturalmente valido ({len(model.estimators_)} alberi)")
    print(f"   ✅ Funzionante (predizioni e probabilità corrette)")
    print(f"\n🔬 Il modello è PRONTO per gli attacchi adversarial!\n")
    print(f"Path verificato: {os.path.abspath(model_path)}")
    print(f"{'='*80}\n")
    
    return True


def main():
    """Funzione principale per verifica da linea di comando."""
    
    if len(sys.argv) < 2:
        print("Uso: python3 verify_saved_model.py <path_modello.pkl>")
        print("\nEsempio:")
        print("  python3 verify_saved_model.py models/federated_rf_global_20251121_014431.pkl")
        print("\nOppure trova automaticamente l'ultimo modello salvato:")
        
        # Cerca automaticamente l'ultimo modello
        models_dir = "models"
        if os.path.exists(models_dir):
            pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f.startswith('federated_rf_global_')]
            if pkl_files:
                # Ordina per data (assumendo formato timestamp nel nome)
                pkl_files.sort(reverse=True)
                latest = os.path.join(models_dir, pkl_files[0])
                print(f"\n  Ultimo modello trovato: {latest}")
                print(f"  Uso: python3 verify_saved_model.py {latest}")
        
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Esegui verifica
    success = verify_model(model_path)
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()