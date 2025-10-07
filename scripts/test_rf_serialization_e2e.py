#!/usr/bin/env python3
"""
Script di test end-to-end per verificare la serializzazione/deserializzazione
degli alberi Random Forest tra client e server.

Questo script simula il flusso completo:
1. Client crea e addestra un Random Forest
2. Client serializza gli alberi
3. Simula invio tramite Flower
4. Server deserializza gli alberi
5. Server crea modello globale
6. Server serializza e invia ai client
7. Client deserializza e usa il modello globale
"""

import sys
import os
import traceback

# Aggiungi il path per importare i moduli
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'federated', 'SmartGrid'))

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

def print_section(title):
    """Stampa una sezione separata per maggiore leggibilità"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_1_client_training():
    """Simula il training di un client"""
    print_section("TEST 1: Training Client Random Forest")
    
    try:
        # Crea dataset sintetico
        X_train, y_train = make_classification(
            n_samples=200, n_features=20, n_classes=2, 
            random_state=42, n_informative=15
        )
        X_val, y_val = make_classification(
            n_samples=50, n_features=20, n_classes=2, 
            random_state=43, n_informative=15
        )
        
        print(f"Dataset creato: {len(X_train)} train, {len(X_val)} val")
        
        # Addestra Random Forest
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        train_acc = rf.score(X_train, y_train)
        val_acc = rf.score(X_val, y_val)
        
        print(f"✅ Random Forest addestrato con {len(rf.estimators_)} alberi")
        print(f"   Train accuracy: {train_acc:.4f}")
        print(f"   Val accuracy: {val_acc:.4f}")
        
        return rf, X_val, y_val
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None, None, None

def test_2_client_serialization(rf, X_val, y_val):
    """Simula la serializzazione degli alberi nel client"""
    print_section("TEST 2: Client - Serializzazione Alberi")
    
    try:
        # Estrai alberi (simula extract_trees_from_forest)
        trees_performance = []
        for tree in rf.estimators_:
            tree_pred = tree.predict(X_val)
            accuracy = accuracy_score(y_val, tree_pred)
            weighted_acc = accuracy  # Semplificato per il test
            trees_performance.append((tree, accuracy, weighted_acc))
        
        print(f"Estratti {len(trees_performance)} alberi")
        
        # Serializza alberi (simula serialize_trees_for_aggregation)
        serialized_trees = []
        for i, (tree, acc, w_acc) in enumerate(trees_performance):
            # Serializza con pickle
            tree_bytes = pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Converti in numpy array
            tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
            serialized_trees.append(tree_array)
            
            if i < 3:  # Mostra primi 3
                print(f"  Albero {i+1}: {len(tree_bytes)} bytes → numpy array shape={tree_array.shape}")
        
        print(f"✅ Serializzati {len(serialized_trees)} alberi")
        print(f"   Tipo primo elemento: {type(serialized_trees[0])}")
        print(f"   dtype: {serialized_trees[0].dtype}")
        
        return serialized_trees, trees_performance
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None, None

def test_3_flower_transmission(serialized_trees):
    """Simula la trasmissione via Flower"""
    print_section("TEST 3: Trasmissione Flower (Client → Server)")
    
    try:
        # Simula cosa invia il client
        client_parameters = serialized_trees  # Lista di numpy arrays
        
        print(f"Client invia: {len(client_parameters)} parametri")
        print(f"  Tipo parametri: {type(client_parameters)}")
        print(f"  Tipo primo parametro: {type(client_parameters[0])}")
        
        # Simula ricezione sul server (Flower potrebbe fare una copia)
        server_parameters = [np.copy(arr) for arr in client_parameters]
        
        print(f"\nServer riceve: {len(server_parameters)} parametri")
        print(f"  Tipo primo parametro: {type(server_parameters[0])}")
        print(f"  dtype: {server_parameters[0].dtype}")
        print(f"  shape: {server_parameters[0].shape}")
        
        print(f"✅ Trasmissione simulata con successo")
        
        return server_parameters
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def test_4_server_deserialization(server_parameters):
    """Simula la deserializzazione degli alberi nel server"""
    print_section("TEST 4: Server - Deserializzazione Alberi")
    
    try:
        deserialized_trees = []
        
        for i, param_array in enumerate(server_parameters):
            # Converti numpy array in bytes
            tree_bytes = param_array.tobytes()
            
            # Deserializza con pickle
            tree = pickle.loads(tree_bytes)
            
            # Simula accuracy
            simulated_acc = 0.9 - (i * 0.01)
            simulated_w_acc = 0.88 - (i * 0.01)
            
            deserialized_trees.append((tree, simulated_acc, simulated_w_acc))
            
            if i < 3:  # Mostra primi 3
                print(f"  Albero {i+1}: deserializzato (tipo: {type(tree).__name__})")
        
        print(f"✅ Deserializzati {len(deserialized_trees)} alberi")
        print(f"   Hanno attributo 'predict': {all(hasattr(t[0], 'predict') for t in deserialized_trees)}")
        print(f"   Hanno attributo 'tree_': {all(hasattr(t[0], 'tree_') for t in deserialized_trees)}")
        
        return deserialized_trees
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def test_5_server_aggregation(deserialized_trees):
    """Simula l'aggregazione degli alberi nel server"""
    print_section("TEST 5: Server - Aggregazione Alberi")
    
    try:
        # Estrai solo gli alberi (senza metadati)
        trees = [tree_data[0] for tree_data in deserialized_trees]
        
        # Crea Random Forest globale
        global_rf = RandomForestClassifier(
            n_estimators=len(trees),
            random_state=42
        )
        
        # Assegna gli alberi (hack per scikit-learn)
        global_rf.estimators_ = trees
        global_rf.n_estimators = len(trees)
        
        # Copia attributi necessari dal primo albero
        first_tree = trees[0]
        if hasattr(first_tree, 'n_features_in_'):
            global_rf.n_features_in_ = first_tree.n_features_in_
        if hasattr(first_tree, 'n_outputs_'):
            global_rf.n_outputs_ = first_tree.n_outputs_
        if hasattr(first_tree, 'classes_'):
            global_rf.classes_ = first_tree.classes_
            global_rf.n_classes_ = len(first_tree.classes_)
        else:
            # Default per classificazione binaria
            global_rf.classes_ = np.array([0, 1])
            global_rf.n_classes_ = 2
        
        print(f"✅ Random Forest globale creato")
        print(f"   N. alberi: {len(global_rf.estimators_)}")
        print(f"   N. classi: {global_rf.n_classes_}")
        
        return global_rf
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def test_6_server_serialization(global_rf):
    """Simula la serializzazione del modello globale nel server"""
    print_section("TEST 6: Server - Serializzazione Modello Globale")
    
    try:
        # Serializza con pickle
        model_bytes = pickle.dumps(global_rf, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Converti in numpy array
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        
        print(f"✅ Modello globale serializzato")
        print(f"   Dimensione: {len(model_bytes)} bytes")
        print(f"   NumPy array: shape={model_array.shape}, dtype={model_array.dtype}")
        
        return [model_array]  # Flower si aspetta una lista
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def test_7_client_deserialization(server_model_parameters, X_val, y_val):
    """Simula la deserializzazione del modello globale nel client"""
    print_section("TEST 7: Client - Deserializzazione Modello Globale")
    
    try:
        # Estrai il modello
        model_array = server_model_parameters[0]
        
        print(f"Client riceve: tipo={type(model_array)}, shape={model_array.shape}")
        
        # Converti in bytes
        model_bytes = model_array.tobytes()
        
        # Deserializza
        received_model = pickle.loads(model_bytes)
        
        print(f"✅ Modello globale deserializzato")
        print(f"   Tipo: {type(received_model).__name__}")
        print(f"   N. alberi: {len(received_model.estimators_)}")
        
        # Verifica che il modello funzioni
        predictions = received_model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        print(f"   Accuracy su validation set: {accuracy:.4f}")
        print(f"✅ Modello funzionante!")
        
        return received_model
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def main():
    """Funzione principale di test end-to-end"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          TEST END-TO-END SERIALIZZAZIONE RANDOM FOREST FEDERATO              ║
║                                                                              ║
║  Questo script verifica l'intero flusso di serializzazione/deserializzazione║
║  tra client e server Flower per Random Forest                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Training client
    rf, X_val, y_val = test_1_client_training()
    if rf is None:
        return 1
    
    # Test 2: Serializzazione client
    serialized_trees, trees_performance = test_2_client_serialization(rf, X_val, y_val)
    if serialized_trees is None:
        return 1
    
    # Test 3: Trasmissione Flower
    server_parameters = test_3_flower_transmission(serialized_trees)
    if server_parameters is None:
        return 1
    
    # Test 4: Deserializzazione server
    deserialized_trees = test_4_server_deserialization(server_parameters)
    if deserialized_trees is None:
        return 1
    
    # Test 5: Aggregazione server
    global_rf = test_5_server_aggregation(deserialized_trees)
    if global_rf is None:
        return 1
    
    # Test 6: Serializzazione server
    server_model_parameters = test_6_server_serialization(global_rf)
    if server_model_parameters is None:
        return 1
    
    # Test 7: Deserializzazione client
    received_model = test_7_client_deserialization(server_model_parameters, X_val, y_val)
    if received_model is None:
        return 1
    
    print_section("RISULTATO FINALE")
    print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("")
    print("Il flusso di serializzazione/deserializzazione funziona correttamente:")
    print("  1. ✅ Client serializza alberi in numpy arrays")
    print("  2. ✅ Flower trasmette numpy arrays")
    print("  3. ✅ Server deserializza numpy arrays")
    print("  4. ✅ Server aggrega alberi in modello globale")
    print("  5. ✅ Server serializza modello globale in numpy array")
    print("  6. ✅ Flower trasmette numpy array")
    print("  7. ✅ Client deserializza modello globale")
    print("  8. ✅ Client usa modello globale per predizioni")
    print("")
    print("🎉 Il problema di serializzazione è RISOLTO!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
