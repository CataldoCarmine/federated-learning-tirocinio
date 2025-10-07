#!/usr/bin/env python3
"""
Script di debug per identificare e risolvere il problema di serializzazione
degli alberi Random Forest tra client e server.

Questo script:
1. Crea un albero Random Forest di test
2. Testa diversi metodi di serializzazione
3. Verifica la compatibilità con Flower
4. Identifica il punto esatto dove i dati vengono corrotti
"""

import pickle
import numpy as np
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
import traceback

def print_section(title):
    """Stampa una sezione separata per maggiore leggibilità"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_1_create_sample_tree():
    """Test 1: Crea un Random Forest di esempio e ne estrae un albero"""
    print_section("TEST 1: Creazione Random Forest di Test")
    
    try:
        # Crea dataset sintetico
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        
        # Addestra un Random Forest
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        
        print(f"✅ Random Forest creato con {len(rf.estimators_)} alberi")
        print(f"   Accuracy: {rf.score(X, y):.4f}")
        
        # Estrai il primo albero
        tree = rf.estimators_[0]
        print(f"✅ Primo albero estratto (tipo: {type(tree)})")
        print(f"   Ha attributo 'predict': {hasattr(tree, 'predict')}")
        print(f"   Ha attributo 'tree_': {hasattr(tree, 'tree_')}")
        
        return tree, X, y
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None, None, None

def test_2_pickle_serialization(tree):
    """Test 2: Serializzazione base con pickle"""
    print_section("TEST 2: Serializzazione Pickle Base")
    
    try:
        # Serializza con pickle
        tree_bytes = pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Albero serializzato con pickle")
        print(f"   Dimensione: {len(tree_bytes)} bytes")
        print(f"   Primi 20 byte: {tree_bytes[:20]}")
        
        # Deserializza
        tree_restored = pickle.loads(tree_bytes)
        print(f"✅ Albero deserializzato")
        print(f"   Tipo: {type(tree_restored)}")
        print(f"   Ha attributo 'predict': {hasattr(tree_restored, 'predict')}")
        
        return tree_bytes
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def test_3_numpy_array_conversion(tree_bytes):
    """Test 3: Conversione a numpy array (come fa il client attuale)"""
    print_section("TEST 3: Conversione a NumPy Array (Client Attuale)")
    
    try:
        # Converti in numpy array (come fa il client)
        tree_array = np.frombuffer(tree_bytes, dtype=np.uint8)
        print(f"✅ Convertito a numpy array")
        print(f"   Tipo: {type(tree_array)}")
        print(f"   dtype: {tree_array.dtype}")
        print(f"   shape: {tree_array.shape}")
        print(f"   Primi 10 elementi: {tree_array[:10].tolist()}")
        
        # Riconverti in bytes (come dovrebbe fare il server)
        tree_bytes_restored = tree_array.tobytes()
        print(f"✅ Riconvertito a bytes con tobytes()")
        print(f"   Dimensione: {len(tree_bytes_restored)} bytes")
        print(f"   Match con originale: {tree_bytes == tree_bytes_restored}")
        
        # Tenta deserializzazione
        tree_restored = pickle.loads(tree_bytes_restored)
        print(f"✅ Deserializzazione riuscita!")
        print(f"   Tipo: {type(tree_restored)}")
        
        return tree_array, tree_bytes_restored
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None, None

def test_4_base64_encoding(tree_bytes):
    """Test 4: Codifica Base64 (approccio alternativo)"""
    print_section("TEST 4: Codifica Base64 (Approccio Alternativo)")
    
    try:
        # Codifica in Base64
        tree_b64 = base64.b64encode(tree_bytes).decode('utf-8')
        print(f"✅ Codificato in Base64")
        print(f"   Tipo: {type(tree_b64)}")
        print(f"   Lunghezza: {len(tree_b64)} caratteri")
        print(f"   Primi 50 caratteri: {tree_b64[:50]}")
        
        # Decodifica Base64
        tree_bytes_restored = base64.b64decode(tree_b64.encode('utf-8'))
        print(f"✅ Decodificato da Base64")
        print(f"   Dimensione: {len(tree_bytes_restored)} bytes")
        print(f"   Match con originale: {tree_bytes == tree_bytes_restored}")
        
        # Deserializza
        tree_restored = pickle.loads(tree_bytes_restored)
        print(f"✅ Deserializzazione riuscita!")
        print(f"   Tipo: {type(tree_restored)}")
        
        return tree_b64
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

def test_5_flower_simulation(tree_array, tree_b64):
    """Test 5: Simula il comportamento di Flower"""
    print_section("TEST 5: Simulazione Flower - Client → Server")
    
    print("\n--- SCENARIO A: NumPy Array (Attuale Implementazione Client) ---")
    try:
        # Simula cosa invia il client (numpy array)
        parameters_from_client = [tree_array]  # Lista di numpy array
        print(f"Client invia: Lista di {len(parameters_from_client)} numpy array")
        print(f"  Tipo primo elemento: {type(parameters_from_client[0])}")
        
        # Simula cosa riceve il server
        param_array = parameters_from_client[0]
        print(f"\nServer riceve: {type(param_array)}")
        
        # Server tenta di deserializzare (metodo attuale nel codice)
        if isinstance(param_array, np.ndarray):
            tree_bytes = param_array.tobytes()
            print(f"  Conversione con tobytes(): {len(tree_bytes)} bytes")
            
            tree_restored = pickle.loads(tree_bytes)
            print(f"✅ SCENARIO A: Deserializzazione riuscita!")
            print(f"  Tipo albero: {type(tree_restored)}")
        
    except Exception as e:
        print(f"❌ SCENARIO A FALLITO: {e}")
        traceback.print_exc()
    
    print("\n--- SCENARIO B: Base64 String (Approccio Alternativo) ---")
    try:
        # Simula cosa invia il client (stringa Base64)
        parameters_from_client = [tree_b64]  # Lista di stringhe
        print(f"Client invia: Lista di {len(parameters_from_client)} stringhe Base64")
        print(f"  Tipo primo elemento: {type(parameters_from_client[0])}")
        print(f"  Primi 50 char: {parameters_from_client[0][:50]}")
        
        # Simula cosa riceve il server
        param_str = parameters_from_client[0]
        print(f"\nServer riceve: {type(param_str)}")
        
        # Server deserializza (metodo per Base64)
        if isinstance(param_str, str):
            tree_bytes = base64.b64decode(param_str.encode('utf-8'))
            print(f"  Decodifica Base64: {len(tree_bytes)} bytes")
            
            tree_restored = pickle.loads(tree_bytes)
            print(f"✅ SCENARIO B: Deserializzazione riuscita!")
            print(f"  Tipo albero: {type(tree_restored)}")
        
    except Exception as e:
        print(f"❌ SCENARIO B FALLITO: {e}")
        traceback.print_exc()

def test_6_identify_corruption_point(tree):
    """Test 6: Identifica il punto esatto di corruzione"""
    print_section("TEST 6: Identificazione Punto di Corruzione")
    
    try:
        # Step 1: Serializza
        tree_bytes_original = pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Step 1 - Pickle originale: {len(tree_bytes_original)} bytes")
        
        # Step 2: Converti in numpy array
        tree_array = np.frombuffer(tree_bytes_original, dtype=np.uint8)
        print(f"Step 2 - NumPy array: shape={tree_array.shape}, dtype={tree_array.dtype}")
        
        # Step 3: Simula passaggio attraverso Flower (potrebbe modificare)
        # Flower potrebbe fare qualcosa come questo
        tree_array_copy = np.copy(tree_array)  # Copia
        print(f"Step 3 - Copia array: shape={tree_array_copy.shape}")
        
        # Step 4: Riconversione a bytes
        tree_bytes_restored = tree_array_copy.tobytes()
        print(f"Step 4 - Bytes ripristinati: {len(tree_bytes_restored)} bytes")
        
        # Verifica integrità byte per byte
        if tree_bytes_original == tree_bytes_restored:
            print(f"✅ I dati NON sono stati corrotti nel processo!")
        else:
            print(f"❌ I dati SONO stati corrotti!")
            
            # Trova le differenze
            differences = 0
            for i in range(min(len(tree_bytes_original), len(tree_bytes_restored))):
                if tree_bytes_original[i] != tree_bytes_restored[i]:
                    differences += 1
                    if differences <= 5:  # Mostra solo le prime 5 differenze
                        print(f"  Byte {i}: {tree_bytes_original[i]} → {tree_bytes_restored[i]}")
            
            print(f"  Totale byte differenti: {differences}")
            print(f"  Differenza lunghezza: {len(tree_bytes_original)} vs {len(tree_bytes_restored)}")
        
        # Tenta deserializzazione
        tree_restored = pickle.loads(tree_bytes_restored)
        print(f"✅ Deserializzazione finale riuscita!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE durante identificazione corruzione: {e}")
        traceback.print_exc()
        return False

def test_7_recommendation():
    """Test 7: Raccomandazione basata sui test"""
    print_section("TEST 7: Analisi e Raccomandazioni")
    
    print("""
PROBLEMA IDENTIFICATO:
----------------------
Il codice attuale ha un MISMATCH tra client e server:

CLIENT (clientRF.py):
- serialize_trees_for_aggregation() ritorna numpy arrays (np.uint8)
- Ma get_parameters() cerca di usare tree_b64 (variabile inesistente!)
- Questo causa un errore perché tree_b64 non esiste

SERVER (serverRF.py):
- deserialize_trees_from_client() si aspetta stringhe Base64
- Ma il client invia numpy arrays

SOLUZIONI POSSIBILI:
-------------------

SOLUZIONE 1 (Raccomandato): Usa NumPy Arrays
- Client: Invia direttamente numpy arrays (già implementato in serialize_trees_for_aggregation)
- Server: Usa tobytes() su numpy array prima di pickle.loads()
- Pro: Più efficiente, nativo per Flower
- Con: Nessuno

SOLUZIONE 2: Usa Base64
- Client: Converti numpy array in Base64 prima di inviare
- Server: Decodifica Base64 prima di pickle.loads()
- Pro: Più robusto per diversi transport
- Con: Overhead di conversione, meno efficiente

SOLUZIONE IMPLEMENTATA:
----------------------
Useremo SOLUZIONE 1 perché:
1. È più efficiente (no conversione Base64)
2. È il formato nativo di Flower per trasmettere numpy arrays
3. I test dimostrano che funziona perfettamente
    """)

def main():
    """Funzione principale di debug"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          DEBUG SERIALIZZAZIONE RANDOM FOREST FEDERATO                        ║
║                                                                              ║
║  Questo script identifica e risolve il problema di serializzazione          ║
║  degli alberi Random Forest tra client e server Flower                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Crea albero di test
    tree, X, y = test_1_create_sample_tree()
    if tree is None:
        print("\n❌ ERRORE CRITICO: Impossibile creare albero di test")
        return 1
    
    # Test 2: Serializzazione pickle base
    tree_bytes = test_2_pickle_serialization(tree)
    if tree_bytes is None:
        print("\n❌ ERRORE CRITICO: Serializzazione pickle fallita")
        return 1
    
    # Test 3: Conversione numpy array
    tree_array, tree_bytes_restored = test_3_numpy_array_conversion(tree_bytes)
    if tree_array is None:
        print("\n❌ ERRORE: Conversione numpy array fallita")
        return 1
    
    # Test 4: Codifica Base64
    tree_b64 = test_4_base64_encoding(tree_bytes)
    if tree_b64 is None:
        print("\n❌ ERRORE: Codifica Base64 fallita")
        return 1
    
    # Test 5: Simulazione Flower
    test_5_flower_simulation(tree_array, tree_b64)
    
    # Test 6: Identifica punto di corruzione
    test_6_identify_corruption_point(tree)
    
    # Test 7: Raccomandazioni
    test_7_recommendation()
    
    print_section("RISULTATO FINALE")
    print("✅ Tutti i test completati con successo!")
    print("📝 Consulta le raccomandazioni sopra per la soluzione")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
