#!/usr/bin/env python3
"""
Test script for serverRF.py
Verifies all functionality without requiring actual client connections
"""

import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# Import both client and server modules
import clientRF
import serverRF

def test_preprocessing():
    """Test preprocessing functions"""
    print("=" * 60)
    print("TEST 1: Preprocessing Functions")
    print("=" * 60)
    
    # Test clip_outliers_iqr
    X = np.random.randn(100, 10)
    X[0, 0] = 1000  # Add outlier
    X_clipped = serverRF.clip_outliers_iqr(X)
    assert X_clipped.shape == X.shape, "Shape mismatch after clipping"
    assert X_clipped[0, 0] < 100, "Outlier not clipped"
    print("✓ clip_outliers_iqr works correctly")
    
    # Test remove_near_constant_features
    X2 = np.random.randn(100, 10)
    X2[:, 0] = 1  # Make first column constant
    X_filtered, mask = serverRF.remove_near_constant_features(X2)
    assert X_filtered.shape[1] < X2.shape[1], "Constant feature not removed"
    print("✓ remove_near_constant_features works correctly")
    
    print("✓ All preprocessing tests passed\n")

def test_tree_serialization():
    """Test tree serialization and deserialization"""
    print("=" * 60)
    print("TEST 2: Tree Serialization/Deserialization")
    print("=" * 60)
    
    # Create a sample tree
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    # Test serialization
    tree_bytes = pickle.dumps(tree)
    tree_restored = serverRF.deserialize_tree_from_bytes(tree_bytes)
    
    # Verify restored tree works
    y_pred1 = tree.predict(X)
    y_pred2 = tree_restored.predict(X)
    assert np.array_equal(y_pred1, y_pred2), "Predictions differ after deserialization"
    print("✓ Tree serialization/deserialization works correctly\n")

def test_tree_selection():
    """Test tree selection methods"""
    print("=" * 60)
    print("TEST 3: Tree Selection Methods")
    print("=" * 60)
    
    # Create mock trees
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    trees_by_client = {
        1: [
            {'tree': tree, 'accuracy': 0.9, 'weighted_accuracy': 0.85, 'tree_size': 100},
            {'tree': tree, 'accuracy': 0.8, 'weighted_accuracy': 0.75, 'tree_size': 100},
        ],
        2: [
            {'tree': tree, 'accuracy': 0.95, 'weighted_accuracy': 0.92, 'tree_size': 100},
            {'tree': tree, 'accuracy': 0.88, 'weighted_accuracy': 0.86, 'tree_size': 100},
        ]
    }
    
    # Test per_forest selection
    selected = serverRF.select_trees_per_forest(trees_by_client, method='accuracy', max_trees_per_client=1)
    assert len(selected) == 2, f"Expected 2 trees, got {len(selected)}"
    print("✓ select_trees_per_forest works correctly")
    
    # Test global selection
    selected_global = serverRF.select_trees_global(trees_by_client, method='weighted_accuracy', max_trees_global=3)
    assert len(selected_global) == 3, f"Expected 3 trees, got {len(selected_global)}"
    # Check best trees are selected
    accuracies = [t['weighted_accuracy'] for t in selected_global]
    assert all(accuracies[i] >= accuracies[i+1] for i in range(len(accuracies)-1)), "Trees not sorted by accuracy"
    print("✓ select_trees_global works correctly\n")

def test_global_rf_construction():
    """Test Global Random Forest construction"""
    print("=" * 60)
    print("TEST 4: Global Random Forest Construction")
    print("=" * 60)
    
    # Create mock trees
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    trees_list = []
    for i in range(5):
        tree = DecisionTreeClassifier(max_depth=3, random_state=42+i)
        tree.fit(X, y)
        trees_list.append({
            'tree': tree,
            'accuracy': 0.85 + i * 0.02,
            'weighted_accuracy': 0.83 + i * 0.02,
            'tree_size': 100,
            'tree_index': i,
            'client_id': (i % 2) + 1
        })
    
    # Test weighted voting
    global_rf = serverRF.create_global_random_forest(trees_list, ensemble_method='weighted_voting')
    assert len(global_rf.estimators_) == 5, "Wrong number of estimators"
    assert hasattr(global_rf, 'tree_weights_'), "Tree weights not set"
    assert len(global_rf.tree_weights_) == 5, "Wrong number of weights"
    print("✓ Global RF construction with weighted voting works")
    
    # Test predictions
    y_pred = global_rf.predict(X)
    assert y_pred.shape == (100,), f"Wrong prediction shape: {y_pred.shape}"
    y_pred_proba = global_rf.predict_proba(X)
    assert y_pred_proba.shape == (100, 2), f"Wrong probability shape: {y_pred_proba.shape}"
    print("✓ Global RF predictions work correctly\n")

def test_client_server_compatibility():
    """Test full compatibility between client and server"""
    print("=" * 60)
    print("TEST 5: Client-Server Compatibility")
    print("=" * 60)
    
    # Create data with realistic feature count
    X_train = np.random.randn(100, 128)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 128)
    y_val = np.random.randint(0, 2, 50)
    
    # Client side: create and serialize RF
    rf_model = RandomForestClassifier(
        n_estimators=5,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    trees_data = clientRF.serialize_random_forest_trees(rf_model, X_val, y_val)
    assert len(trees_data) == 5, "Wrong number of serialized trees"
    print("✓ Client serialization works")
    
    # Convert to Flower format
    client_parameters = []
    for tree_data in trees_data:
        tree_params = np.concatenate([
            np.array([tree_data['tree_index']], dtype=np.float32),
            np.array([tree_data['tree_size']], dtype=np.float32),
            np.array([tree_data['accuracy']], dtype=np.float32),
            np.array([tree_data['weighted_accuracy']], dtype=np.float32),
            tree_data['tree_data'].astype(np.float32)
        ])
        client_parameters.append(tree_params)
    
    # Server side: deserialize
    deserialized_trees = serverRF.deserialize_trees_from_client(client_parameters)
    assert len(deserialized_trees) == 5, "Wrong number of deserialized trees"
    print("✓ Server deserialization works")
    
    # Server: create Global RF
    all_trees_by_client = {1: deserialized_trees}
    selected_trees = serverRF.select_trees_per_forest(
        all_trees_by_client,
        method='weighted_accuracy',
        max_trees_per_client=3
    )
    global_rf = serverRF.create_global_random_forest(selected_trees, 'weighted_voting')
    assert len(global_rf.estimators_) == 3, "Wrong number of trees in global RF"
    print("✓ Server aggregation works")
    
    # Server: serialize back to clients
    server_parameters = serverRF.serialize_trees_for_clients(global_rf, X_val, y_val)
    assert len(server_parameters) == 3, "Wrong number of serialized trees from server"
    print("✓ Server serialization back to clients works")
    
    # Client: deserialize Global RF
    client_trees_data = []
    for tree_params in server_parameters:
        tree_dict = {
            'tree_index': int(tree_params[0]),
            'tree_size': int(tree_params[1]),
            'accuracy': float(tree_params[2]),
            'weighted_accuracy': float(tree_params[3]),
            'tree_data': tree_params[4:].astype(np.uint8)
        }
        client_trees_data.append(tree_dict)
    
    client_global_rf = clientRF.deserialize_random_forest_trees(client_trees_data)
    assert len(client_global_rf.estimators_) == 3, "Wrong number of trees in client's global RF"
    print("✓ Client deserialization of Global RF works")
    
    # Verify predictions match
    y_pred_server = global_rf.predict(X_val)
    y_pred_client = client_global_rf.predict(X_val)
    assert np.array_equal(y_pred_server, y_pred_client), "Server and client predictions differ"
    print("✓ Server and client predictions match\n")

def test_strategy_class():
    """Test strategy class initialization"""
    print("=" * 60)
    print("TEST 6: Strategy Class")
    print("=" * 60)
    
    try:
        strategy = serverRF.SmartGridRandomForestFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=serverRF.get_smartgrid_evaluate_fn()
        )
        assert hasattr(strategy, 'aggregate_fit'), "Missing aggregate_fit method"
        assert hasattr(strategy, 'aggregate_evaluate'), "Missing aggregate_evaluate method"
        assert strategy.evaluate_fn is not None, "evaluate_fn not set"
        print("✓ Strategy class initialization works")
        
        # Test helper method
        X, y, info = strategy._get_test_data()
        if X is not None:
            assert X.shape[0] > 0, "Test data is empty"
            assert y.shape[0] == X.shape[0], "X and y shape mismatch"
            print(f"✓ _get_test_data works (loaded {X.shape[0]} samples)")
        else:
            print("⚠ Test data not available (expected in some environments)")
        
    except FileNotFoundError:
        print("⚠ Test data files not available (expected in some environments)")
    except Exception as e:
        print(f"✗ Strategy class error: {e}")
        raise
    
    print()

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING SERVERRF.PY TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_preprocessing()
        test_tree_serialization()
        test_tree_selection()
        test_global_rf_construction()
        test_client_server_compatibility()
        test_strategy_class()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
