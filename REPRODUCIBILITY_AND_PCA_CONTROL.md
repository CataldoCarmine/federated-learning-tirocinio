# SmartGrid Reproducibility and PCA Control Implementation

## Overview

This implementation adds comprehensive reproducibility control and flexible PCA/feature preprocessing configuration to the SmartGrid federated learning system. The changes enable researchers to test different dimensionality reduction strategies while maintaining reproducible results.

## New Features

### 1. Reproducibility Control
- **Global seed management**: All random number generators (Python, NumPy, TensorFlow) use the same configurable seed
- **Deterministic operations**: TensorFlow operations are set to deterministic mode
- **Consistent results**: Same configuration with same seed produces identical results across runs

### 2. Flexible PCA Control
- **Conditional PCA**: Enable/disable PCA application via `ENABLE_PCA` flag
- **Conditional feature removal**: Control near-constant feature removal via `ENABLE_REMOVE_NEAR_CONSTANT` flag
- **Dynamic architecture**: Model input dimensions automatically adapt to preprocessing output

### 3. Configuration Management
- **Centralized configuration**: Easy-to-modify constants at the top of each file
- **Validation logging**: Configuration settings printed at startup for verification
- **Template system**: Pre-defined configuration templates for common scenarios

## Configuration Options

### Available Flags
```python
ENABLE_PCA = True/False                    # Enable/disable PCA application
ENABLE_REMOVE_NEAR_CONSTANT = True/False  # Enable/disable near-constant feature removal
RANDOM_SEED = 42                          # Global seed for reproducibility
```

### Four Configuration Combinations

| PCA | Remove Constants | Features | Description |
|-----|------------------|----------|-------------|
| ✓   | ✓               | 74       | **Default**: PCA after constant removal |
| ✗   | ✗               | 128      | **Original**: No dimensionality reduction |
| ✓   | ✗               | 74       | **PCA Only**: PCA on all 128 features |
| ✗   | ✓               | 89       | **Constants Only**: Remove constants, no PCA |

## Implementation Details

### Files Modified
1. **`federated/SmartGrid/client.py`**: Client-side preprocessing and model creation
2. **`federated/SmartGrid/server.py`**: Server-side evaluation and model creation
3. **`centralized/SmartGrid/centralized.py`**: Centralized training

### Key Changes

#### Reproducibility Function
```python
def set_reproducibility_seed(seed=42):
    """Imposta tutti i seed per garantire riproducibilità"""
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
```

#### Conditional Preprocessing
```python
# Conditional near-constant feature removal
if ENABLE_REMOVE_NEAR_CONSTANT:
    X_reduced, keep_mask = remove_near_constant_features(X_imputed)
    print(f"Features after constant removal: {X_reduced.shape[1]}")
else:
    X_reduced = X_imputed
    print(f"Constant removal DISABLED: {X_reduced.shape[1]} features")

# Conditional PCA application
if ENABLE_PCA:
    X_final = apply_pca(X_scaled)
    print(f"After PCA: {X_final.shape}")
else:
    X_final = X_scaled
    print(f"PCA DISABLED: {X_final.shape}")
```

#### Dynamic Model Architecture
```python
# Dynamic input features calculation
INPUT_FEATURES = PCA_COMPONENTS if ENABLE_PCA else (128 if not ENABLE_REMOVE_NEAR_CONSTANT else 89)

# Model with dynamic input layer
model = keras.Sequential([
    layers.Input(shape=(INPUT_FEATURES,), name='input_layer'),
    # ... rest of architecture
])
```

## Usage Guide

### Step 1: Choose Configuration
Select one of the four available configurations based on your experimental needs:

1. **Default (PCA + Constant Removal)**: Best for most experiments
2. **Original Dataset**: For baseline comparisons
3. **PCA Only**: To isolate PCA effects
4. **Constants Only**: To isolate constant removal effects

### Step 2: Update Configuration Files
Modify the configuration section in all three files:
- `federated/SmartGrid/client.py`
- `federated/SmartGrid/server.py`
- `centralized/SmartGrid/centralized.py`

**Important**: All files must have identical configuration for compatibility.

### Step 3: Run Training
```bash
# For centralized training
python centralized/SmartGrid/centralized.py

# For federated training
# Terminal 1 (Server)
python federated/SmartGrid/server.py

# Terminal 2-N (Clients)
python federated/SmartGrid/client.py 1
python federated/SmartGrid/client.py 2
# ... add more clients as needed
```

### Step 4: Verify Configuration
Check console output for:
- Configuration confirmation messages
- Feature counts after each preprocessing step
- Model architecture with correct input dimensions
- Seed confirmation messages

## Example Console Output

```
[Client] Seed riproducibilità impostato: 42
[Client] Configurazione: PCA=True, RemoveConstant=True
[Client 1] Feature dopo rimozione quasi-costanti: 89 (da 128)
[Client 1] PCA applicata: 89 → 74 feature
[Client] === CREAZIONE DNN ===
[Client] Input features: 74
[Client] Configurazione: PCA=True, RemoveConstant=True
```

## Testing and Validation

### Reproducibility Testing
1. Run the same configuration twice with the same seed
2. Compare results - they should be identical
3. Change the seed and verify results differ

### Configuration Testing
1. Test all four configuration combinations
2. Verify feature counts match expectations
3. Ensure model architectures adapt correctly
4. Confirm client-server compatibility

### Validation Script
A comprehensive test script is available at `/tmp/simple_test.py` that validates all four configurations:

```bash
python /tmp/simple_test.py
```

Expected output:
```
PCA=True, RemoveConstant=True       ->  74 features ✓ PASS
PCA=False, RemoveConstant=False     -> 128 features ✓ PASS
PCA=True, RemoveConstant=False      ->  74 features ✓ PASS
PCA=False, RemoveConstant=True      ->  89 features ✓ PASS
```

## Configuration Templates

Pre-defined configuration templates are available in `config_templates/smartgrid_configs.py`:

```bash
python config_templates/smartgrid_configs.py
```

This displays all available templates with copy-paste ready code.

## Best Practices

### Configuration Management
1. **Always use the same configuration** across client, server, and centralized files
2. **Document your configuration choice** in experiment logs
3. **Use meaningful seed values** for experiment tracking
4. **Test configuration changes** before running full experiments

### Reproducibility
1. **Set fixed seeds** for all reproducible experiments
2. **Record configuration settings** with experimental results
3. **Use version control** to track configuration changes
4. **Document environmental factors** (TensorFlow version, hardware, etc.)

### Experimentation
1. **Start with default configuration** to validate implementation
2. **Test one change at a time** to isolate effects
3. **Compare against original dataset** for baseline validation
4. **Use multiple seeds** to assess result stability

## Troubleshooting

### Common Issues

1. **Feature count mismatch between client and server**
   - Solution: Ensure all files have identical configuration

2. **Non-reproducible results**
   - Solution: Verify seed is set and TensorFlow deterministic mode is enabled

3. **Model architecture errors**
   - Solution: Check that INPUT_FEATURES matches preprocessing output

4. **Configuration not taking effect**
   - Solution: Verify configuration variables are modified in all three files

### Debugging Steps
1. Check console output for configuration confirmation
2. Verify feature counts at each preprocessing step
3. Compare expected vs actual INPUT_FEATURES values
4. Ensure all files import and set reproducibility correctly

## Performance Considerations

### Feature Count Impact
- **128 features**: Larger models, longer training, potentially better performance
- **89 features**: Moderate size, good balance of performance and efficiency
- **74 features**: Smaller models, faster training, dimensionality reduction benefits

### PCA Considerations
- **With PCA**: Reduced feature correlation, potential information loss
- **Without PCA**: Preserves original feature relationships, higher dimensionality

### Reproducibility Overhead
- Deterministic operations may reduce training speed slightly
- Trade-off between reproducibility and performance is generally acceptable

## Future Enhancements

Potential improvements to consider:
1. **Automated configuration validation** across files
2. **Command-line configuration options** to avoid file editing
3. **Configuration file format** (JSON/YAML) for easier management
4. **Experiment tracking integration** with configuration logging
5. **Performance benchmarking** across different configurations

## Conclusion

This implementation provides a robust foundation for reproducible SmartGrid federated learning experiments with flexible preprocessing options. The modular design allows researchers to easily test different dimensionality reduction strategies while maintaining experimental rigor through comprehensive reproducibility controls.