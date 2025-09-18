#!/usr/bin/env python3
"""
Configuration templates for SmartGrid federated learning experiments.

Copy and paste the desired configuration section into your files:
- federated/SmartGrid/client.py
- federated/SmartGrid/server.py  
- centralized/SmartGrid/centralized.py

Replace the existing configuration section (starting with "# ========== CONFIGURAZIONE ESPERIMENTO ==========")
"""

# ========== TEMPLATE 1: DEFAULT CONFIGURATION (CURRENT BEHAVIOR) ==========
TEMPLATE_1_DEFAULT = """
# ========== CONFIGURAZIONE ESPERIMENTO ==========
# Controllo PCA e rimozione feature quasi-costanti
ENABLE_PCA = True  # True/False per abilitare/disabilitare PCA
ENABLE_REMOVE_NEAR_CONSTANT = True  # True/False per rimozione feature quasi-costanti
RANDOM_SEED = 42  # Seed globale per riproducibilità

# CONFIGURAZIONE PCA DINAMICA
PCA_COMPONENTS = 74  # Numero componenti quando PCA è abilitata
PCA_RANDOM_STATE = RANDOM_SEED

# Numero di feature dinamico basato sulla configurazione
INPUT_FEATURES = PCA_COMPONENTS if ENABLE_PCA else (128 if not ENABLE_REMOVE_NEAR_CONSTANT else 89)  # 89 è il numero tipico dopo rimozione quasi-costanti per SmartGrid
"""

# ========== TEMPLATE 2: ORIGINAL DATASET (NO DIMENSIONALITY REDUCTION) ==========
TEMPLATE_2_ORIGINAL = """
# ========== CONFIGURAZIONE ESPERIMENTO ==========
# Controllo PCA e rimozione feature quasi-costanti
ENABLE_PCA = False  # True/False per abilitare/disabilitare PCA
ENABLE_REMOVE_NEAR_CONSTANT = False  # True/False per rimozione feature quasi-costanti
RANDOM_SEED = 42  # Seed globale per riproducibilità

# CONFIGURAZIONE PCA DINAMICA
PCA_COMPONENTS = 74  # Numero componenti quando PCA è abilitata
PCA_RANDOM_STATE = RANDOM_SEED

# Numero di feature dinamico basato sulla configurazione
INPUT_FEATURES = PCA_COMPONENTS if ENABLE_PCA else (128 if not ENABLE_REMOVE_NEAR_CONSTANT else 89)  # 89 è il numero tipico dopo rimozione quasi-costanti per SmartGrid
"""

# ========== TEMPLATE 3: PCA ONLY (NO CONSTANT REMOVAL) ==========
TEMPLATE_3_PCA_ONLY = """
# ========== CONFIGURAZIONE ESPERIMENTO ==========
# Controllo PCA e rimozione feature quasi-costanti
ENABLE_PCA = True  # True/False per abilitare/disabilitare PCA
ENABLE_REMOVE_NEAR_CONSTANT = False  # True/False per rimozione feature quasi-costanti
RANDOM_SEED = 42  # Seed globale per riproducibilità

# CONFIGURAZIONE PCA DINAMICA
PCA_COMPONENTS = 74  # Numero componenti cuando PCA è abilitata
PCA_RANDOM_STATE = RANDOM_SEED

# Numero di feature dinamico basato sulla configurazione
INPUT_FEATURES = PCA_COMPONENTS if ENABLE_PCA else (128 if not ENABLE_REMOVE_NEAR_CONSTANT else 89)  # 89 è il numero tipico dopo rimozione quasi-costanti per SmartGrid
"""

# ========== TEMPLATE 4: CONSTANT REMOVAL ONLY (NO PCA) ==========
TEMPLATE_4_CONSTANT_ONLY = """
# ========== CONFIGURAZIONE ESPERIMENTO ==========
# Controllo PCA e rimozione feature quasi-costanti
ENABLE_PCA = False  # True/False per abilitare/disabilitare PCA
ENABLE_REMOVE_NEAR_CONSTANT = True  # True/False per rimozione feature quasi-costanti
RANDOM_SEED = 42  # Seed globale per riproducibilità

# CONFIGURAZIONE PCA DINAMICA
PCA_COMPONENTS = 74  # Numero componenti quando PCA è abilitata
PCA_RANDOM_STATE = RANDOM_SEED

# Numero di feature dinamico basato sulla configurazione
INPUT_FEATURES = PCA_COMPONENTS if ENABLE_PCA else (128 if not ENABLE_REMOVE_NEAR_CONSTANT else 89)  # 89 è il numero tipico dopo rimozione quasi-costanti per SmartGrid
"""

def print_configurations():
    """Print all available configuration templates"""
    configs = [
        ("DEFAULT (Current Behavior)", TEMPLATE_1_DEFAULT, "74 features", 
         "PCA applied after removing near-constant features"),
        ("ORIGINAL DATASET", TEMPLATE_2_ORIGINAL, "128 features", 
         "No dimensionality reduction, all original features"),
        ("PCA ONLY", TEMPLATE_3_PCA_ONLY, "74 features", 
         "PCA applied to all 128 original features"),
        ("CONSTANT REMOVAL ONLY", TEMPLATE_4_CONSTANT_ONLY, "89 features", 
         "Only near-constant features removed, no PCA")
    ]
    
    print("SMARTGRID CONFIGURATION TEMPLATES")
    print("=" * 80)
    
    for i, (name, template, features, description) in enumerate(configs, 1):
        print(f"\n{i}. {name}")
        print(f"   Expected features: {features}")
        print(f"   Description: {description}")
        print(f"   Template: TEMPLATE_{i}")
        print("-" * 50)
        print(template.strip())
        print("-" * 50)

if __name__ == "__main__":
    print_configurations()
    
    print("\nUSAGE INSTRUCTIONS:")
    print("=" * 80)
    print("1. Choose the desired configuration template above")
    print("2. Copy the template code")
    print("3. Replace the configuration section in these files:")
    print("   - federated/SmartGrid/client.py")
    print("   - federated/SmartGrid/server.py")
    print("   - centralized/SmartGrid/centralized.py")
    print("4. Ensure all three files have the SAME configuration")
    print("5. Run your training and observe the feature counts in console output")
    print("\nNOTE: All clients and server must use the same configuration for compatibility!")