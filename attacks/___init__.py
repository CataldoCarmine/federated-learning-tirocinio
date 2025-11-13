"""
attacks/__init__.py

Modulo per implementazione attacchi evasion su modelli di Federated Learning.

Questo modulo implementa diversi tipi di attacchi adversariali:
- White-box: Decision Tree Attack (accesso completo al modello)
- Black-box: HopSkipJump Attack (solo query al modello)
- Transfer: PGD via Surrogate Model (transferability)

Autore: Cataldo Carmine
Progetto: Federated Learning SmartGrid IDS
"""

__version__ = "1.0.0"
__author__ = "Cataldo Carmine"

# CORREZIONE: Import assoluto invece che relativo per evitare errori
try:
    from attacks.whitebox_decision_tree_attack import WhiteBoxDecisionTreeAttack
    __all__ = ['WhiteBoxDecisionTreeAttack']
except ImportError as e:
    print(f"⚠️ Errore import WhiteBoxDecisionTreeAttack: {e}")
    __all__ = []