"""
attacks/defense_utils.py

Funzioni utility per Adversarial Training Defense.

Questo modulo fornisce funzioni avanzate per:
- Calcolo vincoli fisici con percentili personalizzabili
- Applicazione vincoli adattivi basati su feature importance
- Calcolo feature importance per guidare difesa

UTILIZZO:
    from attacks.defense_utils import apply_adaptive_constraints
    
    X_adv_safe = apply_adaptive_constraints(X_adv, X_orig, constraints, epsilon)

AUTORE: Carmine Cataldo
DATA: 2025-01-24
"""

import numpy as np
from typing import Dict, Optional


def get_smartgrid_physical_constraints_advanced(X_sample, percentile_low=0.1, percentile_high=99.9):
    """
    Calcola vincoli fisici avanzati con percentili personalizzabili.
    
    UTILIZZO PER ADVERSARIAL TRAINING:
    Durante il training adversarial, vogliamo vincoli più permissivi
    per permettere esplorazione, ma comunque realistici.
    
    Args:
        X_sample: Campione di dati per calcolare statistiche
        percentile_low: Percentile inferiore (default: 0.1)
        percentile_high: Percentile superiore (default: 99.9)
        
    Returns:
        Dictionary con vincoli estesi:
            - feature_min: Minimo per feature
            - feature_max: Massimo per feature
            - feature_mean: Media per feature
            - feature_std: Deviazione standard per feature
    """
    feature_min = np.percentile(X_sample, percentile_low, axis=0)
    feature_max = np.percentile(X_sample, percentile_high, axis=0)
    feature_mean = np.mean(X_sample, axis=0)
    feature_std = np.std(X_sample, axis=0)
    
    return {
        'feature_min': feature_min,
        'feature_max': feature_max,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'percentile_low': percentile_low,
        'percentile_high': percentile_high
    }


def apply_adaptive_constraints(X_adv, X_original, constraints, epsilon, feature_importance=None):
    """
    Applica vincoli adattivi basati su importanza feature.
    
    STRATEGIA AVANZATA:
    - Feature importanti: vincoli più stretti (piccola perturbazione)
    - Feature non importanti: vincoli più permissivi
    
    Questo aiuta a:
    1. Preservare realismo fisico
    2. Concentrare perturbazioni su feature critiche
    3. Ridurre rumore su feature irrilevanti
    
    Args:
        X_adv: Esempi adversarial
        X_original: Esempi originali
        constraints: Vincoli fisici
        epsilon: Budget perturbazione base
        feature_importance: Array con importanza feature (opzionale)
        
    Returns:
        X_adv_constrained: Esempi con vincoli applicati
    """
    X_constrained = X_adv.copy()
    
    # STEP 1: Vincolo range fisico globale
    X_constrained = np.clip(
        X_constrained,
        constraints['feature_min'],
        constraints['feature_max']
    )
    
    # STEP 2: Vincolo perturbazione con epsilon adattivo
    perturbation = X_constrained - X_original
    
    if feature_importance is not None:
        # Epsilon adattivo basato su importanza
        # Feature importanti: epsilon ridotto
        # Feature non importanti: epsilon aumentato
        
        # Normalizza importanza a [0, 1]
        importance_norm = (feature_importance - feature_importance.min()) / \
                         (feature_importance.max() - feature_importance.min() + 1e-10)
        
        # Calcola epsilon per feature
        # Feature importante (importance=1) → epsilon * 0.5
        # Feature non importante (importance=0) → epsilon * 2.0
        epsilon_per_feature = epsilon * (2.0 - 1.5 * importance_norm)
        
        # Clip perturbazione con epsilon adattivo
        for i in range(X_constrained.shape[1]):
            perturbation[:, i] = np.clip(
                perturbation[:, i],
                -epsilon_per_feature[i],
                epsilon_per_feature[i]
            )
    else:
        # Epsilon uniforme (fallback)
        perturbation = np.clip(perturbation, -epsilon, epsilon)
    
    # Applica perturbazione vincolata
    X_constrained = X_original + perturbation
    
    # STEP 3: Ri-applica range fisico
    X_constrained = np.clip(
        X_constrained,
        constraints['feature_min'],
        constraints['feature_max']
    )
    
    return X_constrained


def calculate_feature_importance_for_defense(model, X_sample, method='gini'):
    """
    Calcola feature importance per guidare difesa adversarial.
    
    UTILIZZO:
    Durante adversarial training, possiamo usare feature importance per:
    - Concentrare perturbazioni su feature critiche
    - Applicare vincoli adattivi
    - Ridurre tempo computazionale (perturbare solo feature importanti)
    
    Args:
        model: RandomForestClassifier
        X_sample: Campione per validazione (opzionale)
        method: Metodo calcolo ('gini', 'permutation')
        
    Returns:
        importance: Array con importanza per ogni feature
    """
    if method == 'gini' and hasattr(model, 'feature_importances_'):
        # Usa Gini importance del Random Forest
        importance = model.feature_importances_
        
    elif method == 'permutation':
        # Permutation importance (più lento ma più accurato)
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X_sample, model.predict(X_sample),
            n_repeats=10,
            random_state=42
        )
        importance = result.importances_mean
    else:
        # Fallback: importanza uniforme
        importance = np.ones(model.n_features_in_) / model.n_features_in_
    
    return importance