"""
attacks/data_analysis.py

Funzioni di analisi per comprendere la struttura dei dati SmartGrid
e guidare la generazione di esempi adversarial più efficaci.

FUNZIONALITÀ:
1. Feature importance analysis (Random Forest)
2. Analisi distribuzione feature per classe (Attack vs Natural)
3. Identificazione feature critiche per boundary decisionale
4. Analisi correlazioni tra feature
5. Visualizzazione decision boundaries

UTILIZZO:
    from attacks.data_analysis import (
        analyze_feature_importance,
        analyze_class_distributions,
        find_critical_features_for_evasion
    )
    
    # Analizza feature importance
    importance = analyze_feature_importance(model, X_train, y_train)
    
    # Trova feature critiche per evasion Attack → Natural
    critical_features = find_critical_features_for_evasion(
        model, X_attacks, target_class=0
    )

AUTORE: Carmine Cataldo
DATA: 2025-01-21
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
import os
from datetime import datetime


def analyze_feature_importance(model, X, y, top_n=20, save_report=True):
    """
    Analizza l'importanza delle feature nel Random Forest.
    
    SPIEGAZIONE:
    Random Forest calcola feature importance basandosi su quante volte
    una feature viene usata per split e quanto riduce l'impurity.
    
    Feature con importance alta = più critiche per le predizioni
    → Perturbare QUESTE feature ha maggiore probabilità di successo
    
    Args:
        model: RandomForestClassifier federato
        X: Dati preprocessati
        y: Etichette (0=Natural, 1=Attack)
        top_n: Numero di feature più importanti da mostrare
        save_report: Se True, salva report in file
        
    Returns:
        Dictionary con:
            - feature_importance: Array con importance per ogni feature
            - top_features_indices: Indici delle top_n feature più importanti
            - top_features_importance: Valori importance delle top_n feature
            
    Example:
        >>> importance_info = analyze_feature_importance(model, X_train, y_train)
        >>> print(f"Top 5 feature: {importance_info['top_features_indices'][:5]}")
    """
    print("="*80)
    print("📊 ANALISI FEATURE IMPORTANCE")
    print("="*80)
    
    # Estrai feature importance dal Random Forest
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        # Se il modello non ha feature_importances_, calcoliamo manualmente
        print("⚠️ Modello non ha feature_importances_, calcolo manuale...")
        feature_importance = np.zeros(model.n_features_in_)
        
        # Media delle feature importance degli alberi individuali
        for tree in model.estimators_:
            if hasattr(tree, 'feature_importances_'):
                feature_importance += tree.feature_importances_
        
        feature_importance /= len(model.estimators_)
    
    # Ordina per importance decrescente
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    # Seleziona top N
    top_features_indices = sorted_indices[:top_n]
    top_features_importance = feature_importance[top_features_indices]
    
    # Stampa report
    print(f"\n🔝 TOP {top_n} FEATURE PIÙ IMPORTANTI:\n")
    print(f"{'Rank':<6} {'Feature':<12} {'Importance':<12} {'Importanza %':<15}")
    print("-"*50)
    
    total_importance = feature_importance.sum()
    cumulative = 0.0
    
    for i, (idx, imp) in enumerate(zip(top_features_indices, top_features_importance), 1):
        cumulative += imp
        pct = (imp / total_importance) * 100 if total_importance > 0 else 0
        print(f"{i:<6} Feature_{idx:<4} {imp:<12.6f} {pct:<6.2f}% (cum: {(cumulative/total_importance)*100:.1f}%)")
    
    print(f"\n📈 STATISTICHE GLOBALI:")
    print(f"  - Feature totali: {len(feature_importance)}")
    print(f"  - Importance media: {feature_importance.mean():.6f}")
    print(f"  - Top {top_n} coprono: {(cumulative/total_importance)*100:.1f}% dell'importance totale")
    
    # Identifica feature quasi inutili (importance < 0.001)
    useless_features = np.sum(feature_importance < 0.001)
    print(f"  - Feature quasi inutili (imp<0.001): {useless_features}")
    
    # Salva report se richiesto
    if save_report:
        report_dir = os.path.join("attacks", "analysis_reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(report_dir, f"feature_importance_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ANALISI FEATURE IMPORTANCE - RANDOM FOREST FEDERATO\n")
            f.write("="*80 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Campioni analizzati: {len(X)}\n")
            f.write(f"Feature totali: {len(feature_importance)}\n\n")
            
            f.write(f"TOP {top_n} FEATURE:\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<12} {'Importance':<12}\n")
            
            for i, (idx, imp) in enumerate(zip(top_features_indices, top_features_importance), 1):
                f.write(f"{i:<6} Feature_{idx:<4} {imp:<12.6f}\n")
            
            f.write("\nSTATISTICHE:\n")
            f.write(f"Importance media: {feature_importance.mean():.6f}\n")
            f.write(f"Importance std: {feature_importance.std():.6f}\n")
            f.write(f"Feature inutili: {useless_features}\n")
        
        print(f"\n✅ Report salvato: {report_path}")
    
    print("="*80 + "\n")
    
    return {
        'feature_importance': feature_importance,
        'top_features_indices': top_features_indices,
        'top_features_importance': top_features_importance,
        'total_features': len(feature_importance),
        'useless_features_count': int(useless_features)
    }


def analyze_class_distributions(X, y, feature_indices=None, save_report=True):
    """
    Analizza le distribuzioni delle feature per classe (Attack vs Natural).
    
    SPIEGAZIONE:
    Per evadere (Attack → Natural), dobbiamo capire:
    - Come sono DISTRIBUITE le feature in campioni Attack
    - Come sono DISTRIBUITE le feature in campioni Natural
    - Quale DIREZIONE spostare le feature per sembrare Natural
    
    Esempio:
    Se Feature_42 ha:
    - Attack: media=0.8, std=0.1
    - Natural: media=0.3, std=0.05
    → Per evadere, dobbiamo RIDURRE Feature_42 verso 0.3
    
    Args:
        X: Dati preprocessati
        y: Etichette (0=Natural, 1=Attack)
        feature_indices: Indici feature da analizzare (None = tutte)
        save_report: Se True, salva report in file
        
    Returns:
        Dictionary con statistiche per ogni feature:
            - mean_attack: Media feature in campioni Attack
            - std_attack: Std feature in campioni Attack
            - mean_natural: Media feature in campioni Natural
            - std_natural: Std feature in campioni Natural
            - direction: Direzione per evasion (+1 o -1)
            - separation: Quanto sono separate le distribuzioni
    """
    print("="*80)
    print("📊 ANALISI DISTRIBUZIONE FEATURE PER CLASSE")
    print("="*80)
    
    # Separa dati per classe
    X_attack = X[y == 1]
    X_natural = X[y == 0]
    
    print(f"\n📌 DATASET:")
    print(f"  - Campioni Attack: {len(X_attack)}")
    print(f"  - Campioni Natural: {len(X_natural)}")
    
    # Seleziona feature da analizzare
    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))
    
    # Calcola statistiche per ogni feature
    stats_per_feature = []
    
    for feat_idx in feature_indices:
        # Statistiche Attack
        mean_attack = X_attack[:, feat_idx].mean()
        std_attack = X_attack[:, feat_idx].std()
        
        # Statistiche Natural
        mean_natural = X_natural[:, feat_idx].mean()
        std_natural = X_natural[:, feat_idx].std()
        
        # Direzione per evasion (Attack → Natural)
        # +1 = aumentare feature, -1 = diminuire feature
        direction = +1 if mean_natural > mean_attack else -1
        
        # Separazione tra distribuzioni (Cohen's d)
        pooled_std = np.sqrt((std_attack**2 + std_natural**2) / 2)
        cohens_d = abs(mean_attack - mean_natural) / pooled_std if pooled_std > 0 else 0
        
        # Test statistico (t-test)
        t_stat, p_value = stats.ttest_ind(X_attack[:, feat_idx], X_natural[:, feat_idx])
        
        stats_per_feature.append({
            'feature_idx': feat_idx,
            'mean_attack': mean_attack,
            'std_attack': std_attack,
            'mean_natural': mean_natural,
            'std_natural': std_natural,
            'direction': direction,
            'cohens_d': cohens_d,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    # Converti in DataFrame per analisi
    df_stats = pd.DataFrame(stats_per_feature)
    
    # Ordina per separazione (Cohen's d) decrescente
    df_stats_sorted = df_stats.sort_values('cohens_d', ascending=False)
    
    # Stampa top 20 feature più discriminanti
    print(f"\n🔝 TOP 20 FEATURE PIÙ DISCRIMINANTI (Cohen's d):\n")
    print(f"{'Feature':<10} {'Mean_Attack':<13} {'Mean_Natural':<14} {'Direction':<11} {'Cohen_d':<10} {'Signif':<8}")
    print("-"*85)
    
    for _, row in df_stats_sorted.head(20).iterrows():
        feat = f"F{row['feature_idx']}"
        dir_symbol = "↑" if row['direction'] == +1 else "↓"
        signif = "✓" if row['significant'] else "✗"
        print(f"{feat:<10} {row['mean_attack']:<13.6f} {row['mean_natural']:<14.6f} "
              f"{dir_symbol:<11} {row['cohens_d']:<10.4f} {signif:<8}")
    
    # Statistiche globali
    n_significant = df_stats['significant'].sum()
    mean_cohens_d = df_stats['cohens_d'].mean()
    
    print(f"\n📈 STATISTICHE GLOBALI:")
    print(f"  - Feature significativamente diverse: {n_significant}/{len(df_stats)} ({n_significant/len(df_stats)*100:.1f}%)")
    print(f"  - Cohen's d medio: {mean_cohens_d:.4f}")
    print(f"  - Feature con separazione alta (d>0.5): {(df_stats['cohens_d'] > 0.5).sum()}")
    
    # Salva report
    if save_report:
        report_dir = os.path.join("attacks", "analysis_reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(report_dir, f"class_distributions_{timestamp}.csv")
        
        df_stats_sorted.to_csv(report_path, index=False)
        print(f"\n✅ Report CSV salvato: {report_path}")
    
    print("="*80 + "\n")
    
    return df_stats.to_dict('records')


def find_critical_features_for_evasion(model, X_attacks, y_attacks, target_class=0, 
                                        n_critical=20, save_report=True):
    """
    Identifica le feature CRITICHE per evasion Attack → Natural.
    
    STRATEGIA:
    1. Per ogni feature, prova a perturbarla SINGOLARMENTE
    2. Misura quanti campioni riescono ad evadere modificando SOLO quella feature
    3. Feature con più evasioni = PIÙ CRITICHE per attacco
    
    UTILITÀ:
    Questa funzione ci dice:
    - Quali feature sono più EFFICACI per evasion
    - Su quali feature CONCENTRARE le perturbazioni
    - Quanto è sensibile il modello a cambiamenti su singole feature
    
    Args:
        model: RandomForestClassifier federato
        X_attacks: Campioni di attacco preprocessati
        y_attacks: Etichette (tutte = 1)
        target_class: Classe target per evasion (default: 0 = Natural)
        n_critical: Numero di feature critiche da restituire
        save_report: Se True, salva report in file
        
    Returns:
        Dictionary con:
            - critical_features_indices: Indici delle feature più critiche
            - evasion_success_rate: % di evasioni per ogni feature
            - optimal_directions: Direzione ottimale (+1 o -1) per ogni feature
    """
    print("="*80)
    print("🎯 IDENTIFICAZIONE FEATURE CRITICHE PER EVASION")
    print("="*80)
    
    print(f"\n📌 CONFIGURAZIONE:")
    print(f"  - Campioni di test: {len(X_attacks)}")
    print(f"  - Feature totali: {X_attacks.shape[1]}")
    print(f"  - Target class: {target_class} (Natural)")
    print(f"  - Strategia: Perturbazione SINGOLA feature per volta")
    
    # Predizioni originali
    preds_original = model.predict(X_attacks)
    
    # Limita a primi 500 campioni per velocità (test)
    if len(X_attacks) > 500:
        print(f"\n⚠️ Limitando a 500 campioni per velocità test...")
        X_attacks_test = X_attacks[:500]
        preds_original_test = preds_original[:500]
    else:
        X_attacks_test = X_attacks
        preds_original_test = preds_original
    
    # Array per salvare risultati
    feature_criticality = []
    
    print(f"\n🔬 TESTING PERTURBAZIONE SINGOLA FEATURE...")
    print(f"Testando {X_attacks.shape[1]} feature...")
    
    for feat_idx in range(X_attacks.shape[1]):
        if feat_idx % 20 == 0:
            print(f"  Progress: {feat_idx}/{X_attacks.shape[1]} feature testate...")
        
        # Prova a perturbare SOLO questa feature
        # Test con perturbazione positiva
        X_perturbed_pos = X_attacks_test.copy()
        X_perturbed_pos[:, feat_idx] += 0.5  # Perturbazione fissa (test)
        preds_pos = model.predict(X_perturbed_pos)
        evasions_pos = np.sum((preds_original_test == 1) & (preds_pos == target_class))
        
        # Test con perturbazione negativa
        X_perturbed_neg = X_attacks_test.copy()
        X_perturbed_neg[:, feat_idx] -= 0.5  # Perturbazione fissa (test)
        preds_neg = model.predict(X_perturbed_neg)
        evasions_neg = np.sum((preds_original_test == 1) & (preds_neg == target_class))
        
        # Seleziona direzione migliore
        if evasions_pos > evasions_neg:
            evasions_best = evasions_pos
            optimal_direction = +1
        else:
            evasions_best = evasions_neg
            optimal_direction = -1
        
        # Success rate
        success_rate = evasions_best / len(X_attacks_test)
        
        feature_criticality.append({
            'feature_idx': feat_idx,
            'evasions_count': int(evasions_best),
            'success_rate': float(success_rate),
            'optimal_direction': int(optimal_direction),
            'tested_samples': len(X_attacks_test)
        })
    
    # Converti in DataFrame e ordina
    df_criticality = pd.DataFrame(feature_criticality)
    df_criticality_sorted = df_criticality.sort_values('success_rate', ascending=False)
    
    # Top N feature critiche
    top_critical = df_criticality_sorted.head(n_critical)
    
    # Stampa risultati
    print(f"\n🔝 TOP {n_critical} FEATURE CRITICHE PER EVASION:\n")
    print(f"{'Rank':<6} {'Feature':<10} {'Success Rate':<14} {'Evasions':<11} {'Direction':<12}")
    print("-"*60)
    
    for i, (_, row) in enumerate(top_critical.iterrows(), 1):
        feat = f"F{row['feature_idx']}"
        dir_symbol = "↑ (increase)" if row['optimal_direction'] == +1 else "↓ (decrease)"
        print(f"{i:<6} {feat:<10} {row['success_rate']*100:<12.2f}% {row['evasions_count']:<11} {dir_symbol:<12}")
    
    # Statistiche
    print(f"\n📈 STATISTICHE:")
    print(f"  - Feature con evasions>0: {(df_criticality['evasions_count'] > 0).sum()}/{len(df_criticality)}")
    print(f"  - Max success rate: {df_criticality['success_rate'].max()*100:.2f}%")
    print(f"  - Mean success rate: {df_criticality['success_rate'].mean()*100:.2f}%")
    
    # Salva report
    if save_report:
        report_dir = os.path.join("attacks", "analysis_reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(report_dir, f"critical_features_{timestamp}.csv")
        
        df_criticality_sorted.to_csv(report_path, index=False)
        print(f"\n✅ Report CSV salvato: {report_path}")
    
    print("="*80 + "\n")
    
    return {
        'critical_features_indices': top_critical['feature_idx'].values,
        'success_rates': top_critical['success_rate'].values,
        'optimal_directions': top_critical['optimal_direction'].values,
        'full_results': df_criticality_sorted.to_dict('records')
    }


def run_complete_data_analysis(model, X_train, y_train, X_test, y_test):
    """
    Esegue analisi completa dei dati per guidare attacchi adversarial.
    
    WORKFLOW:
    1. Feature importance (Random Forest)
    2. Distribuzione feature per classe
    3. Feature critiche per evasion
    4. Summary e raccomandazioni
    
    Args:
        model: RandomForestClassifier federato
        X_train: Dati training preprocessati
        y_train: Etichette training
        X_test: Dati test preprocessati
        y_test: Etichette test
        
    Returns:
        Dictionary completo con tutti i risultati analisi
    """
    print("\n" + "="*80)
    print("🔬 ANALISI COMPLETA DATI SMARTGRID PER ATTACCHI ADVERSARIAL")
    print("="*80 + "\n")
    
    # ANALISI 1: Feature Importance
    importance_info = analyze_feature_importance(model, X_train, y_train, top_n=20)
    
    # ANALISI 2: Distribuzioni per classe
    class_dist_info = analyze_class_distributions(
        X_train, y_train, 
        feature_indices=importance_info['top_features_indices'][:30],  # Analizza solo top 30
        save_report=True
    )
    
    # ANALISI 3: Feature critiche per evasion
    X_attacks_test = X_test[y_test == 1]
    y_attacks_test = y_test[y_test == 1]
    
    critical_info = find_critical_features_for_evasion(
        model, X_attacks_test, y_attacks_test, 
        target_class=0, n_critical=20
    )
    
    # SUMMARY
    print("\n" + "="*80)
    print("📋 SUMMARY E RACCOMANDAZIONI PER ATTACCHI")
    print("="*80)
    
    print(f"\n🎯 TOP 10 FEATURE DA PERTURBARE (based on analysis):\n")
    
    # Combina feature importance + criticality
    important_features = set(importance_info['top_features_indices'][:10])
    critical_features = set(critical_info['critical_features_indices'][:10])
    
    # Feature che sono ENTRAMBE importanti E critiche (priorità massima)
    priority_features = important_features.intersection(critical_features)
    
    print(f"🔴 PRIORITÀ MASSIMA (importanti E critiche): {list(priority_features)}")
    print(f"🟡 IMPORTANTE (ma meno critica): {list(important_features - priority_features)}")
    print(f"🟢 CRITICA (ma meno importante): {list(critical_features - priority_features)}")
    
    print(f"\n💡 RACCOMANDAZIONI:")
    print(f"  1. Concentra perturbazioni su feature: {list(priority_features)[:5]}")
    print(f"  2. Usa direction ottimali da critical_features analysis")
    print(f"  3. Considera distribuzione Natural per target values")
    print(f"  4. Monte Carlo dovrebbe usare SOLO queste feature (non tutte le 128)")
    
    print("="*80 + "\n")
    
    return {
        'feature_importance': importance_info,
        'class_distributions': class_dist_info,
        'critical_features': critical_info,
        'priority_features': list(priority_features),
        'recommendations': {
            'top_features_to_perturb': list(priority_features)[:10],
            'optimal_directions': dict(zip(
                critical_info['critical_features_indices'],
                critical_info['optimal_directions']
            ))
        }
    }