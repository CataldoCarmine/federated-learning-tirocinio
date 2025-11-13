"""
Metriche di valutazione per attacchi evasion.

Questo modulo fornisce:
- Calcolo metriche di efficacia dell'attacco (ASR, FNR, etc.)
- Calcolo metriche di perturbazione (L-inf, L2, L0)
- Generazione report testuali
- Salvataggio risultati su file
- Generazione grafici comparativi

Le metriche implementate sono standard nella letteratura
sugli adversarial attacks per valutare sia l'efficacia
dell'attacco che l'impatto sul modello target.
"""

import numpy as np
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_attack(model, X_original, y_original, X_adv, attack_name="Attack"):
    """
    Calcola metriche complete per valutare l'efficacia dell'attacco evasion.
    
    Metriche calcolate:
    1. Attack Success Rate (ASR): percentuale campioni misclassificati
    2. ASR per classe: attack→natural (obiettivo principale)
    3. Perturbation magnitude: L-inf, L2, L0
    4. Metriche modello dopo attacco: accuracy, precision, recall, F1
    5. False Negative Rate (FNR): attacchi classificati come naturali
    
    Args:
        model: Modello target (RandomForestClassifier)
        X_original: Dati originali
        y_original: Etichette originali
        X_adv: Dati adversarial
        attack_name: Nome dell'attacco per il report
        
    Returns:
        dict: Dizionario con tutte le metriche
    """
    print(f"\n{'='*60}")
    print(f"VALUTAZIONE ATTACCO: {attack_name}")
    print(f"{'='*60}")
    
    # Predizioni originali e adversarial
    y_pred_original = model.predict(X_original)
    y_pred_adv = model.predict(X_adv)
    
    # 1. Attack Success Rate (ASR)
    misclassified = (y_pred_original != y_pred_adv)
    asr = misclassified.mean()
    
    # 2. ASR per classe (focus: attack → natural)
    attack_mask = (y_original == 1)  # Solo campioni di attacco
    if attack_mask.sum() > 0:
        attack_to_natural = (y_pred_original[attack_mask] == 1) & (y_pred_adv[attack_mask] == 0)
        asr_attack_to_natural = attack_to_natural.mean()
    else:
        asr_attack_to_natural = 0.0
    
    # Caso contrario: natural → attack (meno rilevante per IDS)
    natural_mask = (y_original == 0)
    if natural_mask.sum() > 0:
        natural_to_attack = (y_pred_original[natural_mask] == 0) & (y_pred_adv[natural_mask] == 1)
        asr_natural_to_attack = natural_to_attack.mean()
    else:
        asr_natural_to_attack = 0.0
    
    # 3. Perturbation Magnitude
    perturbation = X_adv - X_original
    
    # L-infinity: perturbazione massima su una singola feature
    l_inf = np.abs(perturbation).max()
    
    # L2: distanza euclidea media
    l2 = np.sqrt((perturbation ** 2).sum(axis=1)).mean()
    
    # L0: numero di feature modificate (media)
    l0 = (perturbation != 0).sum(axis=1).mean()
    
    # Percentuale feature modificate
    pct_features_modified = (l0 / X_original.shape[1]) * 100
    
    # 4. Metriche modello DOPO l'attacco
    accuracy_original = accuracy_score(y_original, y_pred_original)
    accuracy_after = accuracy_score(y_original, y_pred_adv)
    
    precision_after = precision_score(y_original, y_pred_adv, zero_division=0)
    recall_after = recall_score(y_original, y_pred_adv, zero_division=0)
    f1_after = f1_score(y_original, y_pred_adv, zero_division=0)
    balanced_acc_after = balanced_accuracy_score(y_original, y_pred_adv)
    
    # 5. False Negative Rate (FNR) - attacchi classificati come naturali
    fn = ((y_original == 1) & (y_pred_adv == 0)).sum()
    tp = ((y_original == 1) & (y_pred_adv == 1)).sum()
    fnr_after = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # False Positive Rate (FPR)
    fp = ((y_original == 0) & (y_pred_adv == 1)).sum()
    tn = ((y_original == 0) & (y_pred_adv == 0)).sum()
    fpr_after = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # 6. Confusion Matrix
    conf_matrix = confusion_matrix(y_original, y_pred_adv)
    
    # 7. Classification Report dettagliato
    class_report = classification_report(
        y_original, 
        y_pred_adv, 
        target_names=["natural", "attack"], 
        output_dict=True, 
        zero_division=0
    )
    
    # Costruisci dizionario metriche
    metrics = {
        'attack_name': attack_name,
        
        # Efficacia attacco
        'asr': float(asr),
        'asr_attack_to_natural': float(asr_attack_to_natural),
        'asr_natural_to_attack': float(asr_natural_to_attack),
        'samples_attacked': int(len(X_original)),
        'samples_misclassified': int(misclassified.sum()),
        
        # Perturbation magnitude
        'l_inf': float(l_inf),
        'l2': float(l2),
        'l0': float(l0),
        'pct_features_modified': float(pct_features_modified),
        
        # Performance modello
        'accuracy_original': float(accuracy_original),
        'accuracy_after': float(accuracy_after),
        'accuracy_degradation': float(accuracy_original - accuracy_after),
        'precision_after': float(precision_after),
        'recall_after': float(recall_after),
        'f1_after': float(f1_after),
        'balanced_accuracy_after': float(balanced_acc_after),
        
        # False rates
        'fnr_after': float(fnr_after),
        'fpr_after': float(fpr_after),
        
        # Per classe
        'precision_natural': float(class_report["natural"]["precision"]),
        'recall_natural': float(class_report["natural"]["recall"]),
        'f1_natural': float(class_report["natural"]["f1-score"]),
        'precision_attack': float(class_report["attack"]["precision"]),
        'recall_attack': float(class_report["attack"]["recall"]),
        'f1_attack': float(class_report["attack"]["f1-score"]),
        
        # Confusion matrix
        'tn': int(conf_matrix[0, 0]),
        'fp': int(conf_matrix[0, 1]),
        'fn': int(conf_matrix[1, 0]),
        'tp': int(conf_matrix[1, 1]),
    }
    
    return metrics


def print_attack_report(metrics):
    """
    Stampa report leggibile delle metriche di attacco.
    
    Args:
        metrics: Dizionario metriche da evaluate_attack()
    """
    print(f"\n{'='*60}")
    print(f"REPORT ATTACCO: {metrics['attack_name']}")
    print(f"{'='*60}")
    
    print(f"\n📊 EFFICACIA ATTACCO:")
    print(f"  - Attack Success Rate (ASR): {metrics['asr']:.2%}")
    print(f"  - ASR (Attack → Natural): {metrics['asr_attack_to_natural']:.2%}")
    print(f"  - ASR (Natural → Attack): {metrics['asr_natural_to_attack']:.2%}")
    print(f"  - Campioni misclassificati: {metrics['samples_misclassified']}/{metrics['samples_attacked']}")
    
    print(f"\n📏 MAGNITUDE PERTURBAZIONI:")
    print(f"  - L-infinity (max): {metrics['l_inf']:.6f}")
    print(f"  - L2 (media): {metrics['l2']:.6f}")
    print(f"  - L0 (feature modificate medie): {metrics['l0']:.1f}")
    print(f"  - % Feature modificate: {metrics['pct_features_modified']:.1f}%")
    
    print(f"\n🎯 IMPATTO SUL MODELLO:")
    print(f"  - Accuracy originale: {metrics['accuracy_original']:.2%}")
    print(f"  - Accuracy dopo attacco: {metrics['accuracy_after']:.2%}")
    print(f"  - Degradation: {metrics['accuracy_degradation']:.2%}")
    print(f"  - Precision dopo attacco: {metrics['precision_after']:.2%}")
    print(f"  - Recall dopo attacco: {metrics['recall_after']:.2%}")
    print(f"  - F1-Score dopo attacco: {metrics['f1_after']:.2%}")
    print(f"  - Balanced Accuracy: {metrics['balanced_accuracy_after']:.2%}")
    
    print(f"\n⚠️ FALSE RATES:")
    print(f"  - FNR (False Negative Rate): {metrics['fnr_after']:.2%}")
    print(f"  - FPR (False Positive Rate): {metrics['fpr_after']:.2%}")
    
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"              Predicted Natural  Predicted Attack")
    print(f"  Actual Natural:    {metrics['tn']:<8}     {metrics['fp']:<8}")
    print(f"  Actual Attack:     {metrics['fn']:<8}     {metrics['tp']:<8}")
    
    print(f"\n{'='*60}\n")


def save_attack_results(metrics, X_original, X_adv, epsilons_tested=None, save_dir="results/attacks"):
    """
    Salva risultati attacco su file (report + adversarial examples + grafici).
    
    Args:
        metrics: Dizionario metriche (o lista di dict se epsilons multipli)
        X_original: Dati originali
        X_adv: Adversarial examples (o dict {epsilon: X_adv} se multipli)
        epsilons_tested: Lista epsilon testati (opzionale)
        save_dir: Directory dove salvare risultati
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determina se abbiamo singolo epsilon o multipli
    is_multi_epsilon = isinstance(metrics, list)
    
    if is_multi_epsilon:
        attack_name = metrics[0]['attack_name'] if metrics else "Attack"
    else:
        attack_name = metrics['attack_name']
        metrics = [metrics]  # Converti a lista per uniformità
    
    # 1. SALVA REPORT TESTUALE
    report_path = os.path.join(save_dir, f"{attack_name}_{timestamp}_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"REPORT ATTACCO: {attack_name}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if is_multi_epsilon:
            f.write(f"Epsilon testati: {epsilons_tested}\n")
            f.write(f"Numero configurazioni: {len(metrics)}\n\n")
            
            # Tabella comparativa
            f.write(f"{'='*80}\n")
            f.write(f"TABELLA COMPARATIVA PER EPSILON\n")
            f.write(f"{'='*80}\n\n")
            
            # Header
            f.write(f"{'Epsilon':<12} {'ASR':<10} {'ASR_A→N':<12} {'L-inf':<12} {'L2':<12} {'Acc After':<12}\n")
            f.write(f"{'-'*80}\n")
            
            # Righe
            for i, m in enumerate(metrics):
                eps = epsilons_tested[i] if epsilons_tested else i
                f.write(f"{eps:<12.6f} {m['asr']:<10.4f} {m['asr_attack_to_natural']:<12.4f} "
                       f"{m['l_inf']:<12.6f} {m['l2']:<12.6f} {m['accuracy_after']:<12.4f}\n")
            
            f.write(f"\n{'='*80}\n\n")
        
        # Report dettagliato per ogni epsilon
        for i, m in enumerate(metrics):
            if is_multi_epsilon:
                eps = epsilons_tested[i] if epsilons_tested else i
                f.write(f"\n{'='*80}\n")
                f.write(f"DETTAGLI PER EPSILON = {eps}\n")
                f.write(f"{'='*80}\n\n")
            
            f.write(f"EFFICACIA ATTACCO:\n")
            f.write(f"  Attack Success Rate (ASR): {m['asr']:.4f} ({m['asr']*100:.2f}%)\n")
            f.write(f"  ASR (Attack → Natural): {m['asr_attack_to_natural']:.4f} ({m['asr_attack_to_natural']*100:.2f}%)\n")
            f.write(f"  ASR (Natural → Attack): {m['asr_natural_to_attack']:.4f} ({m['asr_natural_to_attack']*100:.2f}%)\n")
            f.write(f"  Campioni misclassificati: {m['samples_misclassified']}/{m['samples_attacked']}\n\n")
            
            f.write(f"MAGNITUDE PERTURBAZIONI:\n")
            f.write(f"  L-infinity (max): {m['l_inf']:.8f}\n")
            f.write(f"  L2 (media): {m['l2']:.8f}\n")
            f.write(f"  L0 (feature modificate): {m['l0']:.2f}\n")
            f.write(f"  % Feature modificate: {m['pct_features_modified']:.2f}%\n\n")
            
            f.write(f"IMPATTO SUL MODELLO:\n")
            f.write(f"  Accuracy originale: {m['accuracy_original']:.4f}\n")
            f.write(f"  Accuracy dopo attacco: {m['accuracy_after']:.4f}\n")
            f.write(f"  Degradation: {m['accuracy_degradation']:.4f}\n")
            f.write(f"  Precision: {m['precision_after']:.4f}\n")
            f.write(f"  Recall: {m['recall_after']:.4f}\n")
            f.write(f"  F1-Score: {m['f1_after']:.4f}\n")
            f.write(f"  Balanced Accuracy: {m['balanced_accuracy_after']:.4f}\n\n")
            
            f.write(f"FALSE RATES:\n")
            f.write(f"  FNR (False Negative Rate): {m['fnr_after']:.4f}\n")
            f.write(f"  FPR (False Positive Rate): {m['fpr_after']:.4f}\n\n")
            
            f.write(f"CONFUSION MATRIX:\n")
            f.write(f"  True Negatives (TN): {m['tn']}\n")
            f.write(f"  False Positives (FP): {m['fp']}\n")
            f.write(f"  False Negatives (FN): {m['fn']}\n")
            f.write(f"  True Positives (TP): {m['tp']}\n\n")
            
            f.write(f"METRICHE PER CLASSE:\n")
            f.write(f"  NATURAL - Precision: {m['precision_natural']:.4f}, Recall: {m['recall_natural']:.4f}, F1: {m['f1_natural']:.4f}\n")
            f.write(f"  ATTACK  - Precision: {m['precision_attack']:.4f}, Recall: {m['recall_attack']:.4f}, F1: {m['f1_attack']:.4f}\n\n")
    
    print(f"✅ Report salvato in: {report_path}")
    
    # 2. SALVA ADVERSARIAL EXAMPLES (.npy)
    if is_multi_epsilon and isinstance(X_adv, dict):
        # Salva ogni epsilon separatamente
        for eps, X_adv_eps in X_adv.items():
            adv_path = os.path.join(save_dir, f"{attack_name}_{timestamp}_X_adv_eps_{eps:.6f}.npy")
            np.save(adv_path, X_adv_eps[:100])  # Primi 100 per spazio
            print(f"✅ Adversarial examples (epsilon={eps}) salvati in: {adv_path}")
    else:
        # Singolo epsilon
        adv_path = os.path.join(save_dir, f"{attack_name}_{timestamp}_X_adv.npy")
        np.save(adv_path, X_adv[:100])  # Primi 100 per spazio
        print(f"✅ Adversarial examples salvati in: {adv_path}")
    
    # 3. GENERA GRAFICI
    if is_multi_epsilon:
        generate_multi_epsilon_plots(metrics, epsilons_tested, save_dir, attack_name, timestamp)
    else:
        generate_single_epsilon_plots(metrics[0], save_dir, attack_name, timestamp)


def generate_multi_epsilon_plots(metrics_list, epsilons, save_dir, attack_name, timestamp):
    """
    Genera grafici comparativi per diversi valori di epsilon.
    
    Args:
        metrics_list: Lista di dizionari metriche (uno per epsilon)
        epsilons: Lista valori epsilon testati
        save_dir: Directory output
        attack_name: Nome attacco
        timestamp: Timestamp per filename
    """
    print(f"\n[Grafici] Generazione grafici comparativi...")
    
    # Estrai metriche per plotting
    asrs = [m['asr'] for m in metrics_list]
    asrs_attack_to_natural = [m['asr_attack_to_natural'] for m in metrics_list]
    l_infs = [m['l_inf'] for m in metrics_list]
    l2s = [m['l2'] for m in metrics_list]
    accuracy_afters = [m['accuracy_after'] for m in metrics_list]
    fnrs = [m['fnr_after'] for m in metrics_list]
    
    # Crea figura con 6 subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{attack_name} - Analisi Multi-Epsilon', fontsize=16, fontweight='bold')
    
    # 1. ASR vs Epsilon
    axes[0, 0].plot(epsilons, asrs, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[0, 0].set_xlabel('Epsilon', fontsize=12)
    axes[0, 0].set_ylabel('Attack Success Rate', fontsize=12)
    axes[0, 0].set_title('ASR vs Epsilon', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    axes[0, 0].set_ylim([0, 1])
    
    # 2. ASR Attack→Natural vs Epsilon
    axes[0, 1].plot(epsilons, asrs_attack_to_natural, marker='s', linewidth=2, markersize=8, color='#3498db')
    axes[0, 1].set_xlabel('Epsilon', fontsize=12)
    axes[0, 1].set_ylabel('ASR (Attack → Natural)', fontsize=12)
    axes[0, 1].set_title('ASR Attack→Natural vs Epsilon', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    axes[0, 1].set_ylim([0, 1])
    
    # 3. L-infinity vs Epsilon
    axes[0, 2].plot(epsilons, l_infs, marker='^', linewidth=2, markersize=8, color='#2ecc71')
    axes[0, 2].set_xlabel('Epsilon', fontsize=12)
    axes[0, 2].set_ylabel('L-infinity Perturbation', fontsize=12)
    axes[0, 2].set_title('L-inf vs Epsilon', fontsize=14, fontweight='bold')
    axes[0, 2].grid(alpha=0.3, linestyle='--')
    
    # 4. L2 vs Epsilon
    axes[1, 0].plot(epsilons, l2s, marker='D', linewidth=2, markersize=8, color='#f39c12')
    axes[1, 0].set_xlabel('Epsilon', fontsize=12)
    axes[1, 0].set_ylabel('L2 Perturbation (mean)', fontsize=12)
    axes[1, 0].set_title('L2 vs Epsilon', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, linestyle='--')
    
    # 5. Accuracy After Attack vs Epsilon
    axes[1, 1].plot(epsilons, accuracy_afters, marker='v', linewidth=2, markersize=8, color='#9b59b6')
    axes[1, 1].set_xlabel('Epsilon', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy After Attack', fontsize=12)
    axes[1, 1].set_title('Accuracy Degradation vs Epsilon', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    axes[1, 1].set_ylim([0, 1])
    
    # 6. FNR vs Epsilon
    axes[1, 2].plot(epsilons, fnrs, marker='*', linewidth=2, markersize=10, color='#e67e22')
    axes[1, 2].set_xlabel('Epsilon', fontsize=12)
    axes[1, 2].set_ylabel('False Negative Rate', fontsize=12)
    axes[1, 2].set_title('FNR vs Epsilon', fontsize=14, fontweight='bold')
    axes[1, 2].grid(alpha=0.3, linestyle='--')
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Salva figura
    plot_path = os.path.join(save_dir, f"{attack_name}_{timestamp}_multi_epsilon_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafici multi-epsilon salvati in: {plot_path}")


def generate_single_epsilon_plots(metrics, save_dir, attack_name, timestamp):
    """
    Genera grafici per singolo valore di epsilon.
    
    Args:
        metrics: Dizionario metriche
        save_dir: Directory output
        attack_name: Nome attacco
        timestamp: Timestamp per filename
    """
    print(f"\n[Grafici] Generazione grafici single-epsilon...")
    
    # Crea figura con 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{attack_name} - Risultati', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    conf_matrix = np.array([[metrics['tn'], metrics['fp']],
                           [metrics['fn'], metrics['tp']]])
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Natural', 'Attack'],
                yticklabels=['Natural', 'Attack'],
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=12)
    
    # 2. Metriche chiave (bar chart)
    metric_names = ['ASR', 'ASR\n(A→N)', 'FNR', 'Accuracy\nAfter']
    metric_values = [
        metrics['asr'],
        metrics['asr_attack_to_natural'],
        metrics['fnr_after'],
        metrics['accuracy_after']
    ]
    colors = ['#e74c3c', '#3498db', '#e67e22', '#9b59b6']
    
    bars = axes[1].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Metriche Chiave', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Aggiungi valori sopra le barre
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Salva figura
    plot_path = os.path.join(save_dir, f"{attack_name}_{timestamp}_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafici salvati in: {plot_path}")