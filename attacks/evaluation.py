"""
attacks/evaluation.py

Funzioni per valutare l'efficacia degli attacchi adversarial.

Questo modulo fornisce:
- Calcolo Attack Success Rate (ASR)
- Metriche di perturbazione (L0, L2, L-inf)
- Report dettagliati con statistiche
- Salvataggio risultati in file

METRICHE PRINCIPALI:
- ASR (Attack Success Rate): % di attacchi evasi con successo
- Perturbazione L2: Norma euclidea media delle perturbazioni
- Perturbazione L-inf: Massimo cambiamento su singola feature
- Perturbazione L0: Numero di feature modificate

UTILIZZO:
    from attacks.evaluation import evaluate_attack, print_attack_report
    
    metrics = evaluate_attack(model, X_original, y_original, X_adversarial)
    print_attack_report(metrics)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import os
from datetime import datetime


def calculate_perturbation_metrics(X_original, X_adversarial):
    """
    Calcola metriche di perturbazione tra esempi originali e adversarial.
    
    SPIEGAZIONE METRICHE:
    
    1. L0 (Norma zero):
       - Conta quante feature sono state modificate
       - Esempio: Se solo 3 feature su 128 cambiano, L0 = 3
       - Importante per: Valutare concentrazione dell'attacco
    
    2. L2 (Norma euclidea):
       - Distanza geometrica tra vettori
       - Formula: sqrt(sum((x_adv - x_orig)^2))
       - Importante per: Misurare "magnitudine totale" della perturbazione
    
    3. L-inf (Norma infinito):
       - Massimo cambiamento su una singola feature
       - Formula: max(|x_adv - x_orig|)
       - Importante per: Garantire che nessuna feature cambi troppo
    
    Args:
        X_original: Esempi originali (numpy array, shape: [N, features])
        X_adversarial: Esempi adversarial (numpy array, shape: [N, features])
        
    Returns:
        Dictionary con metriche:
            - l0_mean: Media feature modificate per campione
            - l2_mean: Media norma L2 per campione
            - linf_mean: Media norma L-inf per campione
            - l0_std, l2_std, linf_std: Deviazioni standard
            
    Example:
        >>> metrics = calculate_perturbation_metrics(X_orig, X_adv)
        >>> print(f"L2 medio: {metrics['l2_mean']:.6f}")
        >>> print(f"Feature modificate: {metrics['l0_mean']:.2f}")
    """
    # Calcola perturbazioni (delta = x_adv - x_orig)
    perturbations = X_adversarial - X_original
    
    # L0: Numero di feature modificate (per ogni campione)
    l0_norms = np.count_nonzero(perturbations, axis=1)
    
    # L2: Norma euclidea (per ogni campione)
    l2_norms = np.linalg.norm(perturbations, ord=2, axis=1)
    
    # L-inf: Massimo cambiamento su singola feature (per ogni campione)
    linf_norms = np.max(np.abs(perturbations), axis=1)
    
    return {
        'l0_mean': float(np.mean(l0_norms)),
        'l0_std': float(np.std(l0_norms)),
        'l0_min': float(np.min(l0_norms)),
        'l0_max': float(np.max(l0_norms)),
        
        'l2_mean': float(np.mean(l2_norms)),
        'l2_std': float(np.std(l2_norms)),
        'l2_min': float(np.min(l2_norms)),
        'l2_max': float(np.max(l2_norms)),
        
        'linf_mean': float(np.mean(linf_norms)),
        'linf_std': float(np.std(linf_norms)),
        'linf_min': float(np.min(linf_norms)),
        'linf_max': float(np.max(linf_norms)),
    }


def calculate_attack_success_rate(y_original, predictions_original, predictions_adversarial, target_class=0):
    """
    Calcola l'Attack Success Rate (ASR).
    
    SPIEGAZIONE ASR:
    ASR misura la percentuale di attacchi che hanno raggiunto l'obiettivo.
    
    Per attacco di EVASION (Attack → Natural):
    - Successo = Campione originariamente classificato come "Attack" (1)
                 ora classificato come "Natural" (0)
    
    Formula: ASR = (N. campioni evasi) / (N. campioni attacco totali)
    
    Args:
        y_original: Etichette vere originali
        predictions_original: Predizioni sul dataset originale
        predictions_adversarial: Predizioni sul dataset adversarial
        target_class: Classe target per evasion (default: 0 = Natural)
        
    Returns:
        Dictionary con metriche ASR:
            - asr: Attack Success Rate (%)
            - successful_evasions: Numero assoluto di evasioni riuscite
            - total_attacks: Numero totale di campioni di attacco
            - accuracy_original: Accuracy sul dataset originale
            - accuracy_adversarial: Accuracy sul dataset adversarial
            
    Example:
        >>> asr_metrics = calculate_attack_success_rate(y_test, pred_orig, pred_adv)
        >>> print(f"ASR: {asr_metrics['asr']*100:.2f}%")
    """
    # Conta campioni di attacco originali (classe 1)
    attack_mask = (y_original == 1)
    total_attacks = np.sum(attack_mask)
    
    # Conta quanti campioni di attacco sono stati classificati come natural dopo perturbazione
    # Successo = (originale: Attack) AND (adversarial: Natural)
    successful_evasions = np.sum(
        (predictions_original[attack_mask] == 1) &  # Originariamente classificato come Attack
        (predictions_adversarial[attack_mask] == target_class)  # Ora classificato come Natural
    )
    
    # Calcola ASR
    asr = successful_evasions / total_attacks if total_attacks > 0 else 0.0
    
    # Accuracy sui dataset originale e adversarial
    accuracy_original = accuracy_score(y_original, predictions_original)
    accuracy_adversarial = accuracy_score(y_original, predictions_adversarial)
    
    return {
        'asr': float(asr),
        'successful_evasions': int(successful_evasions),
        'total_attacks': int(total_attacks),
        'accuracy_original': float(accuracy_original),
        'accuracy_adversarial': float(accuracy_adversarial),
        'accuracy_drop': float(accuracy_original - accuracy_adversarial)
    }


def evaluate_attack(model, X_original, y_original, X_adversarial, attack_name="Adversarial Attack"):
    """
    Valutazione completa di un attacco adversarial.
    
    Calcola:
    1. Attack Success Rate (ASR)
    2. Metriche di perturbazione (L0, L2, L-inf)
    3. Accuracy before/after attack
    4. Confusion matrix e classification report
    
    Args:
        model: Modello Random Forest target
        X_original: Dataset originale
        y_original: Etichette vere
        X_adversarial: Dataset adversarial perturbato
        attack_name: Nome dell'attacco (per logging)
        
    Returns:
        Dictionary completo con tutte le metriche
        
    Example:
        >>> metrics = evaluate_attack(model, X_test, y_test, X_adv, "WhiteBox_DecisionTree")
        >>> print(f"ASR: {metrics['asr']*100:.2f}%")
        >>> print(f"L2 medio: {metrics['l2_mean']:.6f}")
    """
    print(f"\n{'='*80}")
    print(f"VALUTAZIONE ATTACCO: {attack_name}")
    print(f"{'='*80}")
    
    # Predizioni
    predictions_original = model.predict(X_original)
    predictions_adversarial = model.predict(X_adversarial)
    
    # ASR e metriche successo
    asr_metrics = calculate_attack_success_rate(y_original, predictions_original, predictions_adversarial)
    
    # Metriche perturbazione
    perturbation_metrics = calculate_perturbation_metrics(X_original, X_adversarial)
    
    # Classification report adversarial
    report_adv = classification_report(
        y_original, 
        predictions_adversarial,
        target_names=["Natural", "Attack"],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix adversarial
    conf_matrix_adv = confusion_matrix(y_original, predictions_adversarial)
    
    # Combina tutte le metriche
    metrics = {
        'attack_name': attack_name,
        'n_samples': len(X_original),
        
        # ASR metriche
        **asr_metrics,
        
        # Perturbazione metriche
        **perturbation_metrics,
        
        # Classification metrics adversarial
        'precision_adv': float(report_adv['weighted avg']['precision']),
        'recall_adv': float(report_adv['weighted avg']['recall']),
        'f1_adv': float(report_adv['weighted avg']['f1-score']),
        
        # Confusion matrix adversarial
        'tn_adv': int(conf_matrix_adv[0, 0]),
        'fp_adv': int(conf_matrix_adv[0, 1]),
        'fn_adv': int(conf_matrix_adv[1, 0]),
        'tp_adv': int(conf_matrix_adv[1, 1]),
    }
    
    return metrics


def print_attack_report(metrics):
    """
    Stampa report dettagliato delle metriche di attacco.
    
    Args:
        metrics: Dictionary con metriche (output di evaluate_attack)
    """
    print(f"\n{'='*80}")
    print(f"📊 REPORT ATTACCO: {metrics['attack_name']}")
    print(f"{'='*80}")
    
    print(f"\n🎯 ATTACK SUCCESS RATE (ASR):")
    print(f"  - ASR: {metrics['asr']*100:.2f}% ({metrics['successful_evasions']}/{metrics['total_attacks']} attacchi evasi)")
    print(f"  - Accuracy originale: {metrics['accuracy_original']*100:.2f}%")
    print(f"  - Accuracy adversarial: {metrics['accuracy_adversarial']*100:.2f}%")
    print(f"  - Drop accuracy: {metrics['accuracy_drop']*100:.2f}%")
    
    print(f"\n📏 METRICHE PERTURBAZIONE:")
    print(f"  - L0 (feature modificate): {metrics['l0_mean']:.2f} ± {metrics['l0_std']:.2f}")
    print(f"    Range: [{metrics['l0_min']:.0f}, {metrics['l0_max']:.0f}]")
    print(f"  - L2 (norma euclidea): {metrics['l2_mean']:.6f} ± {metrics['l2_std']:.6f}")
    print(f"    Range: [{metrics['l2_min']:.6f}, {metrics['l2_max']:.6f}]")
    print(f"  - L-inf (max per feature): {metrics['linf_mean']:.6f} ± {metrics['linf_std']:.6f}")
    print(f"    Range: [{metrics['linf_min']:.6f}, {metrics['linf_max']:.6f}]")
    
    print(f"\n📈 METRICHE CLASSIFICAZIONE ADVERSARIAL:")
    print(f"  - Precision: {metrics['precision_adv']:.4f}")
    print(f"  - Recall: {metrics['recall_adv']:.4f}")
    print(f"  - F1-Score: {metrics['f1_adv']:.4f}")
    
    print(f"\n🔢 CONFUSION MATRIX ADVERSARIAL:")
    print(f"  True Positive (TP):  {metrics['tp_adv']}")
    print(f"  False Positive (FP): {metrics['fp_adv']}")
    print(f"  False Negative (FN): {metrics['fn_adv']}")
    print(f"  True Negative (TN):  {metrics['tn_adv']}")
    
    print(f"\n{'='*80}\n")


def save_attack_results(metrics_list, X_original, X_adversarial_dict, epsilons_tested, save_dir="attacks/results"):
    """
    Salva i risultati degli attacchi in file di testo.
    
    Args:
        metrics_list: Lista di dizionari con metriche (uno per epsilon)
        X_original: Dataset originale
        X_adversarial_dict: Dictionary {epsilon: X_adv} con esempi adversarial per ogni epsilon
        epsilons_tested: Lista di epsilon testati
        save_dir: Directory dove salvare i risultati
        
    Returns:
        Path del file salvato
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(save_dir, f"whitebox_attack_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REPORT ATTACCO WHITE-BOX: DECISION TREE ATTACK\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"N. campioni testati: {metrics_list[0]['n_samples']}\n")
        f.write(f"Epsilon testati: {epsilons_tested}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RISULTATI PER EPSILON\n")
        f.write("="*80 + "\n\n")
        
        for metrics in metrics_list:
            f.write(f"EPSILON: {metrics['attack_name'].split('_')[-1]}\n")
            f.write("-"*40 + "\n")
            f.write(f"ASR: {metrics['asr']*100:.2f}%\n")
            f.write(f"Evasioni riuscite: {metrics['successful_evasions']}/{metrics['total_attacks']}\n")
            f.write(f"Accuracy drop: {metrics['accuracy_drop']*100:.2f}%\n")
            f.write(f"L0 medio: {metrics['l0_mean']:.2f}\n")
            f.write(f"L2 medio: {metrics['l2_mean']:.6f}\n")
            f.write(f"L-inf medio: {metrics['linf_mean']:.6f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("CONFRONTO EPSILON\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Epsilon':<12} {'ASR (%)':<10} {'L2 medio':<12} {'L0 medio':<10}\n")
        f.write("-"*50 + "\n")
        
        for metrics in metrics_list:
            eps = metrics['attack_name'].split('_')[-1]
            f.write(f"{eps:<12} {metrics['asr']*100:<10.2f} {metrics['l2_mean']:<12.6f} {metrics['l0_mean']:<10.2f}\n")
    
    print(f"[Evaluation] ✅ Report salvato: {report_path}")
    return report_path