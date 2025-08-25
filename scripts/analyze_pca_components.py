import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')

def clip_outliers_iqr(X, k=5.0):
    """
    Clippa gli outlier per ogni feature usando la regola dei quantili (IQR).
    Limiti: [Q1 - k*IQR, Q3 + k*IQR] (default k=5.0).
    """
    X_clipped = X.copy()
    for col in range(X_clipped.shape[1]):
        col_data = X_clipped[:, col]
        q1 = np.nanpercentile(col_data, 25)
        q3 = np.nanpercentile(col_data, 75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        X_clipped[:, col] = np.clip(col_data, lower, upper)
    return X_clipped

def remove_near_constant_features(X, threshold_var=1e-12, threshold_ratio=0.999):
    """
    Rimuove le feature che sono costanti almeno al 99.9% (tutte uguali tranne lo 0.1%).
    """
    keep_mask = []
    n = X.shape[0]
    for col in range(X.shape[1]):
        col_data = X[:, col]
        # Conta la moda (valore più frequente)
        vals, counts = np.unique(col_data, return_counts=True)
        max_count = np.max(counts)
        ratio = max_count / n
        var = np.nanvar(col_data)
        # Tiene solo se NON è costante al 99.9% e varianza > threshold_var
        keep = not (ratio >= threshold_ratio or var < threshold_var)
        keep_mask.append(keep)
    keep_mask = np.array(keep_mask)
    return X[:, keep_mask], keep_mask


def clean_data_for_pca_analysis(X):
    """
    Pulizia dati: solo sostituzione inf/-inf con NaN.
    """
    if hasattr(X, 'values'):
        X_array = X.values.copy()
    else:
        X_array = X.copy()
    X_array = np.where(np.isinf(X_array), np.nan, X_array)
    return X_array

def varianza_spiegata_per_classe(pca, X_scaled, y):
    """
    Calcola la varianza spiegata dalla proiezione PCA separatamente su attacchi e naturali.
    La varianza spiegata è la varianza totale delle proiezioni rispetto al totale della classe.
    """
    results = {}
    for label, name in [(1, "attack"), (0, "natural")]:
        mask = (y == label)
        if np.sum(mask) > 1:
            projected = pca.transform(X_scaled[mask])
            var_proj = np.var(projected, axis=0, ddof=1)
            total_var = np.sum(var_proj)
            results[name] = {
                "samples": np.sum(mask),
                "var_proj": var_proj,
                "total_var_proj": total_var
            }
        else:
            results[name] = {
                "samples": np.sum(mask),
                "var_proj": None,
                "total_var_proj": None
            }
    return results

def analyze_single_client_pca(client_id, data_dir):
    """
    Analizza la PCA per un singolo client.
    Preprocessing: pulizia inf/NaN, clipping IQR, imputazione mediana, rimozione quasi-costanti, scaling.
    Calcola anche la varianza post-PCA separata per attacchi e naturali.
    """
    file_path = os.path.join(data_dir, f"data{client_id}.csv")
    if not os.path.exists(file_path):
        print(f"File data{client_id}.csv non trovato")
        return None
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=["marker"])
        y = (df["marker"] != "Natural").astype(int)
        print(f"\nAnalisi Client {client_id}:")
        print(f"  Campioni: {len(df)}")
        print(f"  Feature originali: {X.shape[1]}")
        print(f"  Distribuzione attacchi: {y.mean()*100:.1f}%")

        # Pulizia inf/NaN
        X_cleaned = clean_data_for_pca_analysis(X)
        X_np = np.array(X_cleaned, dtype=float)

        # Clipping outlier per quantili
        X_clipped = clip_outliers_iqr(X_np, k=5.0)

        # Imputazione mediana
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_clipped)

        # Rimozione feature quasi-costanti
        X_reduced, keep_mask = remove_near_constant_features(X_imputed, threshold_var=1e-12, threshold_ratio=0.999)

        # Scaling standard
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)

        # PCA completa
        pca_full = PCA()
        pca_full.fit(X_scaled)

        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
        eigenvalues = pca_full.explained_variance_
        n_components_kaiser = np.sum(eigenvalues > 1.0)
        n_samples = len(X_scaled)
        max_practical = min(int(np.sqrt(n_samples)), int(n_samples * 0.5))

        # Analisi varianza post-PCA per classi
        varianza_classi = varianza_spiegata_per_classe(pca_full, X_scaled, y)
        attack_var = varianza_classi["attack"]["total_var_proj"]
        natural_var = varianza_classi["natural"]["total_var_proj"]

        results = {
            'client_id': client_id,
            'total_samples': len(df),
            'original_features': X.shape[1],
            'attack_ratio': y.mean(),
            'n_components_90': min(n_components_90, X.shape[1]),
            'n_components_95': min(n_components_95, X.shape[1]),
            'n_components_99': min(n_components_99, X.shape[1]),
            'n_components_kaiser': min(n_components_kaiser, X.shape[1]),
            'max_practical': min(max_practical, X.shape[1]),
            'variance_explained_90': cumulative_variance[min(n_components_90-1, len(cumulative_variance)-1)] if n_components_90 > 0 else 0,
            'variance_explained_95': cumulative_variance[min(n_components_95-1, len(cumulative_variance)-1)] if n_components_95 > 0 else 0,
            'variance_explained_99': cumulative_variance[min(n_components_99-1, len(cumulative_variance)-1)] if n_components_99 > 0 else 0,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'eigenvalues': eigenvalues,
            'attack_var_post_pca': attack_var,
            'natural_var_post_pca': natural_var,
            'attack_samples': varianza_classi["attack"]["samples"],
            'natural_samples': varianza_classi["natural"]["samples"]
        }

        print(f"  Componenti per 90% varianza: {results['n_components_90']}")
        print(f"  Componenti per 95% varianza: {results['n_components_95']}")
        print(f"  Componenti per 99% varianza: {results['n_components_99']}")
        print(f"  Kaiser criterion (λ>1): {results['n_components_kaiser']}")
        print(f"  Limite pratico: {results['max_practical']}")
        print(f"  Varianza totale post-PCA (attacchi): {attack_var:.2f} su {results['attack_samples']} campioni")
        print(f"  Varianza totale post-PCA (naturali): {natural_var:.2f} su {results['natural_samples']} campioni")

        return results

    except Exception as e:
        print(f"Errore analisi client {client_id}: {e}")
        return None

def analyze_all_clients_pca(data_dir, client_range=(1, 16)):
    """
    Analizza la PCA per tutti i client disponibili.
    
    Args:
        data_dir: Directory contenente i dati
        client_range: Range di client da analizzare (default: 1-15)
    
    Returns:
        Lista con risultati analisi per ogni client
    """
    print("=" * 60)
    print("Determinazione numero ottimale di componenti PCA")
    print("=" * 60)
    
    all_results = []
    
    for client_id in range(client_range[0], client_range[1]):
        result = analyze_single_client_pca(client_id, data_dir)
        if result is not None:
            all_results.append(result)
    
    return all_results

def summarize_pca_analysis(all_results):
    """
    Riassume i risultati dell'analisi PCA e raccomanda numero ottimale di componenti.
    
    Args:
        all_results: Lista con risultati analisi PCA
    
    Returns:
        Dizionario con raccomandazioni
    """
    if not all_results:
        print("Nessun risultato disponibile per l'analisi")
        return None
    
    print("\n" + "=" * 60)
    print("RIASSUNTO ANALISI PCA")
    print("=" * 60)
    
    # Estrai statistiche
    n_components_90_list = [r['n_components_90'] for r in all_results]
    n_components_95_list = [r['n_components_95'] for r in all_results]
    n_components_99_list = [r['n_components_99'] for r in all_results]
    kaiser_list = [r['n_components_kaiser'] for r in all_results]
    practical_list = [r['max_practical'] for r in all_results]
    original_features_list = [r['original_features'] for r in all_results]
    
    # Calcola statistiche aggregate
    stats = {
        'num_clients': len(all_results),
        'original_features_mean': np.mean(original_features_list),
        'original_features_std': np.std(original_features_list),
        'original_features_min': np.min(original_features_list),
        'original_features_max': np.max(original_features_list),
        
        'n_components_90_mean': np.mean(n_components_90_list),
        'n_components_90_std': np.std(n_components_90_list),
        'n_components_90_min': np.min(n_components_90_list),
        'n_components_90_max': np.max(n_components_90_list),
        
        'n_components_95_mean': np.mean(n_components_95_list),
        'n_components_95_std': np.std(n_components_95_list),
        'n_components_95_min': np.min(n_components_95_list),
        'n_components_95_max': np.max(n_components_95_list),
        
        'n_components_99_mean': np.mean(n_components_99_list),
        'n_components_99_std': np.std(n_components_99_list),
        'n_components_99_min': np.min(n_components_99_list),
        'n_components_99_max': np.max(n_components_99_list),
        
        'kaiser_mean': np.mean(kaiser_list),
        'kaiser_std': np.std(kaiser_list),
        'kaiser_min': np.min(kaiser_list),
        'kaiser_max': np.max(kaiser_list),
        
        'practical_mean': np.mean(practical_list),
        'practical_std': np.std(practical_list),
        'practical_min': np.min(practical_list),
        'practical_max': np.max(practical_list)
    }
    
    print(f"Client analizzati: {stats['num_clients']}")
    print(f"Feature originali: {stats['original_features_mean']:.1f} ± {stats['original_features_std']:.1f} (range: {stats['original_features_min']}-{stats['original_features_max']})")
    print()
    
    print("COMPONENTI PCA PER SOGLIA DI VARIANZA:")
    print(f"  90% varianza: {stats['n_components_90_mean']:.1f} ± {stats['n_components_90_std']:.1f} (range: {stats['n_components_90_min']}-{stats['n_components_90_max']})")
    print(f"  95% varianza: {stats['n_components_95_mean']:.1f} ± {stats['n_components_95_std']:.1f} (range: {stats['n_components_95_min']}-{stats['n_components_95_max']})")
    print(f"  99% varianza: {stats['n_components_99_mean']:.1f} ± {stats['n_components_99_std']:.1f} (range: {stats['n_components_99_min']}-{stats['n_components_99_max']})")
    print()
    
    print("ALTRI CRITERI:")
    print(f"  Kaiser (λ>1): {stats['kaiser_mean']:.1f} ± {stats['kaiser_std']:.1f} (range: {stats['kaiser_min']}-{stats['kaiser_max']})")
    print(f"  Limite pratico: {stats['practical_mean']:.1f} ± {stats['practical_std']:.1f} (range: {stats['practical_min']}-{stats['practical_max']})")
    print()
    
    # Componenti consigliate
    print("COMPONENTI CONSIGLIATE:")
    
    # Approccio conservativo (95% varianza)
    recommended_95 = int(np.ceil(stats['n_components_95_mean']))
    recommended_95_safe = min(recommended_95, int(stats['practical_mean']))
    
    # Approccio bilanciato (90% varianza)
    recommended_90 = int(np.ceil(stats['n_components_90_mean']))
    recommended_90_safe = min(recommended_90, int(stats['practical_mean']))
    
    # Raccomandazione Kaiser
    recommended_kaiser = int(np.ceil(stats['kaiser_mean']))
    recommended_kaiser_safe = min(recommended_kaiser, int(stats['practical_mean']))
    
    print(f"  Conservativa (95% varianza): {recommended_95_safe} componenti")
    print(f"  Bilanciata (90% varianza): {recommended_90_safe} componenti")
    print(f"  Kaiser criterion: {recommended_kaiser_safe} componenti")
    
    # Scelta finale basata su compromessi
    if recommended_95_safe <= 50 and stats['n_components_95_std'] < 10:
        final_recommendation = recommended_95_safe
        justification = f"95% varianza con bassa variabilità tra client (σ={stats['n_components_95_std']:.1f})"
    elif recommended_90_safe <= 40:
        final_recommendation = recommended_90_safe
        justification = f"90% varianza per efficienza computazionale"
    else:
        final_recommendation = min(40, recommended_kaiser_safe)
        justification = f"Limite conservativo per evitare overfitting"
    
    print(f"\nRACCOMANDAZIONE FINALE: {final_recommendation} componenti PCA")
    print(f"Giustificazione: {justification}")
    
    # Dettaglio per client
    print(f"\nDETTAGLIO PER CLIENT:")
    print(f"{'Client':<8} {'Campioni':<10} {'Orig':<6} {'90%':<5} {'95%':<5} {'99%':<5} {'Kaiser':<7} {'Pratico':<8}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['client_id']:<8} {r['total_samples']:<10} {r['original_features']:<6} "
              f"{r['n_components_90']:<5} {r['n_components_95']:<5} {r['n_components_99']:<5} "
              f"{r['n_components_kaiser']:<7} {r['max_practical']:<8}")
    
    recommendation = {
        'recommended_components': final_recommendation,
        'justification': justification,
        'stats': stats,
        'method': 'comprehensive_analysis',
        'variance_threshold_used': 0.95 if final_recommendation == recommended_95_safe else 0.90,
        'all_results': all_results
    }
    
    return recommendation

def create_visualization(all_results, recommendation, output_dir):
    """
    Crea grafici di analisi PCA.
    
    Args:
        all_results: Lista con risultati analisi PCA
        recommendation: Dizionario con raccomandazioni
        output_dir: Directory di output
    """
    if not all_results:
        return
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisi PCA Dataset SmartGrid', fontsize=16)
        
        # Grafico 1: Distribuzione componenti per soglie varianza
        ax1 = axes[0, 0]
        client_ids = [r['client_id'] for r in all_results]
        ax1.plot(client_ids, [r['n_components_90'] for r in all_results], 'o-', label='90% varianza', alpha=0.7)
        ax1.plot(client_ids, [r['n_components_95'] for r in all_results], 's-', label='95% varianza', alpha=0.7)
        ax1.plot(client_ids, [r['n_components_99'] for r in all_results], '^-', label='99% varianza', alpha=0.7)
        ax1.axhline(y=recommendation['recommended_components'], color='red', linestyle='--', label=f'Raccomandato: {recommendation["recommended_components"]}')
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Numero Componenti PCA')
        ax1.set_title('Componenti PCA per Soglia Varianza')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Grafico 2: Confronto criteri
        ax2 = axes[0, 1]
        criteria_names = ['90% var', '95% var', '99% var', 'Kaiser', 'Pratico']
        criteria_means = [
            recommendation['stats']['n_components_90_mean'],
            recommendation['stats']['n_components_95_mean'],
            recommendation['stats']['n_components_99_mean'],
            recommendation['stats']['kaiser_mean'],
            recommendation['stats']['practical_mean']
        ]
        criteria_stds = [
            recommendation['stats']['n_components_90_std'],
            recommendation['stats']['n_components_95_std'],
            recommendation['stats']['n_components_99_std'],
            recommendation['stats']['kaiser_std'],
            recommendation['stats']['practical_std']
        ]
        
        bars = ax2.bar(criteria_names, criteria_means, yerr=criteria_stds, capsize=5, alpha=0.7)
        ax2.axhline(y=recommendation['recommended_components'], color='red', linestyle='--', label=f'Raccomandato: {recommendation["recommended_components"]}')
        ax2.set_ylabel('Numero Componenti PCA')
        ax2.set_title('Confronto Criteri PCA (Media ± Dev.Std)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Grafico 3: Distribuzione dimensioni dataset
        ax3 = axes[1, 0]
        samples = [r['total_samples'] for r in all_results]
        attack_ratios = [r['attack_ratio']*100 for r in all_results]
        scatter = ax3.scatter(samples, attack_ratios, c=client_ids, cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Numero Campioni')
        ax3.set_ylabel('Percentuale Attacchi (%)')
        ax3.set_title('Distribuzione Dataset per Client')
        plt.colorbar(scatter, ax=ax3, label='Client ID')
        ax3.grid(True, alpha=0.3)
        
        # Grafico 4: Efficienza dimensionale
        ax4 = axes[1, 1]
        original_features = [r['original_features'] for r in all_results]
        reduction_ratios = [(orig - recommendation['recommended_components']) / orig * 100 for orig in original_features]
        
        ax4.bar(client_ids, reduction_ratios, alpha=0.7, color='green')
        ax4.set_xlabel('Client ID')
        ax4.set_ylabel('Riduzione Dimensionalità (%)')
        ax4.set_title(f'Efficienza PCA (Riduzione con {recommendation["recommended_components"]} componenti)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva grafico
        plot_path = os.path.join(output_dir, 'pca_analysis_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Grafici salvati in: {plot_path}")
        
    except Exception as e:
        print(f"Errore nella creazione dei grafici: {e}")

def main():
    """
    Funzione principale per l'analisi PCA preliminare.
    """
    # Configurazione path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "SmartGrid")
    output_dir = os.path.join(project_root, "scripts", "results_pca")
    
    print("📊 ANALISI PCA PRELIMINARE PER DATASET SMART GRID")
    print("=" * 70)
    print(f"Directory dati: {data_dir}")
    print(f"Directory output: {output_dir}")
    
    # Verifica esistenza directory dati
    if not os.path.exists(data_dir):
        print(f"❌ ERRORE: Directory dati non trovata: {data_dir}")
        print("Assicurati che la directory data/SmartGrid/ contenga i file data1.csv - data15.csv")
        return
    
    # Analizza tutti i client
    all_results = analyze_all_clients_pca(data_dir, client_range=(1, 16))
    
    if not all_results:
        print("❌ ERRORE: Nessun client analizzato con successo")
        return
    
    # Riassumi risultati e genera raccomandazioni
    recommendation = summarize_pca_analysis(all_results)
    
    if recommendation is None:
        print("❌ ERRORE: Impossibile generare raccomandazioni")
        return
    
    # Crea visualizzazioni
    create_visualization(all_results, recommendation, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALISI PCA COMPLETATA")
    print("=" * 70)
    print(f"🎯 NUMERO RACCOMANDATO DI COMPONENTI PCA: {recommendation['recommended_components']}")
    print(f"📊 Giustificazione: {recommendation['justification']}")
    print(f"📁 Grafici salvati in: {output_dir}")
    print()
    
if __name__ == "__main__":
    main()