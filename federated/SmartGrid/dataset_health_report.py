#!/usr/bin/env python3
"""
dataset_health_report.py

Produce:
 - federated/results/dataset_report.txt  (report testuale)
 - federated/results/plots/*.png         (grafici aggregati)
Analizza i file data{client_id}.csv in data/SmartGrid/.
Seleziona automaticamente le top-K feature (per varianza globale) e crea boxplot + altri grafici.

Modifica: migliorata la funzione di heatmap delle feature quasi-costanti (plot_heatmap_near_constant).
Ora ordina le feature per percentuale di client in cui risultano near-constant,
limita il numero di feature mostrato per leggibilità e usa seaborn se disponibile.

Ulteriore modifica richiesta: nei grafici l'etichetta "Client ID" è stata sostituita con "File ID",
e i grafici non mostrano più il nome/titolo del grafico nelle immagini.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import math

# seaborn è opzionale ma migliora l'estetica delle heatmap
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

warnings_enabled = True

# --- Configurazione ---
DATA_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "SmartGrid")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_DatasetReport")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CLIENT_RANGE_DEFAULT = (1, 16)  # python range semantics: 1..15
TOP_K_FEATURES = 8  # numero di feature principali da plottare (modificabile)
HEATMAP_MAX_FEATURES = 40  # massimo feature da mostrare nella heatmap per leggibilità

# --- Utilità ---
def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️  Errore caricamento {path}: {e}")
        return None

def compute_iqr_bounds(arr, k=1.5):
    """Restituisce lower, upper vettoriali basati su IQR per colonne."""
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def is_near_constant(col_values, threshold_ratio=0.999, var_threshold=1e-12):
    """Restituisce True se la colonna è quasi-costante (moda occorre >= threshold_ratio)."""
    n = len(col_values)
    if n == 0:
        return True
    vals = col_values[~pd.isna(col_values)]
    if vals.size == 0:
        return True
    vals_u, counts = np.unique(vals, return_counts=True)
    if vals_u.size == 0:
        return True
    max_count = counts.max()
    ratio = max_count / float(n)
    var = float(np.nanvar(col_values))
    return (ratio >= threshold_ratio) or (var < var_threshold)

# --- Analisi per client ---
def analyze_client(file_path, client_id, iqr_k=1.5, near_const_ratio=0.999):
    df = safe_read_csv(file_path)
    if df is None:
        return None

    info = {}
    info['client_id'] = client_id
    info['n_samples'] = len(df)
    # marker -> label
    if 'marker' not in df.columns:
        print(f"⚠️  client {client_id} file {file_path} non contiene colonna 'marker'")
        df['marker'] = np.nan

    y = (df['marker'] != "Natural").astype(int)
    info['attack_count'] = int(y.sum())
    info['natural_count'] = int((y == 0).sum())
    info['attack_ratio'] = float(y.mean() if len(y) > 0 else 0.0)

    # Features numeriche
    X = df.drop(columns=['marker'], errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Se non ci sono colonne numeriche, converti all numeric forzato
    if len(numeric_cols) == 0:
        X = X.apply(pd.to_numeric, errors='coerce')
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_num = X[numeric_cols].copy()

    # Missing / Inf
    nan_counts = X_num.isna().sum()
    inf_mask = np.isinf(X_num.values)
    inf_counts = np.sum(inf_mask, axis=0) if X_num.shape[1] > 0 else np.array([])

    info['n_features'] = X_num.shape[1]
    info['feature_names'] = numeric_cols
    info['missing_per_feature'] = nan_counts.to_dict()
    info['inf_per_feature'] = dict(zip(numeric_cols, inf_counts.tolist() if len(inf_counts)>0 else []))

    # Statistiche per feature
    desc = X_num.describe().to_dict()
    info['describe'] = desc

    # Feature quasi-costanti
    near_const = {}
    for col in numeric_cols:
        near_const[col] = bool(is_near_constant(X_num[col].values, threshold_ratio=near_const_ratio))
    info['near_constant'] = near_const
    info['n_near_constant'] = int(sum(1 for v in near_const.values() if v))

    # IQR/outlier per feature
    if X_num.shape[1] > 0 and X_num.shape[0] > 0:
        lower, upper = compute_iqr_bounds(X_num.values, k=iqr_k)
        outlier_counts = {}
        for idx, col in enumerate(numeric_cols):
            col_vals = X_num.iloc[:, idx].values
            mask_out = (~np.isnan(col_vals)) & ((col_vals < lower[idx]) | (col_vals > upper[idx]))
            outlier_counts[col] = int(np.sum(mask_out))
        info['outlier_counts'] = outlier_counts
    else:
        info['outlier_counts'] = {}

    return info

# --- Aggregazione per selezione feature principali ---
def select_top_k_features(all_clients_info, k=TOP_K_FEATURES):
    # concatena dati per calcolare varianza globale (su colonne comuni)
    frames = []
    for info in all_clients_info:
        client_id = info['client_id']
        file_path = info.get('file_path')
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            frames.append(df.drop(columns=['marker'], errors='ignore'))
    if not frames:
        return []
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # forza numeriche
    merged = merged.apply(pd.to_numeric, errors='coerce')
    numeric = merged.select_dtypes(include=[np.number])
    # varianza per colonna
    variances = numeric.var(axis=0, skipna=True)
    variances = variances.dropna()
    if variances.empty:
        return numeric.columns.tolist()[:k]
    top_k = variances.sort_values(ascending=False).head(k).index.tolist()
    return top_k

# --- Report testuale e grafici ---
def write_text_report(all_clients_info, report_path):
    with open(report_path, "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"DATASET HEALTH REPORT\nGenerated: {now}\n\n")
        f.write(f"Clients analyzed: {len(all_clients_info)}\n\n")
        for info in all_clients_info:
            f.write(f"--- Client {info['client_id']} ---\n")
            f.write(f"Samples: {info.get('n_samples', 'N/A')}\n")
            f.write(f"Attack / Natural: {info.get('attack_count',0)} / {info.get('natural_count',0)} (attack_ratio={info.get('attack_ratio',0):.3f})\n")
            f.write(f"Numeric features: {info.get('n_features',0)}\n")
            f.write(f"Near-constant features: {info.get('n_near_constant',0)}\n")
            # top 5 missing features
            missing = info.get('missing_per_feature', {})
            if missing:
                sorted_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:5]
                f.write("Top missing (feature: count): " + ", ".join([f"{k}:{v}" for k,v in sorted_missing]) + "\n")
            # top 5 inf features
            infs = info.get('inf_per_feature', {})
            if infs:
                sorted_infs = sorted(infs.items(), key=lambda x: x[1], reverse=True)[:5]
                f.write("Top inf (feature: count): " + ", ".join([f"{k}:{v}" for k,v in sorted_infs]) + "\n")
            # outliers
            outliers = info.get('outlier_counts', {})
            if outliers:
                top_out = sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:5]
                f.write("Top outliers (feature: count): " + ", ".join([f"{k}:{v}" for k,v in top_out]) + "\n")
            # simple stats summary
            desc = info.get('describe', {})
            if desc:
                # print a few feature stats (first 3 features)
                cols = list(desc.keys())[:3]
                for col in cols:
                    colstats = desc[col]
                    f.write(f"Feature '{col}': mean={colstats.get('mean',np.nan):.4f}, std={colstats.get('std',np.nan):.4f}, min={colstats.get('min',np.nan):.4f}, max={colstats.get('max',np.nan):.4f}\n")
            f.write("\n")
    print(f"✅ Report testuale salvato in: {report_path}")

def plot_class_distribution(all_clients_info, out_path):
    """
    Plot a barre della distribuzione delle classi per file/client.
    - Sostituita etichetta asse X: 'File ID' (invece di 'Client ID')
    - Rimosso titolo del grafico
    """
    client_ids = [info['client_id'] for info in all_clients_info]
    attack_perc = [info.get('attack_ratio',0)*100 for info in all_clients_info]
    natural_perc = [100 - a for a in attack_perc]

    x = np.arange(len(client_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, attack_perc, width, label='Attack (%)')
    ax.bar(x + width/2, natural_perc, width, label='Natural (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(cid) for cid in client_ids])
    # Etichetta richiesta: File ID
    ax.set_xlabel("File ID")
    ax.set_ylabel("Percentage (%)")
    # Rimosso il titolo del grafico come richiesto
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Grafico distribuzione classi salvato in: {out_path}")

def plot_boxplots_top_features(all_clients_info, top_features, out_path):
    """
    Boxplot per le top features (una riga per feature, più box per file).
    - L'etichetta dell'asse X è 'File ID'
    - I titoli dei singoli subplot (nomi feature) sono rimossi per rispettare la richiesta
      (le feature restano comunque selezionate dai top_features)
    """
    if not top_features:
        print("⚠️ Nessuna feature per boxplot")
        return
    n = len(top_features)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(-1)
    for i, feat in enumerate(top_features):
        ax = axes[i]
        # raccogli valori per client
        data_per_client = []
        labels = []
        for info in all_clients_info:
            file_path = info.get('file_path')
            if not file_path or not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            if feat in df.columns:
                arr = pd.to_numeric(df[feat], errors='coerce').dropna().values
            else:
                arr = np.array([])
            data_per_client.append(arr)
            labels.append(str(info['client_id']))
        # draw boxplot grouped: one box per file for this feature
        ax.boxplot(data_per_client, notch=False, patch_artist=False)
        # Rimosso titolo del subplot (nome feature)
        # ax.set_title(feat)  <-- intentionally removed
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("File ID")  # sostituisce "Client ID"
        ax.grid(alpha=0.2)
    # hide extra axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Boxplots top features salvati in: {out_path}")

def plot_value_ranges(all_clients_info, top_features, out_path):
    """
    Mostra min/median/max per file per feature in un plot per feature.
    - Asse X etichettato 'File ID'
    - Rimosso il titolo dei singoli plot contenente il nome della feature
    """
    if not top_features:
        return
    fig, axes = plt.subplots(len(top_features), 1, figsize=(10, 3*len(top_features)))
    if len(top_features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_features):
        mins, medians, maxs, cid = [], [], [], []
        for info in all_clients_info:
            file_path = info.get('file_path')
            if not file_path or not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            if feat in df.columns:
                col = pd.to_numeric(df[feat], errors='coerce').dropna()
                if len(col)==0:
                    continue
                mins.append(col.min()); medians.append(col.median()); maxs.append(col.max()); cid.append(info['client_id'])
        if not cid:
            continue
        ax.errorbar(cid, medians, yerr=[np.array(medians)-np.array(mins), np.array(maxs)-np.array(medians)], fmt='o', capsize=5)
        # Rimosso titolo del grafico (il nome della feature)
        # ax.set_title(f"Range & median per client - {feat}")  <-- intentionally removed
        ax.set_xlabel("File ID")  # sostituisce "Client ID"
        ax.set_ylabel(feat)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Value ranges salvati in: {out_path}")

def plot_heatmap_near_constant(all_clients_info, out_path):
    """
    Crea una heatmap (file x feature) con valori:
      1 = feature near-constant per quel file
      0 = feature non near-constant o assente

    Miglioramenti rispetto alla versione precedente:
    - Ordina le feature per percentuale di file in cui sono near-constant (descending)
    - Limita il numero di feature mostrate a HEATMAP_MAX_FEATURES per leggibilità
    - Usa seaborn se disponibile (migliore resa grafica)
    - Stampa un sommario delle feature più frequentemente near-constant

    Modifiche richieste:
    - L'asse y è etichettato 'File ID' (invece di 'Client ID')
    - Non viene mostrato il titolo del grafico nelle immagini
    """
    # Unione delle feature disponibili
    feature_set = set()
    for info in all_clients_info:
        feature_set.update(info.get('feature_names', []))
    if not feature_set:
        print("⚠️ Nessuna feature disponibile per heatmap")
        return

    # Costruisci una tabella feature -> count di file in cui è near-constant
    feature_list_all = sorted(list(feature_set))
    feature_counts = {feat: 0 for feat in feature_list_all}
    client_id_list = [info['client_id'] for info in all_clients_info]

    for info in all_clients_info:
        nc = info.get('near_constant', {})
        for feat in feature_list_all:
            if nc.get(feat, False):
                feature_counts[feat] += 1

    # Calcola percentuale di file per feature
    n_clients = len(all_clients_info)
    feature_percent = {feat: (feature_counts[feat] / n_clients) * 100.0 for feat in feature_list_all}

    # Ordina feature per percentuale descending (le più problematiche prime)
    feature_list_sorted = sorted(feature_percent.keys(), key=lambda x: feature_percent[x], reverse=True)

    # Limita a HEATMAP_MAX_FEATURES per leggibilità
    feature_list = feature_list_sorted[:HEATMAP_MAX_FEATURES]

    # Costruisci la matrice file x feature
    mat = np.zeros((n_clients, len(feature_list)), dtype=int)
    for i, info in enumerate(all_clients_info):
        nc = info.get('near_constant', {})
        for j, feat in enumerate(feature_list):
            if nc.get(feat, False):
                mat[i, j] = 1

    # Stampa sommario features top
    percent_list = [feature_percent[feat] for feat in feature_list]
    print("\nTop features più spesso near-constant (percentuale di file):")
    for feat, pct in zip(feature_list, percent_list):
        print(f"  - {feat}: {pct:.1f}%")

    # Plot (nota: titolo rimosso)
    figsize = (max(10, len(feature_list) * 0.35), max(6, n_clients * 0.35))
    plt.figure(figsize=figsize)
    if _HAS_SEABORN:
        sns.set(style="whitegrid")
        ax = sns.heatmap(mat, cmap="Greys", cbar=True,
                         xticklabels=feature_list, yticklabels=client_id_list,
                         linewidths=0.4, linecolor='lightgray', vmin=0, vmax=1)
        ax.set_xticklabels(feature_list, rotation=90, fontsize=7)
        ax.set_yticklabels([str(cid) for cid in client_id_list], fontsize=9)
    else:
        ax = plt.gca()
        im = ax.imshow(mat, cmap="Greys", aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_xticks(np.arange(len(feature_list)))
        ax.set_xticklabels(feature_list, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(n_clients))
        ax.set_yticklabels([str(cid) for cid in client_id_list], fontsize=9)

    # Etichette richieste: File ID al posto di Client ID; non impostiamo titolo
    ax.set_xlabel("Feature")
    ax.set_ylabel("File ID")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Heatmap near-constant features salvata in: {out_path}")

# --- Main orchestration ---
def main(data_dir, client_range=(1,16), top_k=TOP_K_FEATURES):
    ensure_dirs()

    client_infos = []
    for client_id in range(client_range[0], client_range[1]):
        file_path = os.path.join(data_dir, f"data{client_id}.csv")
        if not os.path.exists(file_path):
            print(f"⚠️ File non trovato: {file_path} -> skip")
            continue
        info = analyze_client(file_path, client_id)
        if info is None:
            continue
        info['file_path'] = file_path
        client_infos.append(info)

    if not client_infos:
        print("❌ Nessun client analizzato. Controlla il percorso dei dati.")
        return

    # seleziona feature principali
    top_features = select_top_k_features(client_infos, k=top_k)
    print(f"Top {len(top_features)} features selezionate: {top_features}")

    # report + plots
    report_path = os.path.join(RESULTS_DIR, "dataset_report.txt")
    write_text_report(client_infos, report_path)

    plot_class_distribution(client_infos, os.path.join(PLOTS_DIR, "class_distribution.png"))
    plot_boxplots_top_features(client_infos, top_features, os.path.join(PLOTS_DIR, "boxplots_selected_features.png"))
    plot_value_ranges(client_infos, top_features, os.path.join(PLOTS_DIR, "value_ranges.png"))
    plot_heatmap_near_constant(client_infos, os.path.join(PLOTS_DIR, "heatmap_constant_features.png"))

    print("\n✅ Analisi completata.")
    print(f"Report: {report_path}")
    print(f"Plots: {PLOTS_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset health check - SmartGrid (principal features only)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR_DEFAULT, help="Path to data/SmartGrid directory")
    parser.add_argument("--start", type=int, default=CLIENT_RANGE_DEFAULT[0], help="First client id (inclusive)")
    parser.add_argument("--end", type=int, default=CLIENT_RANGE_DEFAULT[1], help="End client id (exclusive)")
    parser.add_argument("--top-k", type=int, default=TOP_K_FEATURES, help="Top K features to visualize")
    args = parser.parse_args()
    main(args.data_dir, client_range=(args.start, args.end), top_k=args.top_k)