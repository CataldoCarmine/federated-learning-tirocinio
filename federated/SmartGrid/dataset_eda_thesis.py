#!/usr/bin/env python3
"""
dataset_eda_thesis.py

Exploratory Data Analysis (EDA) per il dataset SmartGrid.
Genera grafici professionali per documentazione e tesi universitaria.

ANALISI EFFETTUATE:
1.Distribuzione dei Dati (Non-IID) tra client
2.Sbilanciamento di Classe per client e globale
3.Qualità dei Dati (missing values, outliers, valori infiniti)
4.Caratteristiche delle Feature (distribuzioni, correlazioni)
5.Variabilità Inter-Client (eterogeneità federata)
6.Analisi Temporale/Strutturale del dataset

OUTPUT:
- federated/SmartGrid/results_EDA/plots/*.png (grafici alta risoluzione)
- federated/SmartGrid/results_EDA/eda_report.txt (report testuale)

UTILIZZO:
    python federated/SmartGrid/dataset_eda_thesis.py
    python federated/SmartGrid/dataset_eda_thesis.py --data-dir /path/to/data
    python federated/SmartGrid/dataset_eda_thesis.py --top-k 10 --dpi 300

AUTORE: Generato per tesi di laurea su Federated Learning
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import datetime
import argparse
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# --- Configurazione ---
DATA_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "SmartGrid")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_EDA")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CLIENT_RANGE_DEFAULT = (1, 16)  # Client 1-15
TOP_K_FEATURES = 8  # Numero di feature principali da visualizzare

# Configurazione stile grafici per tesi
plt.style.use('seaborn-v0_8-whitegrid')
COLORS_CLIENTS = plt.cm.tab20(np.linspace(0, 1, 15))  # Colori per 15 client
COLOR_ATTACK = '#E74C3C'  # Rosso per attacchi
COLOR_NATURAL = '#27AE60'  # Verde per naturali
FIGSIZE_LARGE = (14, 10)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_SMALL = (10, 6)
DPI_DEFAULT = 150


def ensure_dirs():
    """Crea le directory per i risultati se non esistono."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"[EDA] Directory risultati: {RESULTS_DIR}")
    print(f"[EDA] Directory grafici: {PLOTS_DIR}")


def safe_read_csv(path):
    """
    Carica un file CSV in modo sicuro.
    
    Args:
        path: Percorso del file CSV
        
    Returns:
        DataFrame pandas o None se errore
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Errore caricamento {path}: {e}")
        return None


def load_all_clients_data(data_dir, client_range):
    """
    Carica i dati di tutti i client.
    
    SPIEGAZIONE:
    Nel Federated Learning, ogni client ha il proprio dataset locale.
    Questa funzione carica tutti i dataset per analizzare:
    - La distribuzione dei dati tra client (Non-IID)
    - Le differenze nelle caratteristiche dei dati
    - La qualità complessiva del dataset
    
    Args:
        data_dir: Directory contenente i file data{id}.csv
        client_range: Tupla (start, end) per range client
        
    Returns:
        clients_data: Lista di dizionari con dati per client
        combined_df: DataFrame con tutti i dati combinati
    """
    print(f"\n{'='*60}")
    print("CARICAMENTO DATI CLIENT")
    print(f"{'='*60}")
    
    clients_data = []
    all_dfs = []
    
    for client_id in range(client_range[0], client_range[1]):
        file_path = os.path.join(data_dir, f"data{client_id}.csv")
        
        if not os.path.exists(file_path):
            print(f"⚠️ File non trovato: data{client_id}.csv")
            continue
        
        df = safe_read_csv(file_path)
        if df is None:
            continue
        
        # Aggiungi colonna client_id per tracking
        df['client_id'] = client_id
        
        # Estrai informazioni base
        if 'marker' in df.columns:
            y = (df['marker'] != "Natural").astype(int)
            attack_count = int(y.sum())
            natural_count = int((y == 0).sum())
        else:
            attack_count = 0
            natural_count = len(df)
            y = pd.Series([0] * len(df))
        
        client_info = {
            'client_id': client_id,
            'df': df,
            'n_samples': len(df),
            'attack_count': attack_count,
            'natural_count': natural_count,
            'attack_ratio': attack_count / len(df) if len(df) > 0 else 0,
            'n_features': df.shape[1] - 2  # Esclude 'marker' e 'client_id'
        }
        
        clients_data.append(client_info)
        all_dfs.append(df)
        
        print(f"✅ Client {client_id}: {len(df)} campioni, "
              f"{attack_count} attacchi ({client_info['attack_ratio']*100:.1f}%)")
    
    # Combina tutti i DataFrame
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n📊 Dataset combinato: {len(combined_df)} campioni totali")
    else:
        combined_df = pd.DataFrame()
        print("❌ Nessun dato caricato")
    
    return clients_data, combined_df


# =============================================================================
# SEZIONE 1: DISTRIBUZIONE DEI DATI (NON-IID)
# =============================================================================

def plot_samples_distribution(clients_data, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 1.1: Distribuzione del numero di campioni per client.
    NOTE: asse X etichettato 'File ID' e nessun titolo nei grafici.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)
    
    client_ids = [c['client_id'] for c in clients_data]
    n_samples = [c['n_samples'] for c in clients_data]
    
    # Sottografico 1: Bar chart
    ax1 = axes[0]
    bars = ax1.bar(client_ids, n_samples, color=COLORS_CLIENTS[:len(client_ids)], 
                   edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('File ID', fontsize=12)
    ax1.set_ylabel('Numero di Campioni', fontsize=12)
    # titolo rimosso
    ax1.set_xticks(client_ids)
    
    # Aggiungi etichette sui bar
    for bar, n in zip(bars, n_samples):
        ax1.annotate(f'{n}', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    # Linea media
    mean_samples = np.mean(n_samples)
    ax1.axhline(y=mean_samples, color='red', linestyle='--', linewidth=2, 
                label=f'Media: {mean_samples:.0f}')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Sottografico 2: Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(n_samples, 
                                        labels=[f'{cid}' for cid in client_ids],
                                        autopct='%1.1f%%',
                                        colors=COLORS_CLIENTS[:len(client_ids)],
                                        explode=[0.02]*len(client_ids),
                                        shadow=True)
    # titolo rimosso
    
    # Statistiche
    std_samples = np.std(n_samples)
    cv = std_samples / mean_samples * 100  # Coefficiente di variazione
    
    # suptitle rimosso (nessun titolo globale)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")
    
    return {'mean': mean_samples, 'std': std_samples, 'cv': cv}


def plot_feature_distribution_per_client(clients_data, top_k_features, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 1.2: Distribuzione delle feature principali per client.
    NOTE: rimosso titolo globale e titoli subplot; asse X etichettato 'File ID'.
    """
    # Seleziona le top-k feature per varianza globale
    all_dfs = [c['df'].drop(columns=['marker', 'client_id'], errors='ignore') for c in clients_data]
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_numeric = combined.select_dtypes(include=[np.number])
    
    # Calcola varianza per feature
    variances = combined_numeric.var().dropna().sort_values(ascending=False)
    top_features = variances.head(top_k_features).index.tolist()
    
    if not top_features:
        print("⚠️ Nessuna feature numerica trovata per violin plot")
        return
    
    # Crea figura con subplot per ogni feature
    n_features = len(top_features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Prepara dati per violin plot
        data_for_violin = []
        labels = []
        
        for c in clients_data:
            df = c['df']
            if feature in df.columns:
                values = pd.to_numeric(df[feature], errors='coerce').dropna().values
                if len(values) > 0:
                    data_for_violin.append(values)
                    labels.append(f"{c['client_id']}")
        
        if data_for_violin:
            parts = ax.violinplot(data_for_violin, positions=range(len(labels)), 
                                  showmeans=True, showmedians=True)
            
            # Colora i violin
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(COLORS_CLIENTS[i % len(COLORS_CLIENTS)])
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, fontsize=9)
            # titolo subplot rimosso
            ax.set_ylabel('Valore', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
    
    # Nascondi assi extra
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    # suptitle rimosso
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


# =============================================================================
# SEZIONE 2: SBILANCIAMENTO DI CLASSE
# =============================================================================

def plot_class_imbalance(clients_data, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 2.1: Sbilanciamento di classe per client e globale.
    NOTE: asse X etichettato 'File ID' e nessun titolo nei subplot o suptitle.
    """
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    client_ids = [c['client_id'] for c in clients_data]
    attack_counts = [c['attack_count'] for c in clients_data]
    natural_counts = [c['natural_count'] for c in clients_data]
    attack_ratios = [c['attack_ratio'] * 100 for c in clients_data]
    
    # Sottografico 1: Stacked bar chart (valori assoluti)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(client_ids))
    width = 0.6
    
    bars1 = ax1.bar(x, natural_counts, width, label='Natural', color=COLOR_NATURAL, 
                    edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x, attack_counts, width, bottom=natural_counts, label='Attack', 
                    color=COLOR_ATTACK, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('File ID', fontsize=12)
    ax1.set_ylabel('Numero di Campioni', fontsize=12)
    # titolo rimosso
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_ids)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Sottografico 2: Bar chart percentuale
    ax2 = fig.add_subplot(gs[0, 1])
    natural_perc = [100 - ar for ar in attack_ratios]
    
    bars1 = ax2.bar(x, natural_perc, width, label='Natural', color=COLOR_NATURAL,
                    edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x, attack_ratios, width, bottom=natural_perc, label='Attack', 
                    color=COLOR_ATTACK, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('File ID', fontsize=12)
    ax2.set_ylabel('Percentuale (%)', fontsize=12)
    # titolo rimosso
    ax2.set_xticks(x)
    ax2.set_xticklabels(client_ids)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # Sottografico 3: Pie chart globale
    ax3 = fig.add_subplot(gs[1, 0])
    total_attack = sum(attack_counts)
    total_natural = sum(natural_counts)
    
    wedges, texts, autotexts = ax3.pie([total_natural, total_attack], 
                                        labels=['Natural', 'Attack'],
                                        autopct='%1.1f%%',
                                        colors=[COLOR_NATURAL, COLOR_ATTACK],
                                        explode=[0, 0.05],
                                        shadow=True,
                                        startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # titolo rimosso
    
    # Sottografico 4: Variabilità attack ratio tra client
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x, attack_ratios, width, color=COLOR_ATTACK, edgecolor='black', linewidth=0.5)
    
    mean_ratio = np.mean(attack_ratios)
    std_ratio = np.std(attack_ratios)
    
    ax4.axhline(y=mean_ratio, color='blue', linestyle='--', linewidth=2, 
                label=f'Media: {mean_ratio:.1f}%')
    ax4.axhspan(mean_ratio - std_ratio, mean_ratio + std_ratio, alpha=0.2, color='blue',
                label=f'±1 SD: {std_ratio:.1f}%')
    
    ax4.set_xlabel('File ID', fontsize=12)
    ax4.set_ylabel('Attack Ratio (%)', fontsize=12)
    # titolo rimosso
    ax4.set_xticks(x)
    ax4.set_xticklabels(client_ids)
    ax4.legend(loc='upper right')
    ax4.grid(axis='y', alpha=0.3)
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")
    
    return {
        'global_attack_ratio': total_attack / (total_attack + total_natural),
        'mean_client_ratio': mean_ratio / 100,
        'std_client_ratio': std_ratio / 100,
        'imbalance_ratio': max(total_natural, total_attack) / min(total_natural, total_attack)
    }


def plot_class_distribution_heatmap(clients_data, combined_df, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 2.2: Heatmap delle classi di attacco dettagliate.
    NOTE: y ticklabels mostrano 'File {id}' e titoli rimossi.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)
    
    # Estrai tutti i marker unici
    all_markers = combined_df['marker'].unique()
    
    # Crea matrice client x marker
    client_ids = [c['client_id'] for c in clients_data]
    matrix = np.zeros((len(client_ids), len(all_markers)))
    
    for i, c in enumerate(clients_data):
        marker_counts = c['df']['marker'].value_counts()
        for j, marker in enumerate(all_markers):
            if marker in marker_counts.index:
                matrix[i, j] = marker_counts[marker]
    
    # Heatmap valori assoluti
    ax1 = axes[0]
    im1 = ax1.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_yticks(range(len(client_ids)))
    ax1.set_yticklabels([f'File {cid}' for cid in client_ids])
    ax1.set_xticks(range(len(all_markers)))
    ax1.set_xticklabels(all_markers, rotation=45, ha='right', fontsize=9)
    # titolo rimosso
    plt.colorbar(im1, ax=ax1, label='N.Campioni')
    
    # Heatmap normalizzata per riga
    ax2 = axes[1]
    matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.nan_to_num(matrix_norm)  # Gestisci divisione per zero
    
    im2 = ax2.imshow(matrix_norm, aspect='auto', cmap='YlOrRd')
    ax2.set_yticks(range(len(client_ids)))
    ax2.set_yticklabels([f'File {cid}' for cid in client_ids])
    ax2.set_xticks(range(len(all_markers)))
    ax2.set_xticklabels(all_markers, rotation=45, ha='right', fontsize=9)
    # titolo rimosso
    plt.colorbar(im2, ax=ax2, label='Proporzione')
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


# =============================================================================
# SEZIONE 3: QUALITÀ DEI DATI
# =============================================================================

def plot_data_quality(clients_data, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 3.1: Analisi qualità dei dati per client.
    NOTE: asse X etichettato 'File ID' e titoli rimossi.
    """
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    client_ids = [c['client_id'] for c in clients_data]
    
    # Calcola metriche di qualità per ogni client
    missing_counts = []
    inf_counts = []
    constant_counts = []
    outlier_counts = []
    
    for c in clients_data:
        df = c['df'].drop(columns=['marker', 'client_id'], errors='ignore')
        df_numeric = df.select_dtypes(include=[np.number])
        
        # Missing values
        n_missing = df_numeric.isna().sum().sum()
        missing_counts.append(n_missing)
        
        # Valori infiniti
        n_inf = np.isinf(df_numeric.values).sum()
        inf_counts.append(n_inf)
        
        # Feature costanti (varianza < 1e-10)
        n_constant = (df_numeric.var() < 1e-10).sum()
        constant_counts.append(n_constant)
        
        # Outlier (usando IQR)
        n_outliers = 0
        for col in df_numeric.columns:
            q1 = df_numeric[col].quantile(0.25)
            q3 = df_numeric[col].quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (df_numeric[col] < q1 - 1.5*iqr) | (df_numeric[col] > q3 + 1.5*iqr)
            n_outliers += outlier_mask.sum()
        outlier_counts.append(n_outliers)
    
    x = np.arange(len(client_ids))
    width = 0.6
    
    # Sottografico 1: Missing values
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x, missing_counts, width, color='#3498DB', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('File ID', fontsize=11)
    ax1.set_ylabel('N.Missing Values', fontsize=11)
    # titolo rimosso
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_ids)
    ax1.grid(axis='y', alpha=0.3)
    
    # Aggiungi annotazioni
    for bar, count in zip(bars1, missing_counts):
        if count > 0:
            ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    
    # Sottografico 2: Valori infiniti
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, inf_counts, width, color='#E74C3C', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('File ID', fontsize=11)
    ax2.set_ylabel('N.Valori Infiniti', fontsize=11)
    # titolo rimosso
    ax2.set_xticks(x)
    ax2.set_xticklabels(client_ids)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars2, inf_counts):
        if count > 0:
            ax2.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    
    # Sottografico 3: Feature costanti
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x, constant_counts, width, color='#9B59B6', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('File ID', fontsize=11)
    ax3.set_ylabel('N.Feature Costanti', fontsize=11)
    # titolo rimosso
    ax3.set_xticks(x)
    ax3.set_xticklabels(client_ids)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars3, constant_counts):
        if count > 0:
            ax3.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    
    # Sottografico 4: Outlier
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(x, outlier_counts, width, color='#F39C12', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('File ID', fontsize=11)
    ax4.set_ylabel('N.Outlier (IQR)', fontsize=11)
    # titolo rimosso
    ax4.set_xticks(x)
    ax4.set_xticklabels(client_ids)
    ax4.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")
    
    return {
        'total_missing': sum(missing_counts),
        'total_inf': sum(inf_counts),
        'avg_constant_features': np.mean(constant_counts),
        'total_outliers': sum(outlier_counts)
    }


def plot_missing_heatmap(clients_data, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 3.2: Heatmap dei missing values per feature e client.
    NOTE: asse X etichettato 'File ID' e titolo rimosso.
    """
    # Trova le feature con più missing values
    all_missing = {}
    
    for c in clients_data:
        df = c['df'].drop(columns=['marker', 'client_id'], errors='ignore')
        df_numeric = df.select_dtypes(include=[np.number])
        
        for col in df_numeric.columns:
            if col not in all_missing:
                all_missing[col] = {}
            all_missing[col][c['client_id']] = df_numeric[col].isna().sum()
    
    # Seleziona top 30 feature con più missing
    total_missing_per_feature = {col: sum(clients.values()) for col, clients in all_missing.items()}
    top_features = sorted(total_missing_per_feature.keys(), 
                          key=lambda x: total_missing_per_feature[x], 
                          reverse=True)[:30]
    
    if not top_features or all(total_missing_per_feature[f] == 0 for f in top_features):
        print("ℹ️ Nessun missing value trovato, skip heatmap")
        return
    
    # Crea matrice
    client_ids = [c['client_id'] for c in clients_data]
    matrix = np.zeros((len(top_features), len(client_ids)))
    
    for i, feature in enumerate(top_features):
        for j, cid in enumerate(client_ids):
            if cid in all_missing.get(feature, {}):
                matrix[i, j] = all_missing[feature][cid]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.3)))
    
    im = ax.imshow(matrix, aspect='auto', cmap='Reds')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels([str(cid) for cid in client_ids])
    ax.set_xlabel('File ID', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    # titolo rimosso
    
    plt.colorbar(im, ax=ax, label='N.Missing Values')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


# =============================================================================
# SEZIONE 4: CARATTERISTICHE DELLE FEATURE
# =============================================================================

def plot_feature_statistics(combined_df, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 4.1: Statistiche globali delle feature.
    NOTE: titoli rimosso dove possibile; asse X/Y lasciati come prima.
    """
    df_numeric = combined_df.drop(columns=['marker', 'client_id'], errors='ignore')
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    
    # Calcola statistiche
    stats_df = df_numeric.describe().T
    stats_df['skewness'] = df_numeric.skew()
    stats_df['kurtosis'] = df_numeric.kurtosis()
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['cv'] = stats_df['std'] / stats_df['mean'].abs()  # Coefficiente di variazione
    
    # Ordina per varianza
    stats_df = stats_df.sort_values('std', ascending=False)
    
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Top 20 feature per deviazione standard
    top_20 = stats_df.head(20)
    
    # Sottografico 1: Range delle feature
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(top_20))
    ax1.barh(y_pos, top_20['range'].values, color='steelblue', edgecolor='black', linewidth=0.3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_20.index, fontsize=8)
    ax1.set_xlabel('Range (Max - Min)', fontsize=11)
    # titolo rimosso
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Sottografico 2: Deviazione standard
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(y_pos, top_20['std'].values, color='coral', edgecolor='black', linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_20.index, fontsize=8)
    ax2.set_xlabel('Deviazione Standard', fontsize=11)
    # titolo rimosso
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Sottografico 3: Distribuzione skewness
    ax3 = fig.add_subplot(gs[1, 0])
    skewness_all = df_numeric.skew().dropna()
    ax3.hist(skewness_all, bins=50, color='mediumpurple', edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Distribuzione simmetrica')
    ax3.axvline(x=skewness_all.mean(), color='blue', linestyle='-', linewidth=2, 
                label=f'Media: {skewness_all.mean():.2f}')
    ax3.set_xlabel('Skewness', fontsize=11)
    ax3.set_ylabel('N.Feature', fontsize=11)
    # titolo rimosso
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Sottografico 4: Coefficiente di variazione
    ax4 = fig.add_subplot(gs[1, 1])
    cv_valid = stats_df['cv'].replace([np.inf, -np.inf], np.nan).dropna()
    cv_clipped = cv_valid.clip(upper=cv_valid.quantile(0.95))  # Clip per visualizzazione
    ax4.hist(cv_clipped, bins=50, color='teal', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Coefficiente di Variazione (CV)', fontsize=11)
    ax4.set_ylabel('N.Feature', fontsize=11)
    # titolo rimosso
    ax4.grid(axis='y', alpha=0.3)
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


def plot_correlation_matrix(combined_df, top_k, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 4.2: Matrice di correlazione delle feature principali.
    NOTE: titolo rimosso.
    """
    df_numeric = combined_df.drop(columns=['marker', 'client_id'], errors='ignore')
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    
    # Imputa NaN con mediana per calcolare correlazione
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # Seleziona top-k feature per varianza
    variances = df_numeric.var().sort_values(ascending=False)
    top_features = variances.head(top_k).index.tolist()
    
    # Calcola matrice di correlazione
    corr_matrix = df_numeric[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap con maschera triangolare
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    
    # titolo rimosso
    
    # Ruota etichette
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


def plot_pca_analysis(combined_df, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 4.3: Analisi PCA del dataset.
    NOTE: titoli/subtitles rimossi; mantenute etichette assi; legend minima.
    """
    df_numeric = combined_df.drop(columns=['marker', 'client_id'], errors='ignore')
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    
    # Prepara dati per PCA
    df_clean = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(df_numeric.median())
    
    # Standardizza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Applica PCA completa
    n_components = min(50, X_scaled.shape[1])
    pca_full = PCA(n_components=n_components)
    X_pca_full = pca_full.fit_transform(X_scaled)
    
    # Estrai labels se disponibili
    if 'marker' in combined_df.columns:
        y = (combined_df['marker'] != "Natural").astype(int).values
    else:
        y = np.zeros(len(combined_df))
    
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Sottografico 1: Varianza spiegata per componente
    ax1 = fig.add_subplot(gs[0, 0])
    explained_var = pca_full.explained_variance_ratio_ * 100
    ax1.bar(range(1, len(explained_var)+1), explained_var, color='steelblue', 
            edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Componente Principale', fontsize=11)
    ax1.set_ylabel('Varianza Spiegata (%)', fontsize=11)
    # titolo rimosso
    ax1.grid(axis='y', alpha=0.3)
    
    # Sottografico 2: Varianza cumulativa
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_var = np.cumsum(explained_var)
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% varianza')
    ax2.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='95% varianza')
    
    # Trova numero componenti per 90% e 95%
    n_90 = np.argmax(cumulative_var >= 90) + 1
    n_95 = np.argmax(cumulative_var >= 95) + 1
    
    ax2.axvline(x=n_90, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(x=n_95, color='orange', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('N.Componenti', fontsize=11)
    ax2.set_ylabel('Varianza Cumulativa (%)', fontsize=11)
    # titolo rimosso
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Sottografico 3: Proiezione 2D (PC1 vs PC2)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Sottocampiona per visualizzazione
    n_plot = min(5000, len(X_pca_full))
    idx_plot = np.random.choice(len(X_pca_full), n_plot, replace=False)
    
    scatter = ax3.scatter(X_pca_full[idx_plot, 0], X_pca_full[idx_plot, 1], 
                          c=y[idx_plot], cmap='RdYlGn_r', alpha=0.5, s=10)
    ax3.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=11)
    ax3.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=11)
    # titolo rimosso
    
    # Legenda minima
    legend_elements = [Patch(facecolor=plt.cm.RdYlGn_r(0.1), label='Natural'),
                       Patch(facecolor=plt.cm.RdYlGn_r(0.9), label='Attack')]
    ax3.legend(handles=legend_elements, loc='upper right')
    ax3.grid(alpha=0.3)
    
    # Sottografico 4: Proiezione 2D (PC1 vs PC3)
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(X_pca_full[idx_plot, 0], X_pca_full[idx_plot, 2], 
                          c=y[idx_plot], cmap='RdYlGn_r', alpha=0.5, s=10)
    ax4.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=11)
    ax4.set_ylabel(f'PC3 ({explained_var[2]:.1f}%)', fontsize=11)
    # titolo rimosso
    ax4.legend(handles=legend_elements, loc='upper right')
    ax4.grid(alpha=0.3)
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")
    
    return {'n_components_90': n_90, 'n_components_95': n_95}


# =============================================================================
# SEZIONE 5: VARIABILITÀ INTER-CLIENT (ETEROGENEITÀ FEDERATA)
# =============================================================================

def plot_client_heterogeneity(clients_data, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 5.1: Analisi dell'eterogeneità tra client.
    NOTE: asse X etichettato 'File ID' e titoli rimossi.
    """
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    client_ids = [c['client_id'] for c in clients_data]
    n_clients = len(clients_data)
    
    # Calcola statistiche per ogni client
    client_means = []
    client_stds = []
    
    for c in clients_data:
        df = c['df'].drop(columns=['marker', 'client_id'], errors='ignore')
        df_numeric = df.select_dtypes(include=[np.number])
        df_clean = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(df_numeric.median())
        
        client_means.append(df_clean.mean().values)
        client_stds.append(df_clean.std().values)
    
    client_means = np.array(client_means)
    client_stds = np.array(client_stds)
    
    # Calcola divergenza dalla media globale
    global_mean = np.mean(client_means, axis=0)
    global_std = np.mean(client_stds, axis=0)
    
    divergence_mean = np.sqrt(np.sum((client_means - global_mean)**2, axis=1))
    divergence_std = np.sqrt(np.sum((client_stds - global_std)**2, axis=1))
    
    # Sottografico 1: Divergenza dalla media globale
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(n_clients)
    ax1.bar(x, divergence_mean, color='#3498DB', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('File ID', fontsize=11)
    ax1.set_ylabel('Divergenza (L2)', fontsize=11)
    # titolo rimosso
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_ids)
    ax1.grid(axis='y', alpha=0.3)
    
    # Sottografico 2: Divergenza delle varianze
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, divergence_std, color='#E74C3C', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('File ID', fontsize=11)
    ax2.set_ylabel('Divergenza (L2)', fontsize=11)
    # titolo rimosso
    ax2.set_xticks(x)
    ax2.set_xticklabels(client_ids)
    ax2.grid(axis='y', alpha=0.3)
    
    # Sottografico 3: Matrice di similarità tra client (basata su medie)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calcola distanze pairwise
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(client_means, metric='euclidean'))
    
    # Normalizza per visualizzazione
    distances_norm = distances / distances.max() if distances.max() != 0 else distances
    similarity = 1 - distances_norm
    
    im = ax3.imshow(similarity, cmap='YlGnBu', vmin=0, vmax=1)
    ax3.set_xticks(range(n_clients))
    ax3.set_yticks(range(n_clients))
    ax3.set_xticklabels([f'File {cid}' for cid in client_ids])
    ax3.set_yticklabels([f'File {cid}' for cid in client_ids])
    # titolo rimosso
    plt.colorbar(im, ax=ax3, label='Similarità')
    
    # Sottografico 4: Eterogeneità complessiva
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Metrica combinata
    heterogeneity_score = (divergence_mean / divergence_mean.max() + 
                           divergence_std / divergence_std.max()) / 2 if (divergence_mean.max() > 0 and divergence_std.max() > 0) else np.zeros_like(divergence_mean)
    
    colors = plt.cm.RdYlGn_r(heterogeneity_score)
    bars = ax4.bar(x, heterogeneity_score, color=colors, edgecolor='black', linewidth=0.5)
    
    ax4.axhline(y=np.mean(heterogeneity_score), color='blue', linestyle='--', 
                linewidth=2, label=f'Media: {np.mean(heterogeneity_score):.3f}')
    
    ax4.set_xlabel('File ID', fontsize=11)
    ax4.set_ylabel('Score Eterogeneità', fontsize=11)
    # titolo rimosso
    ax4.set_xticks(x)
    ax4.set_xticklabels(client_ids)
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")
    
    return {
        'mean_heterogeneity': np.mean(heterogeneity_score),
        'max_heterogeneity_client': client_ids[np.argmax(heterogeneity_score)],
        'min_heterogeneity_client': client_ids[np.argmin(heterogeneity_score)]
    }


def plot_feature_importance_comparison(clients_data, top_k, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 5.2: Confronto feature importance tra client.
    NOTE: y ticklabels mostrano 'File {id}', titoli rimossi.
    """
    # Calcola varianza per feature per ogni client
    all_variances = {}
    
    for c in clients_data:
        df = c['df'].drop(columns=['marker', 'client_id'], errors='ignore')
        df_numeric = df.select_dtypes(include=[np.number])
        variances = df_numeric.var()
        all_variances[c['client_id']] = variances
    
    # Trova feature comuni
    common_features = set.intersection(*[set(v.index) for v in all_variances.values()])
    
    # Seleziona top-k feature per varianza globale
    global_variance = sum([all_variances[cid][list(common_features)] for cid in all_variances.keys()])
    top_features = global_variance.sort_values(ascending=False).head(top_k).index.tolist()
    
    # Crea matrice (client x feature) con rank
    client_ids = list(all_variances.keys())
    rank_matrix = np.zeros((len(client_ids), len(top_features)))
    
    for i, cid in enumerate(client_ids):
        client_var = all_variances[cid][top_features]
        ranks = client_var.rank(ascending=False)
        for j, feat in enumerate(top_features):
            rank_matrix[i, j] = ranks[feat]
    
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)
    
    # Heatmap dei rank
    ax1 = axes[0]
    im1 = ax1.imshow(rank_matrix, aspect='auto', cmap='RdYlGn_r')
    ax1.set_yticks(range(len(client_ids)))
    ax1.set_yticklabels([f'File {cid}' for cid in client_ids])
    ax1.set_xticks(range(len(top_features)))
    ax1.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
    # titolo rimosso
    plt.colorbar(im1, ax=ax1, label='Rank')
    
    # Varianza dei rank (misura di disaccordo tra client)
    ax2 = axes[1]
    rank_variance = np.var(rank_matrix, axis=0)
    colors = plt.cm.RdYlGn_r(rank_variance / rank_variance.max() if rank_variance.max() > 0 else rank_variance)
    bars = ax2.barh(range(len(top_features)), rank_variance, color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=9)
    ax2.set_xlabel('Varianza del Rank tra Client', fontsize=11)
    # titolo rimosso
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


# =============================================================================
# SEZIONE 6: ANALISI SCENARI DI ATTACCO
# =============================================================================

def plot_attack_scenarios_analysis(combined_df, clients_data, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 6.1: Analisi dettagliata degli scenari di attacco.
    NOTE: asse X etichettato 'File ID' nelle heatmap, titoli rimossi.
    """
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Sottografico 1: Distribuzione globale scenari
    ax1 = fig.add_subplot(gs[0, 0])
    scenario_counts = combined_df['marker'].value_counts()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_counts)))
    wedges, texts, autotexts = ax1.pie(
        scenario_counts.values,
        labels=None,  # Etichette nella legenda
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(scenario_counts),
        shadow=True
    )
    
    # Legenda esterna
    ax1.legend(wedges, scenario_counts.index, 
               title="Scenari", loc="center left", 
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    # titolo rimosso
    
    # Sottografico 2: Bar chart scenari
    ax2 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(scenario_counts))
    bars = ax2.barh(y_pos, scenario_counts.values, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(scenario_counts.index, fontsize=9)
    ax2.set_xlabel('Numero Campioni', fontsize=11)
    # titolo rimosso
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, count in zip(bars, scenario_counts.values):
        ax2.annotate(f'{count}', 
                     xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                     ha='left', va='center', fontsize=8, xytext=(3, 0),
                     textcoords='offset points')
    
    # Sottografico 3: Heatmap scenario x client
    ax3 = fig.add_subplot(gs[1, :])
    
    all_scenarios = combined_df['marker'].unique()
    client_ids = [c['client_id'] for c in clients_data]
    
    # Matrice scenario x client (normalizzata per client)
    matrix = np.zeros((len(all_scenarios), len(client_ids)))
    
    for j, c in enumerate(clients_data):
        scenario_counts_client = c['df']['marker'].value_counts()
        total_client = len(c['df'])
        for i, scenario in enumerate(all_scenarios):
            if scenario in scenario_counts_client.index and total_client > 0:
                matrix[i, j] = scenario_counts_client[scenario] / total_client * 100
    
    im = ax3.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax3.set_yticks(range(len(all_scenarios)))
    ax3.set_yticklabels(all_scenarios, fontsize=9)
    ax3.set_xticks(range(len(client_ids)))
    ax3.set_xticklabels([str(cid) for cid in client_ids])
    ax3.set_xlabel('File ID', fontsize=11)
    ax3.set_ylabel('Scenario', fontsize=11)
    # titolo rimosso
    
    plt.colorbar(im, ax=ax3, label='% del File', shrink=0.8)
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


# =============================================================================
# SEZIONE 7: SUMMARY STATISTICO
# =============================================================================

def plot_dataset_summary(clients_data, combined_df, save_path, dpi=DPI_DEFAULT):
    """
    Grafico 7.1: Summary statistico del dataset.
    NOTE: sostituita etichetta asse X con 'File ID' dove presente e rimossi i titoli.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
    
    # ========== SEZIONE 1: STATISTICHE GLOBALI (Testo) ==========
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis('off')
    
    # Calcola statistiche
    total_samples = len(combined_df)
    n_clients = len(clients_data)
    n_features = combined_df.shape[1] - 2  # Escludi marker e client_id
    n_scenarios = combined_df['marker'].nunique()
    attack_ratio = (combined_df['marker'] != 'Natural').mean() * 100
    
    stats_text = f"""
    📊 STATISTICHE GLOBALI DATASET
    ─────────────────────────────
    
    Campioni totali: {total_samples:,}
    Numero client: {n_clients}
    Feature: {n_features}
    Scenari unici: {n_scenarios}
    
    Distribuzione classi:
    • Attack: {attack_ratio:.1f}%
    • Natural: {100-attack_ratio:.1f}%
    
    Campioni per client:
    • Media: {total_samples/n_clients:,.0f}
    • Min: {min([c['n_samples'] for c in clients_data]):,}
    • Max: {max([c['n_samples'] for c in clients_data]):,}
    """
    
    ax_text.text(0.1, 0.9, stats_text, transform=ax_text.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    # titolo rimosso
    
    # ========== SEZIONE 2: Distribuzione campioni ==========
    ax2 = fig.add_subplot(gs[0, 1])
    client_ids = [c['client_id'] for c in clients_data]
    n_samples = [c['n_samples'] for c in clients_data]
    
    bars = ax2.bar(client_ids, n_samples, color=COLORS_CLIENTS[:len(client_ids)], 
                   edgecolor='black', linewidth=0.5)
    ax2.axhline(y=np.mean(n_samples), color='red', linestyle='--', linewidth=2, label='Media')
    ax2.set_xlabel('File ID', fontsize=10)
    ax2.set_ylabel('Campioni', fontsize=10)
    # titolo rimosso
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # ========== SEZIONE 3: Distribuzione classi ==========
    ax3 = fig.add_subplot(gs[0, 2])
    attack_ratios = [c['attack_ratio'] * 100 for c in clients_data]
    
    ax3.bar(client_ids, attack_ratios, color=COLOR_ATTACK, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=np.mean(attack_ratios), color='blue', linestyle='--', linewidth=2, label='Media')
    ax3.set_xlabel('File ID', fontsize=10)
    ax3.set_ylabel('% Attacchi', fontsize=10)
    # titolo rimosso
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # ========== SEZIONE 4: Box plot feature principali ==========
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Seleziona top 6 feature per varianza
    df_numeric = combined_df.drop(columns=['marker', 'client_id'], errors='ignore')
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    variances = df_numeric.var().sort_values(ascending=False)
    top_6_features = variances.head(6).index.tolist()
    
    # Prepara dati per boxplot
    data_for_boxplot = [df_numeric[feat].dropna().values for feat in top_6_features]
    
    bp = ax4.boxplot(data_for_boxplot, patch_artist=True, notch=True)
    colors_box = plt.cm.Set2(np.linspace(0, 1, len(top_6_features)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xticklabels(top_6_features, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Valore', fontsize=10)
    # titolo rimosso
    ax4.grid(axis='y', alpha=0.3)
    
    # ========== SEZIONE 5: Pie chart scenari ==========
    ax5 = fig.add_subplot(gs[1, 2])
    scenario_counts = combined_df['marker'].value_counts()
    
    # Limita a top 5 + "Altri"
    if len(scenario_counts) > 5:
        top_5 = scenario_counts.head(5)
        others = scenario_counts[5:].sum()
        scenario_plot = pd.concat([top_5, pd.Series({'Altri': others})])
    else:
        scenario_plot = scenario_counts
    
    colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(scenario_plot)))
    ax5.pie(scenario_plot.values, labels=scenario_plot.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    # titolo rimosso
    
    # ========== SEZIONE 6: Correlazione top feature ==========
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Seleziona top 8 feature
    top_8_features = variances.head(8).index.tolist()
    corr_matrix = df_numeric[top_8_features].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.1f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax6, annot_kws={'size': 8})
    # titolo rimosso
    
    # ========== SEZIONE 7: Eterogeneità client ==========
    ax7 = fig.add_subplot(gs[2, 1])
    
    # Calcola eterogeneità semplificata
    client_means = []
    for c in clients_data:
        df = c['df'].drop(columns=['marker', 'client_id'], errors='ignore')
        df_num = df.select_dtypes(include=[np.number])
        df_clean = df_num.replace([np.inf, -np.inf], np.nan).fillna(df_num.median())
        client_means.append(df_clean.mean().values)
    
    client_means = np.array(client_means)
    global_mean = np.mean(client_means, axis=0)
    heterogeneity = np.sqrt(np.sum((client_means - global_mean)**2, axis=1))
    heterogeneity_norm = heterogeneity / heterogeneity.max() if heterogeneity.max() > 0 else heterogeneity
    
    colors_het = plt.cm.RdYlGn_r(heterogeneity_norm)
    ax7.bar(client_ids, heterogeneity_norm, color=colors_het, edgecolor='black', linewidth=0.5)
    ax7.axhline(y=np.mean(heterogeneity_norm), color='blue', linestyle='--', linewidth=2, label='Media')
    ax7.set_xlabel('File ID', fontsize=10)
    ax7.set_ylabel('Score Eterogeneità (norm)', fontsize=10)
    # titolo rimosso
    ax7.legend()
    ax7.set_ylim(0, 1.1)
    ax7.grid(axis='y', alpha=0.3)
    
    # ========== SEZIONE 8: Legenda interpretativa ==========
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    legend_text = """
    📖 LEGENDA INTERPRETATIVA
    ─────────────────────────
    
    🔴 Eterogeneità Alta:
       File con dati molto diversi
       dalla media globale
    
    🟢 Eterogeneità Bassa:
       File con dati simili
       alla media globale
    
    📊 Non-IID:
       Distribuzione non identica
       tra file → sfida per FL
    
    ⚖️ Sbilanciamento:
       Classi non equamente
       rappresentate
    """
    
    ax8.text(0.1, 0.9, legend_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    # suptitle rimosso
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Salvato: {os.path.basename(save_path)}")


# =============================================================================
# REPORT TESTUALE
# =============================================================================

def write_eda_report(clients_data, combined_df, stats, report_path):
    """
    Genera report testuale dettagliato dell'EDA.
    
    Args:
        clients_data: Lista dati per client
        combined_df: DataFrame combinato
        stats: Dictionary con statistiche calcolate
        report_path: Path del file di output
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS (EDA) - DATASET SMARTGRID\n")
        f.write("Per Federated Learning - Tesi di Laurea\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data generazione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Sezione 1: Overview Dataset
        f.write("="*80 + "\n")
        f.write("1.OVERVIEW DATASET\n")
        f.write("="*80 + "\n\n")
        
        total_samples = len(combined_df)
        n_clients = len(clients_data)
        n_features = combined_df.shape[1] - 2
        n_scenarios = combined_df['marker'].nunique()
        
        f.write(f"Campioni totali: {total_samples:,}\n")
        f.write(f"Numero client: {n_clients}\n")
        f.write(f"Feature numeriche: {n_features}\n")
        f.write(f"Scenari unici: {n_scenarios}\n\n")
        
        # Sezione 2: Distribuzione per Client
        f.write("="*80 + "\n")
        f.write("2.DISTRIBUZIONE PER CLIENT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Client':<10} {'Campioni':<12} {'Attacchi':<12} {'Naturali':<12} {'% Attack':<12}\n")
        f.write("-"*60 + "\n")
        
        for c in clients_data:
            f.write(f"{c['client_id']:<10} {c['n_samples']:<12} {c['attack_count']:<12} "
                    f"{c['natural_count']:<12} {c['attack_ratio']*100:<12.2f}\n")
        
        f.write("\n")
        
        # Sezione 3: Distribuzione Scenari
        f.write("="*80 + "\n")
        f.write("3.DISTRIBUZIONE SCENARI\n")
        f.write("="*80 + "\n\n")
        
        scenario_counts = combined_df['marker'].value_counts()
        f.write(f"{'Scenario':<30} {'Campioni':<12} {'Percentuale':<12}\n")
        f.write("-"*60 + "\n")
        
        for scenario, count in scenario_counts.items():
            pct = count / total_samples * 100
            f.write(f"{scenario:<30} {count:<12} {pct:<12.2f}%\n")
        
        f.write("\n")
        
        # Sezione 4: Statistiche Feature
        f.write("="*80 + "\n")
        f.write("4.STATISTICHE FEATURE PRINCIPALI\n")
        f.write("="*80 + "\n\n")
        
        df_numeric = combined_df.drop(columns=['marker', 'client_id'], errors='ignore')
        df_numeric = df_numeric.select_dtypes(include=[np.number])
        
        # Top 10 feature per varianza
        variances = df_numeric.var().sort_values(ascending=False)
        top_10 = variances.head(10)
        
        f.write("Top 10 Feature per Varianza:\n")
        f.write(f"{'Feature':<25} {'Varianza':<15} {'Media':<15} {'Std':<15}\n")
        f.write("-"*70 + "\n")
        
        for feat in top_10.index:
            var = df_numeric[feat].var()
            mean = df_numeric[feat].mean()
            std = df_numeric[feat].std()
            f.write(f"{feat:<25} {var:<15.4f} {mean:<15.4f} {std:<15.4f}\n")
        
        f.write("\n")
        
        # Sezione 5: Qualità Dati
        f.write("="*80 + "\n")
        f.write("5.QUALITÀ DEI DATI\n")
        f.write("="*80 + "\n\n")
        
        total_missing = df_numeric.isna().sum().sum()
        total_inf = np.isinf(df_numeric.values).sum()
        total_values = df_numeric.size
        
        f.write(f"Valori totali: {total_values:,}\n")
        f.write(f"Valori mancanti (NaN): {total_missing:,} ({total_missing/total_values*100:.4f}%)\n")
        f.write(f"Valori infiniti: {total_inf:,} ({total_inf/total_values*100:.4f}%)\n\n")
        
        # Sezione 6: Eterogeneità
        f.write("="*80 + "\n")
        f.write("6.ANALISI ETEROGENEITÀ (NON-IID)\n")
        f.write("="*80 + "\n\n")
        
        # Calcola CV della distribuzione campioni
        n_samples_list = [c['n_samples'] for c in clients_data]
        cv_samples = np.std(n_samples_list) / np.mean(n_samples_list) * 100
        
        attack_ratios = [c['attack_ratio'] for c in clients_data]
        cv_attack = np.std(attack_ratios) / np.mean(attack_ratios) * 100 if np.mean(attack_ratios) > 0 else 0
        
        f.write(f"Coefficiente di Variazione (CV) campioni: {cv_samples:.2f}%\n")
        f.write(f"Coefficiente di Variazione (CV) attack ratio: {cv_attack:.2f}%\n")
        f.write(f"\nInterpretazione:\n")
        
        if cv_samples > 50:
            f.write("  - CV campioni ALTO: distribuzione molto sbilanciata tra client\n")
        elif cv_samples > 20:
            f.write("  - CV campioni MEDIO: distribuzione moderatamente sbilanciata\n")
        else:
            f.write("  - CV campioni BASSO: distribuzione relativamente uniforme\n")
        
        if cv_attack > 50:
            f.write("  - CV attack ratio ALTO: forte eterogeneità nelle classi → Non-IID significativo\n")
        elif cv_attack > 20:
            f.write("  - CV attack ratio MEDIO: eterogeneità moderata → Non-IID presente\n")
        else:
            f.write("  - CV attack ratio BASSO: eterogeneità limitata → Quasi IID\n")
        
        f.write("\n")
        
        # Sezione 7: Conclusioni
        f.write("="*80 + "\n")
        f.write("7.CONCLUSIONI E IMPLICAZIONI PER FEDERATED LEARNING\n")
        f.write("="*80 + "\n\n")
        
        f.write("Caratteristiche principali del dataset:\n\n")
        
        f.write("✓ DISTRIBUZIONE NON-IID:\n")
        f.write(f"  Il dataset presenta una distribuzione Non-IID tra i {n_clients} client,\n")
        f.write("  con variazioni significative nella proporzione di attacchi e campioni.\n")
        f.write("  Questo rappresenta una sfida tipica del Federated Learning reale.\n\n")
        
        f.write("✓ SBILANCIAMENTO DI CLASSE:\n")
        attack_global = (combined_df['marker'] != 'Natural').mean() * 100
        f.write(f"  Proporzione globale attacchi: {attack_global:.1f}%\n")
        f.write("  Lo sbilanciamento richiede strategie appropriate (class weighting, etc.)\n\n")
        
        f.write("✓ QUALITÀ DEI DATI:\n")
        if total_missing + total_inf > 0:
            f.write(f"  Presenza di valori mancanti/infiniti → richiede preprocessing\n")
        else:
            f.write(f"  Dati puliti senza valori mancanti/infiniti significativi\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Fine Report EDA\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Report testuale salvato: {report_path}")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main(data_dir=None, client_range=(1, 16), top_k=TOP_K_FEATURES, dpi=DPI_DEFAULT):
    """
    Funzione principale che orchestra l'intera EDA.
    
    Args:
        data_dir: Directory contenente i file data{id}.csv
        client_range: Tupla (start, end) per range client
        top_k: Numero di feature principali da visualizzare
        dpi: Risoluzione grafici
    """
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS (EDA) - DATASET SMARTGRID")
    print("Per Federated Learning - Generazione Grafici per Tesi")
    print("="*80 + "\n")
    
    # Setup directory
    ensure_dirs()
    
    # Auto-detect data directory
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "SmartGrid"))
    
    print(f"Directory dati: {data_dir}")
    print(f"Range client: {client_range[0]}-{client_range[1]-1}")
    print(f"Top-K feature: {top_k}")
    print(f"DPI grafici: {dpi}")
    print()
    
    # ========== CARICAMENTO DATI ==========
    clients_data, combined_df = load_all_clients_data(data_dir, client_range)
    
    if combined_df.empty:
        print("❌ Nessun dato caricato.Verificare il percorso.")
        return
    
    # ========== GENERAZIONE GRAFICI ==========
    print(f"\n{'='*60}")
    print("GENERAZIONE GRAFICI PER TESI")
    print(f"{'='*60}\n")
    
    # 1.Distribuzione campioni
    print("\n📊 Sezione 1: Distribuzione dei Dati (Non-IID)")
    stats_samples = plot_samples_distribution(
        clients_data, 
        os.path.join(PLOTS_DIR, "01_samples_distribution.png"),
        dpi=dpi
    )
    
    # 2.Distribuzione feature per client
    plot_feature_distribution_per_client(
        clients_data, top_k,
        os.path.join(PLOTS_DIR, "02_feature_distribution_violin.png"),
        dpi=dpi
    )
    
    # 3.Sbilanciamento classi
    print("\n📊 Sezione 2: Sbilanciamento di Classe")
    stats_imbalance = plot_class_imbalance(
        clients_data,
        os.path.join(PLOTS_DIR, "03_class_imbalance.png"),
        dpi=dpi
    )
    
    # 4.Heatmap classi dettagliate
    plot_class_distribution_heatmap(
        clients_data, combined_df,
        os.path.join(PLOTS_DIR, "04_class_heatmap.png"),
        dpi=dpi
    )
    
    # 5.Qualità dati
    print("\n📊 Sezione 3: Qualità dei Dati")
    stats_quality = plot_data_quality(
        clients_data,
        os.path.join(PLOTS_DIR, "05_data_quality.png"),
        dpi=dpi
    )
    
    # 6.Heatmap missing values
    plot_missing_heatmap(
        clients_data,
        os.path.join(PLOTS_DIR, "06_missing_heatmap.png"),
        dpi=dpi
    )
    
    # 7.Statistiche feature
    print("\n📊 Sezione 4: Caratteristiche delle Feature")
    plot_feature_statistics(
        combined_df,
        os.path.join(PLOTS_DIR, "07_feature_statistics.png"),
        dpi=dpi
    )
    
    # 8.Matrice correlazione
    plot_correlation_matrix(
        combined_df, top_k,
        os.path.join(PLOTS_DIR, "08_correlation_matrix.png"),
        dpi=dpi
    )
    
    # 9.Analisi PCA
    stats_pca = plot_pca_analysis(
        combined_df,
        os.path.join(PLOTS_DIR, "09_pca_analysis.png"),
        dpi=dpi
    )
    
    # 10.Eterogeneità client
    print("\n📊 Sezione 5: Variabilità Inter-Client (Eterogeneità)")
    stats_heterogeneity = plot_client_heterogeneity(
        clients_data,
        os.path.join(PLOTS_DIR, "10_client_heterogeneity.png"),
        dpi=dpi
    )
    
    # 11.Confronto feature importance
    plot_feature_importance_comparison(
        clients_data, top_k,
        os.path.join(PLOTS_DIR, "11_feature_importance_comparison.png"),
        dpi=dpi
    )
    
    # 12.Analisi scenari attacco
    print("\n📊 Sezione 6: Analisi Scenari di Attacco")
    plot_attack_scenarios_analysis(
        combined_df, clients_data,
        os.path.join(PLOTS_DIR, "12_attack_scenarios.png"),
        dpi=dpi
    )
    
    # 13.Summary statistico
    print("\n📊 Sezione 7: Summary Statistico")
    plot_dataset_summary(
        clients_data, combined_df,
        os.path.join(PLOTS_DIR, "13_dataset_summary.png"),
        dpi=dpi
    )
    
    # ========== REPORT TESTUALE ==========
    print("\n📄 Generazione Report Testuale")
    stats = {
        'samples': stats_samples,
        'imbalance': stats_imbalance,
        'quality': stats_quality,
        'pca': stats_pca,
        'heterogeneity': stats_heterogeneity
    }
    
    write_eda_report(
        clients_data, combined_df, stats,
        os.path.join(RESULTS_DIR, "eda_report.txt")
    )
    
    # ========== SUMMARY FINALE ==========
    print(f"\n{'='*80}")
    print("✅ EDA COMPLETATA CON SUCCESSO!")
    print(f"{'='*80}")
    print(f"\n📁 Directory risultati: {RESULTS_DIR}")
    print(f"📊 Grafici generati: {PLOTS_DIR}")
    print(f"\nGrafici generati:")
    
    plot_files = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])
    for i, f in enumerate(plot_files, 1):
        print(f"  {i:2d}.{f}")
    
    print(f"\nReport testuale: {os.path.join(RESULTS_DIR, 'eda_report.txt')}")
    print(f"\n💡 Suggerimento: Usa questi grafici nella sezione 'Dataset' della tesi")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EDA per SmartGrid Dataset - Genera grafici per tesi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:

  # Esecuzione standard
  python dataset_eda_thesis.py

  # Con percorso dati personalizzato
  python dataset_eda_thesis.py --data-dir /path/to/SmartGrid

  # Con più feature e risoluzione maggiore
  python dataset_eda_thesis.py --top-k 12 --dpi 300

OUTPUT:
  I grafici vengono salvati in: federated/SmartGrid/results_EDA/plots/
  Il report testuale in: federated/SmartGrid/results_EDA/eda_report.txt
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory contenente i file data{id}.csv (default: auto-detect)"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Primo client ID (inclusive, default: 1)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=16,
        help="Ultimo client ID (exclusive, default: 16)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_FEATURES,
        help=f"Numero di feature principali da visualizzare (default: {TOP_K_FEATURES})"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI_DEFAULT,
        help=f"Risoluzione grafici in DPI (default: {DPI_DEFAULT})"
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        client_range=(args.start, args.end),
        top_k=args.top_k,
        dpi=args.dpi
    )