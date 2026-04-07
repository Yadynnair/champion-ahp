"""
Visualization Functions for IPCM Method Comparison

Plotting functions for Mean Euclidean Distance and Kendall's Tau results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Algorithm name mapping (used across visualization and analysis)
ALG_NAME_MAPPING = {
    'create_incomplete_pcm_AHP_express': 'AHP Express',
    'create_incomplete_pcm_star': 'Star',
    'create_incomplete_pcm_cycle': 'Cycle',
    'create_incomplete_pcm_tournament': 'C-AHP',
    'GeometricMean': 'Benchmark',
    'create_incomplete_pcm_tournament_champion_closure': 'CC-AHP'
}

# Marker mapping for line plots
MARKER_MAPPING = {
    'Star': '*',
    'AHP Express': 'v',
    'C-AHP': '^',
    'Cycle': 'o',
    'Benchmark': 's',
    'CC-AHP': 'X',
}

# Algorithm order for plots and legends
ALG_ORDER = ['Benchmark', 'Star', 'AHP Express', 'C-AHP', 'Cycle', 'CC-AHP']


def create_performance_summary(df_performance):
    """
    Create summary statistics from performance DataFrame.
    
    Parameters
    ----------
    df_performance : DataFrame
        Must have columns: Sigma, Alternatives, Algorithm, EuclideanDistance, KendallsTau
    
    Returns
    -------
    performance_summary : DataFrame
        Aggregated mean performance by Sigma, Alternatives, Algorithm
    """
    performance_summary = df_performance.groupby(
        ['Sigma', 'Alternatives', 'Algorithm']
    ).agg(
        MeanEuclideanDistance=('EuclideanDistance', 'mean'),
        StdEuclideanDistance=('EuclideanDistance', 'std'),
        MeanKendallsTau=('KendallsTau', 'mean'),
        StdKendallsTau=('KendallsTau', 'std'),
        Count=('EuclideanDistance', 'count')
    ).reset_index()
    
    return performance_summary


def plot_mean_euclidean_line(performance_summary, selected_sigmas=[0.5, 1.5, 2.5], 
                              save_path=None):
    """
    Plot Mean Euclidean Distance vs Number of Alternatives for each Sigma.
    
    Parameters
    ----------
    performance_summary : DataFrame
        From create_performance_summary()
    selected_sigmas : list
        Sigma values to include in plot
    save_path : str, optional
        Path to save figure (without extension). Saves both PDF and TIFF.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelweight': 'bold',
        'savefig.dpi': 300
    })

    df_plot = performance_summary.copy()
    df_plot['Method'] = df_plot['Algorithm'].map(ALG_NAME_MAPPING)
    df_plot = df_plot[df_plot['Sigma'].isin(selected_sigmas)]

    g = sns.relplot(
        data=df_plot,
        x='Alternatives',
        y='MeanEuclideanDistance',
        col='Sigma',
        hue='Method',
        style='Method',
        kind='line',
        markers=MARKER_MAPPING,
        height=5,
        aspect=0.6,
        palette='tab10',
        legend="full",
        hue_order=ALG_ORDER,
        markersize=10,
        linewidth=2.5
    )

    g.fig.suptitle('Mean Euclidean Distance Across Methods', fontsize=16, y=1)
    g.set_titles(r"$\sigma$ = {col_name}", size=14)
    g.set_axis_labels("", "Mean Euclidean Distance", fontsize=14)
    g.fig.text(0.525, 0.0, 'Number of Alternatives',
               ha='center', va='center', fontsize=14, fontweight='bold')

    for ax in g.axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=13)
        sns.despine(ax=ax)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    g.set(xticks=df_plot['Alternatives'].unique())
    plt.setp(g._legend.get_texts(), fontsize=12)
    plt.setp(g._legend.get_title(), fontsize=14)
    g._legend.set_bbox_to_anchor((1.15, 0.5))
    plt.tight_layout(rect=[0, 0, 1, 1.02])

    if save_path:
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.tiff", dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_mean_kendall_line(performance_summary, selected_sigmas=[0.5, 1.5, 2.5],
                            save_path=None):
    """
    Plot Mean Kendall's Tau vs Number of Alternatives for each Sigma.
    
    Parameters
    ----------
    performance_summary : DataFrame
        From create_performance_summary()
    selected_sigmas : list
        Sigma values to include in plot
    save_path : str, optional
        Path to save figure (without extension). Saves both PDF and TIFF.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelweight': 'bold',
        'savefig.dpi': 300
    })

    df_plot = performance_summary.copy()
    df_plot['Method'] = df_plot['Algorithm'].map(ALG_NAME_MAPPING)
    df_plot = df_plot[df_plot['Sigma'].isin(selected_sigmas)]

    g = sns.relplot(
        data=df_plot,
        x='Alternatives',
        y='MeanKendallsTau',
        col='Sigma',
        hue='Method',
        style='Method',
        kind='line',
        markers=MARKER_MAPPING,
        height=5,
        aspect=0.6,
        palette='tab10',
        legend="full",
        hue_order=ALG_ORDER,
        markersize=10,
        linewidth=2.5
    )

    g.fig.suptitle("Mean Kendall's Tau Across Methods", fontsize=16, y=1)
    g.set_titles(r"$\sigma$ = {col_name}", size=14)
    g.set_axis_labels("", "Mean Kendall's Tau", fontsize=14)
    g.fig.text(0.525, 0.0, 'Number of Alternatives',
               ha='center', va='center', fontsize=14, fontweight='bold')

    for ax in g.axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=13)
        sns.despine(ax=ax)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    g.set(xticks=df_plot['Alternatives'].unique())
    plt.setp(g._legend.get_texts(), fontsize=12)
    plt.setp(g._legend.get_title(), fontsize=14)
    g._legend.set_bbox_to_anchor((1.15, 0.5))
    plt.tight_layout(rect=[0, 0, 1, 1.02])

    if save_path:
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.tiff", dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_violin_euclidean(df_performance, n_alternatives=5, sigma=1.5, save_path=None):
    """
    Plot violin plot of Euclidean Distance distribution for a specific condition.
    
    Parameters
    ----------
    df_performance : DataFrame
        Raw performance data
    n_alternatives : int
        Number of alternatives to filter
    sigma : float
        Sigma value to filter
    save_path : str, optional
        Path to save figure
    """
    sns.set_theme(style="whitegrid")
    
    df_plot = df_performance[
        (df_performance['Alternatives'] == n_alternatives) &
        (df_performance['Sigma'] == sigma)
    ].copy()
    df_plot['Method'] = df_plot['Algorithm'].map(ALG_NAME_MAPPING)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_plot, x='Method', y='EuclideanDistance',
                   order=ALG_ORDER, palette='Set2', ax=ax)
    
    ax.set_title(f'Euclidean Distance Distribution (n={n_alternatives}, σ={sigma})')
    ax.set_xlabel('')
    ax.set_ylabel('Euclidean Distance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_violin_kendall(df_performance, n_alternatives=5, sigma=1.5, save_path=None):
    """
    Plot violin plot of Kendall's Tau distribution for a specific condition.
    
    Parameters
    ----------
    df_performance : DataFrame
        Raw performance data
    n_alternatives : int
        Number of alternatives to filter
    sigma : float
        Sigma value to filter
    save_path : str, optional
        Path to save figure
    """
    sns.set_theme(style="whitegrid")
    
    df_plot = df_performance[
        (df_performance['Alternatives'] == n_alternatives) &
        (df_performance['Sigma'] == sigma)
    ].copy()
    df_plot['Method'] = df_plot['Algorithm'].map(ALG_NAME_MAPPING)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_plot, x='Method', y='KendallsTau',
                   order=ALG_ORDER, palette='Set2', ax=ax)
    
    ax.set_title(f"Kendall's Tau Distribution (n={n_alternatives}, σ={sigma})")
    ax.set_xlabel('')
    ax.set_ylabel("Kendall's Tau")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
