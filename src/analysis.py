"""
Statistical Analysis for IPCM Method Comparison

Friedman test, Conover post-hoc, Kendall's W effect size, and Compact Letter Display.
"""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from itertools import combinations

try:
    import scikit_posthocs as sp
    from scikit_posthocs import posthoc_conover_friedman
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("Warning: scikit-posthocs not installed. Install with: pip install scikit-posthocs")


# Algorithm name mapping
ALG_NAME_MAPPING = {
    'create_incomplete_pcm_AHP_express': 'AHP Express',
    'create_incomplete_pcm_star': 'Star',
    'create_incomplete_pcm_cycle': 'Cycle',
    'create_incomplete_pcm_tournament': 'C-AHP',
    'GeometricMean': 'Benchmark',
    'create_incomplete_pcm_tournament_champion_closure': 'CC-AHP'
}

EXCLUDE_ALGORITHMS = ['Benchmark', 'AHP Express Second']


def calculate_kendalls_w(chi_square, n, k):
    """Kendall's W (Coefficient of Concordance): W = chi^2 / [n(k-1)]"""
    return chi_square / (n * (k - 1))


def interpret_kendalls_w(w):
    """Return interpretation of Kendall's W"""
    if w < 0.1:
        return "negligible"
    elif w < 0.3:
        return "weak"
    elif w < 0.5:
        return "moderate"
    elif w < 0.7:
        return "strong"
    else:
        return "very strong"
        
def generate_compact_letter_display(algorithms, conover_matrix, alpha=0.05):
    """
    Generate Compact Letter Display (CLD) from Conover post-hoc results.
    Based on Piepho (2004) method.
    """
    n_algs = len(algorithms)
    display = np.ones((n_algs, 1), dtype=bool)

    sig_pairs = []
    for i in range(n_algs):
        for j in range(i+1, n_algs):
            if conover_matrix.loc[algorithms[i], algorithms[j]] < alpha:
                sig_pairs.append((i, j))

    for i, j in sig_pairs:
        for col_idx in range(display.shape[1]):
            if display[i, col_idx] and display[j, col_idx]:
                new_col1 = display[:, col_idx].copy()
                new_col2 = display[:, col_idx].copy()
                new_col1[i] = False
                new_col2[j] = False
                display = np.column_stack([display, new_col1, new_col2])
                display = np.delete(display, col_idx, axis=1)

                cols_to_keep = []
                for c1 in range(display.shape[1]):
                    is_subset = False
                    for c2 in range(display.shape[1]):
                        if c1 != c2:
                            if np.all(display[:, c1] <= display[:, c2]) and np.any(display[:, c1] < display[:, c2]):
                                is_subset = True
                                break
                    if not is_subset:
                        cols_to_keep.append(c1)
                display = display[:, cols_to_keep]
                break

    letters = 'abcdefghijklmnopqrstuvwxyz'
    cld = {}
    for i, alg in enumerate(algorithms):
        alg_letters = ''.join([letters[j] for j in range(display.shape[1]) if display[i, j]])
        cld[alg] = alg_letters if alg_letters else 'a'

    return cld


def run_analysis(df_performance, metric='EuclideanDistance', alpha=0.05):
    """
    Run complete statistical analysis: Friedman test, Kendall's W, Conover post-hoc, CLD.
    
    Parameters
    ----------
    df_performance : DataFrame
        Must have columns: Sigma, Alternatives, Algorithm, and the metric column
    metric : str
        'EuclideanDistance' or 'KendallsTau'
    alpha : float
        Significance level
    
    Returns
    -------
    results : list of dict
        Analysis results for each (Sigma, Alternatives) condition
    """
    if not HAS_POSTHOCS:
        raise ImportError("scikit-posthocs required. Install with: pip install scikit-posthocs")
    
    # Map algorithm names
    df = df_performance.copy()
    df['Algorithm_Short'] = df['Algorithm'].map(ALG_NAME_MAPPING)
    
    # Exclude benchmark and other algorithms
    df_filtered = df[~df['Algorithm_Short'].isin(EXCLUDE_ALGORITHMS)].copy()
    
    # Add RunID for paired comparisons
    df_filtered['RunID'] = df_filtered.groupby(
        ['Sigma', 'Alternatives', 'Algorithm_Short']
    ).cumcount()
    
    results = []
    
    for sigma_value in sorted(df_filtered['Sigma'].unique()):
        for alt_value in sorted(df_filtered['Alternatives'].unique()):
            subset = df_filtered[
                (df_filtered['Sigma'] == sigma_value) &
                (df_filtered['Alternatives'] == alt_value)
            ]
            
            # Pivot for Friedman test
            df_pivot = subset.pivot(
                index='RunID', 
                columns='Algorithm_Short', 
                values=metric
            ).dropna()
            
            if df_pivot.shape[0] < 2 or df_pivot.shape[1] < 2:
                continue
            
            # Friedman test
            stat, p = friedmanchisquare(*[df_pivot[col] for col in df_pivot.columns])
            
            # Kendall's W effect size
            n, k = df_pivot.shape
            w = calculate_kendalls_w(stat, n, k)
            w_interp = interpret_kendalls_w(w)
            
            # Mean performance
            mean_perf = df_pivot.mean()
            if metric == 'EuclideanDistance':
                mean_perf = mean_perf.sort_values(ascending=True)  # Lower is better
            else:
                mean_perf = mean_perf.sort_values(ascending=False)  # Higher is better
            best_algo = mean_perf.index[0]
            
            # Conover post-hoc and CLD if significant
            cld = None
            conover_matrix = None
            if p < alpha:
                df_long = df_pivot.reset_index().melt(
                    id_vars='RunID', 
                    var_name='Algorithm', 
                    value_name=metric
                )
                conover_matrix = sp.posthoc_conover(
                    df_long, 
                    val_col=metric, 
                    group_col='Algorithm', 
                    p_adjust='holm'
                )
                algorithms = list(df_pivot.columns)
                cld = generate_compact_letter_display(algorithms, conover_matrix, alpha)
            
            results.append({
                'Sigma': sigma_value,
                'Alternatives': alt_value,
                'Metric': metric,
                'Friedman_stat': stat,
                'Friedman_p': p,
                'Kendalls_W': w,
                'W_interpretation': w_interp,
                'Best_Algorithm': best_algo,
                'Mean_Performance': mean_perf.to_dict(),
                'CLD': cld,
                'Conover_matrix': conover_matrix,
                'n_samples': n
            })
    
    return results


def print_summary_table(results, metric='EuclideanDistance'):
    """Print a formatted summary table of analysis results."""
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS SUMMARY: {metric}")
    print(f"{'='*80}")
    
    for r in results:
        print(f"\nσ={r['Sigma']}, n={r['Alternatives']}")
        print(f"  Friedman χ²={r['Friedman_stat']:.2f}, p={'<0.001' if r['Friedman_p'] < 0.001 else f'{r[\"Friedman_p\"]:.4f}'}")
        print(f"  Kendall's W={r['Kendalls_W']:.3f} ({r['W_interpretation']})")
        print(f"  Best: {r['Best_Algorithm']}")
        if r['CLD']:
            print(f"  CLD: {r['CLD']}")


def save_summary_csv(results, filepath):
    """Save analysis results to CSV."""
    rows = []
    for r in results:
        row = {
            'Sigma': r['Sigma'],
            'Alternatives': r['Alternatives'],
            'Metric': r['Metric'],
            'Friedman_stat': r['Friedman_stat'],
            'Friedman_p': r['Friedman_p'],
            'Kendalls_W': r['Kendalls_W'],
            'W_interpretation': r['W_interpretation'],
            'Best_Algorithm': r['Best_Algorithm'],
            'n_samples': r['n_samples']
        }
        # Add CLD columns
        if r['CLD']:
            for alg, letters in r['CLD'].items():
                row[f'CLD_{alg}'] = letters
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def print_manuscript_summary(results_euclidean, results_kendall):
    """Print summary formatted for manuscript inclusion."""
    print("\n" + "="*80)
    print("MANUSCRIPT SUMMARY")
    print("="*80)
    
    # Euclidean summary
    print("\nEuclidean Distance Analysis:")
    w_values = [r['Kendalls_W'] for r in results_euclidean]
    print(f"  Kendall's W range: [{min(w_values):.3f}, {max(w_values):.3f}]")
    
    best_counts = {}
    for r in results_euclidean:
        best = r['Best_Algorithm']
        best_counts[best] = best_counts.get(best, 0) + 1
    print(f"  Best algorithm frequency: {best_counts}")
    
    # Kendall's tau summary
    print("\nKendall's Tau Analysis:")
    w_values = [r['Kendalls_W'] for r in results_kendall]
    print(f"  Kendall's W range: [{min(w_values):.3f}, {max(w_values):.3f}]")
    
    best_counts = {}
    for r in results_kendall:
        best = r['Best_Algorithm']
        best_counts[best] = best_counts.get(best, 0) + 1
    print(f"  Best algorithm frequency: {best_counts}")
