"""
Champion AHP: Incomplete Pairwise Comparison Methods for AHP
"""

from .ipcm_methods import (
    create_incomplete_pcm_AHP_express,
    create_incomplete_pcm_star,
    create_incomplete_pcm_cycle,
    create_incomplete_pcm_tournament,
    create_incomplete_pcm_tournament_champion_closure,
)

from .pcm_utils import (
    generate_weights_with_max_ratio,
    generate_consistent_PCM,
    add_noise_linear_scale,
    calculate_consistency_ratio,
    geometric_mean_incomplete_pcm_spanning_trees,
    llsm_complete_pcm,
)

from .visualization import (
    plot_violin_euclidean,
    plot_violin_kendall,
    plot_mean_euclidean_line,
    plot_mean_kendall_line,
    create_performance_summary,
    ALG_NAME_MAPPING,
)

from .analysis import (
    run_analysis,
    print_summary_table,
    save_summary_csv,
    print_manuscript_summary,
    calculate_kendalls_w,
    interpret_kendalls_w,
)

__version__ = '1.0.0'
__author__ = 'Saronsad Sokantika, Parot Ratnapinda'
