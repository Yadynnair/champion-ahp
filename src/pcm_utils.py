"""
Pairwise Comparison Matrix Utilities

Functions for generating consistent PCMs, adding noise/inconsistency,
computing consistency ratios, and deriving weights via LLSM.
"""

import numpy as np
import networkx as nx
from scipy.stats import gmean
from collections import deque


def generate_weights_with_max_ratio(n_alternatives):
    """
    Generate positive weights for n_alternatives by randomly selecting
    weights from a uniform distribution between 1 and 9, and then normalizing them.
    """
    weights = np.random.uniform(1, 9, size=n_alternatives)
    weights /= weights.sum()
    return weights


def generate_consistent_PCM(n_alternatives):
    """
    Generate a consistent pairwise comparison matrix using weights generated
    from a uniform distribution between 1 and 9.
    Returns both the matrix and the weights used.
    """
    weights = generate_weights_with_max_ratio(n_alternatives)
    matrix = np.zeros((n_alternatives, n_alternatives))
    for i in range(n_alternatives):
        for j in range(n_alternatives):
            matrix[i, j] = weights[i] / weights[j]
    return matrix, weights


def add_noise_linear_scale(pcm, sigma, clip_to_saaty=True):
    """
    Add noise in linear scale perspective: x' = x * 9^(eta/10), eta ~ N(0, sigma).
    Simulates decision-makers thinking in linear 0-10 scale converted to Saaty's 1-9 scale.
    Clips output values to Saaty's [1/9, 9] range.
    """
    n = pcm.shape[0]
    noisy_pcm = np.ones_like(pcm, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            original_value = pcm[i, j]
            is_reciprocal = original_value < 1

            eta = np.random.normal(0, sigma)

            if not is_reciprocal:
                 noisy_value = original_value * (9 ** (eta / 10))
            else:
                 noisy_value = pcm[j, i] * (9 ** (eta / 10))

            if clip_to_saaty:
                noisy_value = min(max(noisy_value, 1), 9)

            if not is_reciprocal:
                noisy_pcm[i, j] = noisy_value
                noisy_pcm[j, i] = 1 / noisy_value
            else:
                noisy_pcm[j, i] = noisy_value
                noisy_pcm[i, j] = 1 / noisy_value

    np.fill_diagonal(noisy_pcm, 1)

    return noisy_pcm


def calculate_consistency_ratio(pcm):
    """
    Calculate Saaty's consistency ratio for a PCM.
    """
    n = pcm.shape[0]
    if n < 3:
        return 0.0

    RI = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49}

    try:
        eigenvals = np.linalg.eigvals(pcm)
        lambda_max = np.max(np.real(eigenvals))
        CI = (lambda_max - n) / (n - 1)
        CR = CI / RI.get(n, 1.49)
        return CR
    except:
        return float('inf')


def find_all_spanning_trees(graph):
    """
    Recursively finds all spanning trees of an undirected graph.
    Uses a recursive approach by building trees edge by edge.
    """
    nodes = list(graph.nodes())
    edges = list(graph.edges(data=True))
    n = len(nodes)

    if len(edges) == n - 1:
        if nx.is_connected(graph):
            return [graph]
        else:
            return []

    def recursive_spanning_tree_builder(current_graph, remaining_edges):
        if len(current_graph.edges()) == n - 1:
            if nx.is_connected(current_graph):
                return [current_graph]
            else:
                 return []

        if not remaining_edges:
            return []

        u, v, data = remaining_edges[0]
        rest_of_edges = remaining_edges[1:]

        trees_excluding = recursive_spanning_tree_builder(current_graph.copy(), rest_of_edges)

        temp_graph = current_graph.copy()
        temp_graph.add_edge(u, v, **data)

        try:
            if nx.has_path(current_graph, u, v):
                 trees_including = []
            else:
                 trees_including = recursive_spanning_tree_builder(temp_graph, rest_of_edges)
        except nx.NetworkXNoPath:
             trees_including = recursive_spanning_tree_builder(temp_graph, rest_of_edges)
        except Exception as e:
             print(f"Error checking for cycle: {e}")
             trees_including = []

        return trees_excluding + trees_including

    empty_graph_with_nodes = nx.Graph()
    empty_graph_with_nodes.add_nodes_from(nodes)
    return recursive_spanning_tree_builder(empty_graph_with_nodes, edges)


def geometric_mean_incomplete_pcm_spanning_trees(incomplete_pcm):
    """
    Calculate local priorities (weights) from an incomplete PCM using the geometric mean
    of weight vectors derived from all spanning trees of the comparison graph.
    Weight vectors for each spanning tree are calculated by propagating weights from
    a reference node (node 0), normalized afterwards.
    """
    n = incomplete_pcm.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i != j and incomplete_pcm[i, j] != 0:
                G.add_edge(i, j, comparison=incomplete_pcm[i, j])

    G_undirected = G.to_undirected()
    if not nx.is_connected(G_undirected):
        print("Warning: The comparison graph is not connected. Cannot find spanning trees.")
        return np.zeros(n)

    try:
        spanning_trees_graphs = find_all_spanning_trees(G_undirected)
    except Exception as e:
        print(f"Error finding spanning trees: {e}")
        spanning_trees_graphs = []

    if not spanning_trees_graphs:
        print("Warning: No spanning trees found. Cannot calculate weights.")
        return np.zeros(n)

    spanning_tree_weights = []

    for T in spanning_trees_graphs:
        if 0 not in T.nodes():
             reference_node = list(T.nodes())[0]
        else:
             reference_node = 0

        tree_weight_vector = np.zeros(n)
        tree_weight_vector[reference_node] = 1.0

        visited = {reference_node}
        queue = deque([reference_node])

        while queue:
            u = queue.popleft()

            for v in T.neighbors(u):
                if v not in visited:
                    comparison_value_uv = incomplete_pcm[u, v]

                    if comparison_value_uv != 0:
                        tree_weight_vector[v] = tree_weight_vector[u] * (1.0 / comparison_value_uv)

                    visited.add(v)
                    queue.append(v)

        weights_sum = np.sum(tree_weight_vector)
        if weights_sum > 0:
            tree_weight_vector /= weights_sum
        else:
            tree_weight_vector = np.zeros(n)

        spanning_tree_weights.append(tree_weight_vector)

    if not spanning_tree_weights:
        print("Warning: No weight vectors generated from spanning trees.")
        return np.zeros(n)

    spanning_tree_weights_array = np.array(spanning_tree_weights)

    try:
        geometric_mean_weights = gmean(spanning_tree_weights_array, axis=0)
    except ValueError as e:
        print(f"Error calculating geometric mean of weights: {e}")
        return np.zeros(n)

    final_weights_sum = np.sum(geometric_mean_weights)
    if final_weights_sum > 0:
        normalized_final_weights = geometric_mean_weights / final_weights_sum
    else:
        print("Warning: Sum of geometric mean weights is zero. Returning zeros.")
        normalized_final_weights = np.zeros(n)

    return normalized_final_weights


def llsm_complete_pcm(pcm):
    """
    Compute weights from a complete PCM using LLSM (geometric mean of rows).
    """
    n = pcm.shape[0]
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = np.prod(pcm[i, :]) ** (1.0 / n)
    weights /= weights.sum()
    return weights
