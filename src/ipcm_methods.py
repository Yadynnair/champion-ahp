"""
Incomplete Pairwise Comparison Matrix (IPCM) Construction Methods

Original implementation from the paper:
Sokantika & Ratnapinda (2025). The Role of Ranking Information in AHP
with Limited Pairwise Comparisons.
"""

import numpy as np


def create_incomplete_pcm_AHP_express(pcm, weights):
    """
    Create an incomplete PCM using the AHP Express method: select the row
    of alternatives that has a highest weight from the PCM.
    The diagonal is 1, the selected row's non-diagonal elements are kept,
    the corresponding reciprocal column's non-diagonal elements are kept,
    and all other elements are set to 0.
    """
    n = pcm.shape[0]
    incomplete_pcm = np.zeros_like(pcm)

    # Find the index of the alternative with the highest weight
    highest_weight_index = np.argmax(weights)

    # Set the diagonal elements to 1
    np.fill_diagonal(incomplete_pcm, 1)

    # Keep the row of the highest weight alternative
    incomplete_pcm[highest_weight_index, :] = pcm[highest_weight_index, :]

    # Keep the corresponding reciprocal column
    for i in range(n):
        if i != highest_weight_index:
            incomplete_pcm[i, highest_weight_index] = pcm[i, highest_weight_index]

    return incomplete_pcm


def create_incomplete_pcm_star(pcm):
    """
    Create an incomplete PCM using the Star method: randomly select a row
    to keep from the PCM in a uniformly distributed manner.
    The diagonal is 1, the selected row's non-diagonal elements are kept,
    the corresponding reciprocal column's non-diagonal elements are kept,
    and all other elements are set to 0.
    """
    n = pcm.shape[0]
    incomplete_pcm = np.zeros_like(pcm)

    # Randomly select an index (row) to keep
    selected_index = np.random.randint(0, n)

    # Set the diagonal elements to 1
    np.fill_diagonal(incomplete_pcm, 1)

    # Keep the row of the selected alternative
    incomplete_pcm[selected_index, :] = pcm[selected_index, :]

    # Keep the corresponding reciprocal column
    for i in range(n):
        if i != selected_index:
            incomplete_pcm[i, selected_index] = pcm[i, selected_index]

    return incomplete_pcm


def create_incomplete_pcm_cycle(pcm):
    """
    Create an incomplete PCM based on a random cyclic permutation of alternatives.
    Keeps comparison values (and their reciprocals) in the cyclic order.
    The diagonal is 1, and all other elements are 0.
    """
    n = pcm.shape[0]
    incomplete_pcm = np.zeros_like(pcm)

    # Generate a random permutation of alternative indices
    permutation = np.random.permutation(n)

    # Set the diagonal elements to 1
    np.fill_diagonal(incomplete_pcm, 1)

    # Keep the comparison values (and their reciprocals) in the cyclic order
    for i in range(n):
        current_alt = permutation[i]
        next_alt = permutation[(i + 1) % n]

        incomplete_pcm[current_alt, next_alt] = pcm[current_alt, next_alt]
        incomplete_pcm[next_alt, current_alt] = pcm[next_alt, current_alt]

    return incomplete_pcm


def create_incomplete_pcm_tournament(pcm):
    """
    Create an incomplete PCM based on a tournament among alternatives.
    A random permutation of alternatives is computed. Alternatives compare
    in this order, and the alternative judged "better" according to the PCM value
    proceeds to the next comparison until all alternatives have been compared.
    Keeps only the comparison values (and their reciprocals) that occurred
    during the tournament. The diagonal is 1, and all other elements are 0.
    """
    n = pcm.shape[0]
    incomplete_pcm = np.zeros_like(pcm)

    # Generate a random permutation of alternative indices
    permutation = np.random.permutation(n)

    # Set the diagonal elements to 1
    np.fill_diagonal(incomplete_pcm, 1)

    # Simulate the tournament
    current_winner_index = permutation[0]

    for i in range(1, n):
        next_challenger_index = permutation[i]

        comparison_value = pcm[current_winner_index, next_challenger_index]

        if comparison_value > 1:
            winner_index = current_winner_index
            loser_index = next_challenger_index
        elif comparison_value < 1:
            winner_index = next_challenger_index
            loser_index = current_winner_index
        else:
            winner_index = np.random.choice([current_winner_index, next_challenger_index])
            loser_index = current_winner_index if winner_index == next_challenger_index else next_challenger_index

        incomplete_pcm[winner_index, loser_index] = pcm[winner_index, loser_index]
        incomplete_pcm[loser_index, winner_index] = pcm[loser_index, winner_index]

        current_winner_index = winner_index

    return incomplete_pcm


def create_incomplete_pcm_tournament_champion_closure(pcm):
    """
    Create an incomplete PCM based on a tournament among alternatives, with a random permutation.
    Alternatives compare in this order, and the alternative judged "better" according to the PCM value proceeds.
    After the tournament, adds a comparison between the first alternative that dropped out of the tournament
    and the final winner alternative.
    Keeps only the comparisons that occurred during the tournament (and their reciprocals) plus the added comparison.
    The diagonal is 1, and all other elements are 0.
    """
    n = pcm.shape[0]
    incomplete_pcm = np.zeros_like(pcm)

    # Generate a random permutation of alternative indices
    permutation = np.random.permutation(n)

    # Set the diagonal elements to 1
    np.fill_diagonal(incomplete_pcm, 1)

    # Simulate the tournament and keep track of the first eliminated alternative
    current_winner_index = permutation[0]
    first_eliminated_index = None

    for i in range(1, n):
        next_challenger_index = permutation[i]

        comparison_value = pcm[current_winner_index, next_challenger_index]

        if comparison_value > 1:
            winner_index = current_winner_index
            loser_index = next_challenger_index
        elif comparison_value < 1:
            winner_index = next_challenger_index
            loser_index = current_winner_index
        else:
            winner_index = np.random.choice([current_winner_index, next_challenger_index])
            loser_index = current_winner_index if winner_index == next_challenger_index else next_challenger_index

        # If this is the first comparison (i=1), the loser is the first eliminated alternative
        if i == 1:
            first_eliminated_index = loser_index

        incomplete_pcm[winner_index, loser_index] = pcm[winner_index, loser_index]
        incomplete_pcm[loser_index, winner_index] = pcm[loser_index, winner_index]

        current_winner_index = winner_index

    final_winner_index = current_winner_index

    # Add a comparison between the first eliminated alternative and the final winner
    if first_eliminated_index is not None and first_eliminated_index != final_winner_index:
        incomplete_pcm[first_eliminated_index, final_winner_index] = pcm[first_eliminated_index, final_winner_index]
        incomplete_pcm[final_winner_index, first_eliminated_index] = pcm[final_winner_index, first_eliminated_index]

    return incomplete_pcm
