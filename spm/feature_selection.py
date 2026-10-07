import numpy as np
from sklearn.feature_selection import chi2


def build_pattern_matrix(X_discretized, patterns, window_size=3):
    X_discretized = np.asarray(X_discretized)

    pattern_to_index = {
        pattern: index
        for index, pattern in enumerate(patterns)
    }

    X_pattern = np.zeros(
        (len(X_discretized), len(patterns)),
        dtype=np.int8
    )

    for sample_index, sequence in enumerate(X_discretized):
        sequence_patterns = set()

        time_steps, num_features = sequence.shape

        for feature_index in range(num_features):
            feature_values = sequence[:, feature_index]

            for start in range(time_steps - window_size + 1):
                window = feature_values[
                    start:start + window_size
                ]

                pattern = (
                    feature_index,
                    tuple(int(value) for value in window)
                )

                sequence_patterns.add(pattern)

        for pattern in sequence_patterns:
            pattern_index = pattern_to_index.get(pattern)

            if pattern_index is not None:
                X_pattern[sample_index, pattern_index] = 1

    return X_pattern


def select_patterns_chi2(
    X_pattern,
    y,
    patterns,
    top_k=2157,
):
    """
    Select the most class-relevant temporal patterns
    using the chi-square test.

    Parameters
    ----------
    X_pattern : np.ndarray
        Binary pattern matrix.

    y : np.ndarray
        Class labels.

    patterns : list
        Pattern corresponding to each column of X_pattern.

    top_k : int
        Number of patterns to retain.

    Returns
    -------
    selected_matrix : np.ndarray
        Matrix containing only selected patterns.

    selected_patterns : list
        Selected temporal patterns.

    scores : np.ndarray
        Chi-square score for each selected pattern.
    """
    X_pattern = np.asarray(X_pattern)
    y = np.asarray(y)

    if X_pattern.ndim != 2:
        raise ValueError(
            "X_pattern must be a 2D matrix."
        )

    if len(X_pattern) != len(y):
        raise ValueError(
            "X_pattern and y must contain the same "
            "number of samples."
        )

    if len(patterns) != X_pattern.shape[1]:
        raise ValueError(
            "Number of patterns must match "
            "number of feature columns."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    top_k = min(top_k, X_pattern.shape[1])

    chi_scores, p_values = chi2(
        X_pattern,
        y,
    )

    # Replace invalid scores with zero.
    chi_scores = np.nan_to_num(
        chi_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    ranked_indices = np.argsort(
        chi_scores
    )[::-1]

    selected_indices = ranked_indices[:top_k]

    selected_matrix = X_pattern[
        :,
        selected_indices
    ]

    selected_patterns = [
        patterns[index]
        for index in selected_indices
    ]

    selected_scores = chi_scores[
        selected_indices
    ]

    return (
        selected_matrix,
        selected_patterns,
        selected_scores,
    )