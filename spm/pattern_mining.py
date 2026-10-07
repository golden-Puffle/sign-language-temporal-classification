import numpy as np


def extract_patterns(sequence, window_size=3):
    """
    Extract short temporal patterns from a discretized sequence.

    Each pattern describes the temporal direction of one feature
    across consecutive time steps.

    Parameters
    ----------
    sequence : np.ndarray
        Shape: (time_steps, num_features)
        Values should be -1, 0, or 1.

    window_size : int
        Number of consecutive time steps used to form a pattern.

    Returns
    -------
    list
        List of (feature_index, pattern) tuples.
    """
    sequence = np.asarray(sequence)

    if sequence.ndim != 2:
        raise ValueError(
            f"Expected a 2D sequence, got shape {sequence.shape}"
        )

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    time_steps, num_features = sequence.shape

    if window_size > time_steps:
        raise ValueError(
            f"window_size ({window_size}) cannot exceed "
            f"time_steps ({time_steps})."
        )

    patterns = []

    for feature_index in range(num_features):
        feature_values = sequence[:, feature_index]

        for start in range(time_steps - window_size + 1):
            window = feature_values[start:start + window_size]
            pattern = tuple(int(value) for value in window)

            patterns.append((feature_index, pattern))

    return patterns


def count_patterns(sequence, window_size=3):
    """
    Count pattern occurrences within one sequence.
    """
    patterns = extract_patterns(
        sequence,
        window_size=window_size
    )

    counts = {}

    for pattern in patterns:
        counts[pattern] = counts.get(pattern, 0) + 1

    return counts


def count_dataset_support(X_discretized, window_size=3):
    """
    Count how many sequences contain each temporal pattern.

    A pattern contributes at most once per sequence. This means
    the result measures sequence-level support rather than the
    total number of times a pattern appears.

    Parameters
    ----------
    X_discretized : np.ndarray
        Shape: (samples, time_steps, num_features)

    window_size : int
        Number of consecutive time steps per pattern.

    Returns
    -------
    dict
        Mapping:
            (feature_index, pattern) -> number of sequences
            containing that pattern.
    """
    X_discretized = np.asarray(X_discretized)

    if X_discretized.ndim != 3:
        raise ValueError(
            f"Expected a 3D dataset, got shape {X_discretized.shape}"
        )

    support_counts = {}

    for sequence in X_discretized:
        patterns = extract_patterns(
            sequence,
            window_size=window_size
        )

        # Count each pattern only once for this sequence.
        unique_patterns = set(patterns)

        for pattern in unique_patterns:
            support_counts[pattern] = (
                support_counts.get(pattern, 0) + 1
            )

    return support_counts


def get_frequent_patterns(
    X_discretized,
    min_support=0.10,
    window_size=5,
):
    """
    Return patterns whose sequence-level support meets min_support.

    Parameters
    ----------
    X_discretized : np.ndarray
        Shape: (samples, time_steps, num_features)

    min_support : float
        Minimum fraction of sequences containing the pattern.
        Must be between 0 and 1.

    window_size : int
        Number of consecutive time steps per pattern.

    Returns
    -------
    dict
        Mapping:
            (feature_index, pattern) -> support fraction
    """
    if not 0 < min_support <= 1:
        raise ValueError(
            "min_support must be greater than 0 and at most 1."
        )

    support_counts = count_dataset_support(
        X_discretized,
        window_size=window_size
    )

    num_sequences = len(X_discretized)

    frequent_patterns = {
        pattern: count / num_sequences
        for pattern, count in support_counts.items()
        if count / num_sequences >= min_support
    }

    return frequent_patterns