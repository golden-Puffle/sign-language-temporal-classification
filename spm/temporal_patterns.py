import numpy as np


def discretize_sequence(sequence):
    """
    Convert a continuous time-series sequence into temporal states.

    Each feature at each time step is classified as:
        -1 = decreasing
         0 = steady
         1 = increasing

    Parameters
    ----------
    sequence : np.ndarray
        Shape: (time_steps, num_features)

    Returns
    -------
    np.ndarray
        Discretized sequence with the same shape.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(
            f"Expected a 2D sequence, got shape {sequence.shape}"
        )

    differences = np.diff(sequence, axis=0)

    states = np.zeros_like(differences, dtype=np.int8)

    states[differences > 0] = 1
    states[differences < 0] = -1

    # Keep the first frame as steady because it has no previous frame.
    first_frame = np.zeros(
        (1, sequence.shape[1]),
        dtype=np.int8
    )

    discretized = np.vstack([first_frame, states])

    return discretized


def discretize_dataset(X):
    """
    Discretize a dataset of temporal sequences.

    Parameters
    ----------
    X : np.ndarray
        Shape: (samples, time_steps, num_features)

    Returns
    -------
    np.ndarray
        Discretized dataset with the same shape as X.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected a 3D dataset, got shape {X.shape}"
        )

    return np.array([
        discretize_sequence(sequence)
        for sequence in X
    ])