import numpy as np

from spm.temporal_patterns import discretize_dataset
from spm.pattern_mining import get_frequent_patterns
from spm.feature_selection import (
    build_pattern_matrix,
    select_patterns_chi2,
)


DATA_PATH = r"C:\Users\Nad Jr\Downloads\wlasl25_landmarks.npz"


def main():
    data = np.load(DATA_PATH)

    X_train = data["X_train_aug"]
    y_train = data["y_train_aug"]

    print("Training data:")
    print(X_train.shape)

    X_discretized = discretize_dataset(
        X_train
    )

    frequent_patterns = get_frequent_patterns(
        X_discretized,
        min_support=0.10,
        window_size=5,
    )

    patterns = list(
        frequent_patterns.keys()
    )

    print("\nFrequent patterns:")
    print(len(patterns))

    X_pattern = build_pattern_matrix(
        X_discretized,
        patterns,
    )

    print("\nPattern matrix:")
    print(X_pattern.shape)

    (
        X_selected,
        selected_patterns,
        scores,
    ) = select_patterns_chi2(
        X_pattern,
        y_train,
        patterns,
        top_k=100,
    )

    print("\nSelected feature matrix:")
    print(X_selected.shape)

    print("\nSelected patterns:")
    for pattern, score in zip(
        selected_patterns[:10],
        scores[:10],
    ):
        print(
            pattern,
            f"-> chi2: {score:.4f}"
        )


if __name__ == "__main__":
    main()