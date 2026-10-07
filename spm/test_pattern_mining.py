import numpy as np

from spm.temporal_patterns import discretize_dataset
from spm.pattern_mining import (
    count_dataset_support,
    get_frequent_patterns,
)


DATA_PATH = r"C:\Users\Nad Jr\Downloads\wlasl25_landmarks.npz"


def main():
    data = np.load(DATA_PATH)

    X_train = data["X_train_aug"]

    print("Original training shape:")
    print(X_train.shape)

    X_discretized = discretize_dataset(X_train)

    print("\nDiscretized training shape:")
    print(X_discretized.shape)

    support_counts = count_dataset_support(
        X_discretized,
        window_size=3
    )

    print("\nNumber of unique patterns:")
    print(len(support_counts))

    frequent_patterns = get_frequent_patterns(
        X_discretized,
        min_support=0.10,
        window_size=3
    )

    print("\nNumber of frequent patterns:")
    print(len(frequent_patterns))

    print("\nFirst 10 frequent patterns:")

    for pattern, support in list(frequent_patterns.items())[:10]:
        print(
            pattern,
            f"-> support: {support:.4f}"
        )


if __name__ == "__main__":
    main()