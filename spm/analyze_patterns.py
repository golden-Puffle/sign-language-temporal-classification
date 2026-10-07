import numpy as np

from spm.temporal_patterns import discretize_dataset
from spm.pattern_mining import get_frequent_patterns
from spm.feature_selection import build_pattern_matrix


DATA_PATH = r"C:\Users\Nad Jr\Downloads\wlasl25_landmarks.npz"


def main():
    data = np.load(DATA_PATH)

    X_train = data["X_train_aug"]
    y_train = data["y_train_aug"]

    print("Training data:")
    print(X_train.shape)

    # ---------------------------------------------------------
    # 1. Discretize training data
    # ---------------------------------------------------------
    X_discretized = discretize_dataset(X_train)

    # ---------------------------------------------------------
    # 2. Mine the same frequent patterns as our best model
    # ---------------------------------------------------------
    frequent_patterns = get_frequent_patterns(
        X_discretized,
        min_support=0.10,
        window_size=5,
    )

    patterns = list(frequent_patterns.keys())

    print("\nFrequent patterns:")
    print(len(patterns))

    # ---------------------------------------------------------
    # 3. Build binary pattern matrix
    # ---------------------------------------------------------
    X_pattern = build_pattern_matrix(
        X_discretized,
        patterns,
    )

    print("\nPattern matrix:")
    print(X_pattern.shape)

    # ---------------------------------------------------------
    # 4. Analyze each pattern
    # ---------------------------------------------------------
    class_counts = np.zeros(
        (len(patterns), 25),
        dtype=np.int32,
    )

    for pattern_index in range(len(patterns)):

        samples_with_pattern = (
            X_pattern[:, pattern_index] == 1
        )

        labels_with_pattern = y_train[
            samples_with_pattern
        ]

        for label in labels_with_pattern:
            class_counts[
                pattern_index,
                int(label)
            ] += 1

    # ---------------------------------------------------------
    # 5. Calculate statistics
    # ---------------------------------------------------------
    pattern_support = (
        X_pattern.sum(axis=0)
    )

    class_coverage = np.sum(
        class_counts > 0,
        axis=1,
    )

    # ---------------------------------------------------------
    # 6. Print overall statistics
    # ---------------------------------------------------------
    print("\n===== PATTERN DIAGNOSTICS =====")

    print(
        "\nPatterns appearing in only one class:"
    )
    print(
        np.sum(class_coverage == 1)
    )

    print(
        "Patterns appearing in 2-5 classes:"
    )
    print(
        np.sum(
            (class_coverage >= 2)
            & (class_coverage <= 5)
        )
    )

    print(
        "Patterns appearing in 6-10 classes:"
    )
    print(
        np.sum(
            (class_coverage >= 6)
            & (class_coverage <= 10)
        )
    )

    print(
        "Patterns appearing in 11-20 classes:"
    )
    print(
        np.sum(
            (class_coverage >= 11)
            & (class_coverage <= 20)
        )
    )

    print(
        "Patterns appearing in 21-25 classes:"
    )
    print(
        np.sum(
            class_coverage >= 21
        )
    )

    # ---------------------------------------------------------
    # 7. Most class-specific patterns
    # ---------------------------------------------------------
    most_specific = np.argsort(
        class_coverage
    )

    print(
        "\n===== MOST CLASS-SPECIFIC PATTERNS ====="
    )

    shown = 0

    for index in most_specific:

        if class_coverage[index] > 5:
            break

        print(
            patterns[index],
            f"-> support: {pattern_support[index]}",
            f"classes: {class_coverage[index]}",
        )

        shown += 1

        if shown >= 20:
            break

    # ---------------------------------------------------------
    # 8. Most widespread patterns
    # ---------------------------------------------------------
    most_widespread = np.argsort(
        class_coverage
    )[::-1]

    print(
        "\n===== MOST WIDESPREAD PATTERNS ====="
    )

    for index in most_widespread[:20]:

        print(
            patterns[index],
            f"-> support: {pattern_support[index]}",
            f"classes: {class_coverage[index]}",
        )


if __name__ == "__main__":
    main()