import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from spm.temporal_patterns import discretize_dataset
from spm.pattern_mining import get_frequent_patterns
from spm.feature_selection import build_pattern_matrix, select_patterns_chi2


def load_dataset(data_path):
    data = np.load(data_path)

    X_train = data["X_train_aug"]
    y_train = data["y_train_aug"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    print("Dataset loaded successfully.")
    print(f"X_train_aug: {X_train.shape}")
    print(f"y_train_aug: {y_train.shape}")
    print(f"X_val:       {X_val.shape}")
    print(f"y_val:       {y_val.shape}")

    return X_train, y_train, X_val, y_val


def fit_spm_features(
    X_train,
    y_train,
    min_support=0.10,
    window_size=5,
    top_k=10000,
):
    print("\n===== SPM FEATURE MINING =====")

    print("Discretizing training data...")
    X_train_discretized = discretize_dataset(X_train)

    print(f"Mining frequent patterns...")
    print(f"Minimum support: {min_support}")
    print(f"Window size: {window_size}")

    frequent_patterns = get_frequent_patterns(
        X_train_discretized,
        min_support=min_support,
        window_size=window_size,
    )

    patterns = list(frequent_patterns.keys())

    print(f"Frequent patterns found: {len(patterns)}")

    print("\nBuilding training pattern matrix...")

    X_train_pattern = build_pattern_matrix(
        X_train_discretized,
        patterns,
        window_size=window_size,
    )

    print(f"Training pattern matrix: {X_train_pattern.shape}")

    print("\nSelecting patterns using chi-square...")

    selected_top_k = min(top_k, len(patterns))

    X_train_selected, selected_patterns, selected_scores = (
        select_patterns_chi2(
            X_train_pattern,
            y_train,
            patterns,
            top_k=selected_top_k,
        )
    )

    print(f"Selected patterns: {len(selected_patterns)}")
    print(f"Selected training features: {X_train_selected.shape}")

    return (
        X_train_selected,
        selected_patterns,
        frequent_patterns,
    )


def transform_spm_features(
    X,
    selected_patterns,
    window_size=5,
):
    print("\nDiscretizing validation data...")

    X_discretized = discretize_dataset(X)

    print("Building pattern matrix using training vocabulary...")

    X_pattern = build_pattern_matrix(
        X_discretized,
        selected_patterns,
        window_size=window_size,
    )

    print(f"Validation pattern matrix: {X_pattern.shape}")

    return X_pattern


def main():
    parser = argparse.ArgumentParser(
        description="Train SPM + SVM temporal classification model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the processed .npz dataset.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # ---------------------------------------------------------
    # SPM parameters
    # ---------------------------------------------------------

    min_support = 0.10
    window_size = 5
    top_k = 10000

    # ---------------------------------------------------------
    # SVM parameters
    # ---------------------------------------------------------

    svm_C = 10.0
    svm_gamma = "scale"
    svm_kernel = "rbf"

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    X_train, y_train, X_val, y_val = load_dataset(
        data_path
    )

    # ---------------------------------------------------------
    # Build SPM features
    # ---------------------------------------------------------

    (
        X_train_selected,
        selected_patterns,
        frequent_patterns,
    ) = fit_spm_features(
        X_train,
        y_train,
        min_support=min_support,
        window_size=window_size,
        top_k=top_k,
    )

    # ---------------------------------------------------------
    # Transform validation data
    # ---------------------------------------------------------

    X_val_selected = transform_spm_features(
        X_val,
        selected_patterns,
        window_size=window_size,
    )

    print("\n===== FINAL FEATURE SHAPES =====")
    print(f"X_train_selected: {X_train_selected.shape}")
    print(f"X_val_selected:   {X_val_selected.shape}")

    # ---------------------------------------------------------
    # Train SVM
    # ---------------------------------------------------------

    print("\n===== TRAINING SVM =====")
    print(f"Kernel: {svm_kernel}")
    print(f"C: {svm_C}")
    print(f"Gamma: {svm_gamma}")

    svm_pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel=svm_kernel,
                    C=svm_C,
                    gamma=svm_gamma,
                ),
            ),
        ]
    )

    svm_pipeline.fit(
        X_train_selected,
        y_train,
    )

    # ---------------------------------------------------------
    # Evaluate on training and validation sets
    # ---------------------------------------------------------

    train_accuracy = svm_pipeline.score(
        X_train_selected,
        y_train,
    )

    val_accuracy = svm_pipeline.score(
        X_val_selected,
        y_val,
    )

    print(
        f"\nTraining accuracy:   {train_accuracy:.4f}"
    )

    print(
        f"Validation accuracy: {val_accuracy:.4f}"
    )

    # ---------------------------------------------------------
    # Save model and SPM information
    # ---------------------------------------------------------

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    model_path = (
        output_dir / "spm_svm_model.joblib"
    )

    model_data = {
        "model": svm_pipeline,
        "patterns": selected_patterns,
        "frequent_patterns": frequent_patterns,
        "min_support": min_support,
        "window_size": window_size,
        "top_k": top_k,
    }

    joblib.dump(
        model_data,
        model_path,
    )

    print(
        f"\nModel saved to: {model_path}"
    )


if __name__ == "__main__":
    main()