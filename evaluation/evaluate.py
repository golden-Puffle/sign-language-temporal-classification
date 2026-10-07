import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.transformer_model import TransformerEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def load_test_data(data_path):
    data = np.load(data_path)

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Test dataset loaded.")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    return X_test, y_test


def evaluate_model(model_path, data_path, output_name):
    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    X_test, y_test = load_test_data(data_path)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print(f"\nLoading model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "TransformerEncoder": TransformerEncoder
        },
    )

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    print("\n===== TEST SET EVALUATION =====")

    probabilities = model.predict(
        X_test,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print("\n===== TEST RESULTS =====")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(25)),
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(25)),
    )

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    confusion_path = (
        figures_dir
        / f"confusion_matrix_{output_name}.png"
    )

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title(
        f"Confusion Matrix - {output_name}"
    )

    plt.colorbar()

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(25),
        range(25),
        rotation=90,
    )

    plt.yticks(
        range(25),
        range(25),
    )

    plt.tight_layout()

    plt.savefig(
        confusion_path,
        dpi=300,
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to: "
        f"{confusion_path}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a temporal classification model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the processed .npz dataset.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained Keras model.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Name used for the confusion matrix file.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    evaluate_model(
        model_path=model_path,
        data_path=data_path,
        output_name=args.name,
    )


if __name__ == "__main__":
    main()