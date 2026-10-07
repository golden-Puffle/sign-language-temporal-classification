import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.transformer_model import TransformerEncoder


DEFAULT_MODEL = Path("models/best_transformer_model.keras")
DEFAULT_LABELS = Path("data/processed/label_map.json")


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as file:
        label_map = json.load(file)

    # label_map is {label_name: class_id}
    id_to_label = {
        int(class_id): label
        for label, class_id in label_map.items()
    }

    return id_to_label


def load_test_data(data_path):
    data = np.load(data_path)

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Test dataset loaded.")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    return X_test, y_test


def load_model(model_path):
    print(f"\nLoading model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "TransformerEncoder": TransformerEncoder
        },
    )

    print("Model loaded successfully.")

    return model


def predict_sample(
    model,
    X_test,
    y_test,
    id_to_label,
    index,
    top_k=3,
):
    sequence = X_test[index:index + 1]

    probabilities = model.predict(
        sequence,
        verbose=0,
    )[0]

    ranked_indices = np.argsort(
        probabilities
    )[::-1]

    actual_id = int(y_test[index])

    print("\n===== SINGLE SAMPLE PREDICTION =====")

    print(f"Sample index: {index}")

    print(
        f"Actual class: {actual_id} "
        f"({id_to_label.get(actual_id, 'unknown')})"
    )

    predicted_id = int(ranked_indices[0])

    print(
        f"Predicted class: {predicted_id} "
        f"({id_to_label.get(predicted_id, 'unknown')})"
    )

    print(
        f"Confidence: "
        f"{probabilities[predicted_id]:.4f}"
    )

    print(f"\nTop {top_k} predictions:")

    for rank, class_id in enumerate(
        ranked_indices[:top_k],
        start=1,
    ):
        class_id = int(class_id)

        print(
            f"{rank}. "
            f"{id_to_label.get(class_id, 'unknown')} "
            f"(class {class_id}) - "
            f"{probabilities[class_id]:.4f}"
        )


def predict_all(
    model,
    X_test,
    y_test,
    id_to_label,
):
    probabilities = model.predict(
        X_test,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    correct = np.sum(
        y_pred == y_test
    )

    accuracy = correct / len(y_test)

    print("\n===== TEST SET PREDICTION =====")

    print(
        f"Correct predictions: "
        f"{correct}/{len(y_test)}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\n===== PER-SAMPLE PREDICTIONS =====")

    for index in range(len(y_test)):
        actual_id = int(y_test[index])
        predicted_id = int(y_pred[index])
        confidence = float(
            probabilities[index][predicted_id]
        )

        correct_marker = (
            "✓"
            if actual_id == predicted_id
            else "✗"
        )

        actual_label = id_to_label.get(
            actual_id,
            "unknown",
        )

        predicted_label = id_to_label.get(
            predicted_id,
            "unknown",
        )

        print(
            f"{index:02d} {correct_marker} "
            f"Actual: {actual_label:<15} "
            f"Predicted: {predicted_label:<15} "
            f"Confidence: {confidence:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict sign language classes using the Transformer model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the processed .npz dataset.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained Transformer model.",
    )

    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        help="Path to label_map.json.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Test sample index for single-sample prediction.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label map not found: {labels_path}"
        )

    X_test, y_test = load_test_data(
        data_path
    )

    id_to_label = load_labels(
        labels_path
    )

    model = load_model(
        model_path
    )

    if args.index is not None:
        if args.index < 0 or args.index >= len(X_test):
            raise IndexError(
                f"Index must be between "
                f"0 and {len(X_test) - 1}."
            )

        predict_sample(
            model,
            X_test,
            y_test,
            id_to_label,
            args.index,
            top_k=args.top_k,
        )

    else:
        predict_all(
            model,
            X_test,
            y_test,
            id_to_label,
        )


if __name__ == "__main__":
    main()