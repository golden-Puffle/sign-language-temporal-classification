import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.ltsm_model import build_lstm_model
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def load_dataset(data_path):
    """Load and verify the processed landmark dataset."""

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


def main():
    parser = argparse.ArgumentParser(
        description="Train the LSTM temporal classification model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the processed .npz dataset."
    )

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # Load dataset
    X_train, y_train, X_val, y_val = load_dataset(data_path)



    classes = np.unique(y_train)

    class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
    )

    class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, class_weights_array)
    }

    print("\nClass weights:")
    for class_id, weight in class_weights.items():
        print(f"Class {class_id}: {weight:.4f}")

    # Build model
    model = build_lstm_model(
        time_steps=40,
        num_features=144,
        num_classes=25
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # Save the best model based on validation accuracy
    checkpoint_path = output_dir / "best_lstm_model_class_weighted.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )
    ]

    # Train
    history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

    print("\nTraining complete.")
    print(f"Best model saved to: {checkpoint_path}")
    
    # Create figures directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    accuracy_path = figures_dir / "training_accuracy_class_weighted.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = figures_dir / "training_loss_class_weighted.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print(f"Training accuracy plot saved to: {accuracy_path}")
    print(f"Training loss plot saved to: {loss_path}")

    return history


if __name__ == "__main__":
    main()
