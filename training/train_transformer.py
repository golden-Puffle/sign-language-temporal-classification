import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.transformer_model import build_transformer_model


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


def main():
    parser = argparse.ArgumentParser(
        description="Train the Transformer Encoder temporal classification model."
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
    # Load dataset
    # ---------------------------------------------------------

    X_train, y_train, X_val, y_val = load_dataset(
        data_path
    )

    # ---------------------------------------------------------
    # Build Transformer model
    # ---------------------------------------------------------

    model = build_transformer_model(
        time_steps=40,
        num_features=144,
        num_classes=25,
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n===== TRANSFORMER MODEL =====")
    model.summary()

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    checkpoint_path = (
        output_dir / "best_transformer_model.keras"
    )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\n===== TRAINING TRANSFORMER =====")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------------
    # Training complete
    # ---------------------------------------------------------

    print("\n===== TRAINING COMPLETE =====")

    print(
        f"Best model saved to: {checkpoint_path}"
    )

    # ---------------------------------------------------------
    # Report best validation accuracy
    # ---------------------------------------------------------

    best_val_accuracy = max(
    history.history["val_accuracy"]
    )

    best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    )
    + 1
    )

    best_train_accuracy = history.history["accuracy"][
    best_epoch - 1
    ]

    print(
    f"Training accuracy at best epoch: "
    f"{best_train_accuracy:.4f}"
    )

    print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
    )

    print(
    f"Best validation epoch: "
    f"{best_epoch}"
    )


if __name__ == "__main__":
    main()