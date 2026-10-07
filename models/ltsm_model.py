import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.regularizers import l2


def build_lstm_model(
    time_steps=40,
    num_features=144,
    num_classes=25
):
    model = Sequential([
        Input(shape=(time_steps, num_features)),

        LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4),
        ),
        Dropout(0.3),

        LSTM(
            64,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4),
        ),
        Dropout(0.3),

        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(1e-4),
        ),
        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        ),
    ])

    return model


if __name__ == "__main__":
    model = build_lstm_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()