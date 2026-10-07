import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )

        self.feed_forward = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dropout(dropout),
                Dense(d_model),
            ]
        )

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        # Self-attention
        attention_output = self.attention(
            inputs,
            inputs,
            training=training,
        )

        attention_output = self.dropout1(
            attention_output,
            training=training,
        )

        # First residual connection + normalization
        x = self.norm1(
            inputs + attention_output
        )

        # Feed-forward network
        ff_output = self.feed_forward(
            x,
            training=training,
        )

        ff_output = self.dropout2(
            ff_output,
            training=training,
        )

        # Second residual connection + normalization
        return self.norm2(
            x + ff_output
        )


def build_transformer_model(
    time_steps=40,
    num_features=144,
    num_classes=25,
):
    inputs = Input(
        shape=(time_steps, num_features)
    )

    # Project the 144 input features into the Transformer
    # representation dimension.
    x = Dense(64)(inputs)

    # Learnable positional embeddings.
    positions = tf.range(
        start=0,
        limit=time_steps,
        delta=1,
    )

    positional_embedding = tf.keras.layers.Embedding(
        input_dim=time_steps,
        output_dim=64,
    )(positions)

    x = x + positional_embedding

    # One lightweight Transformer Encoder block.
    x = TransformerEncoder(
        d_model=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.2,
    )(x)

    # Convert the sequence into one feature vector.
    x = GlobalAveragePooling1D()(x)

    # Classification head.
    x = Dense(
        32,
        activation="relu",
    )(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
    )

    return model