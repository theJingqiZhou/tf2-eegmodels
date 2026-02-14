"""BrainNet model backbone definition in Keras (Keras Applications Style).

This module implements the architecture described in the BrainNet paper
with a keras.applications-like API.

Reference:
    Fallahi et al., "BrainNet: Improving Brainwave-based Biometric Recognition with Siamese Networks"
    2023 IEEE International Conference on Pervasive Computing and Communications (PerCom)
    https://doi.org/10.1109/PERCOM56429.2023.10099367
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Literal, override

import keras
import tensorflow as tf
from keras import backend, initializers, layers


def _paper_initializer() -> initializers.Initializer:
    """Return the weight initializer used in the original BrainNet paper (LeCun Normal)."""
    return initializers.LecunNormal()


class BrainNetBlock(layers.Layer):
    """A standard BrainNet Convolutional block (Conv2D -> AvgPool -> Dropout).

    The architecture consists of a 2D convolution (processing time within channels),
    followed by Average Pooling and Dropout.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: tuple[int, int] = (1, 15),
        pool_size: tuple[int, int] = (1, 2),
        dropout_rate: float = 0.3,
        activation: str = "selu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        self.conv: layers.Conv2D | None = None
        self.pool: layers.AveragePooling2D | None = None
        self.dropout: layers.Dropout | None = None

    @override
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="valid",
            activation=self.activation_name,
            kernel_initializer=_paper_initializer(),
            name="conv",
        )
        self.pool = layers.AveragePooling2D(pool_size=self.pool_size, name="avg_pool")
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")
        super().build(input_shape)

    @override
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.conv is None:
            raise RuntimeError("Layer not built.")

        x = self.conv(inputs)
        x = self.pool(x)
        return self.dropout(x, training=training)

    @override
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation_name,
            }
        )
        return config


def BrainNet(
    # Domain-specific args
    initial_downsample: bool = False,
    dropout_rate: float = 0.3,
    # Standard keras.applications args
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg", "max"] | None = None,
    classes: int = 40,
    classifier_activation: str | Callable | None = "softmax",
    name: str = "brainnet",
) -> keras.Model:
    """Instantiates the BrainNet architecture.

    The model expects a 2D EEG input (Channels, Samples), internally reshapes it
    to (Channels, Samples, 1), and extracts features using a series of 2D Convolutions
    (acting on time axis per channel).

    Args:
        initial_downsample: Whether to apply an initial AveragePooling2D((1, 2))
            layer. The original paper uses this for 1024Hz datasets (D1/D2) but
            not for 512Hz datasets (D3).
        dropout_rate: Dropout rate for internal blocks (default 0.3).
        include_top: Whether to include the fully-connected classification layer
            at the top of the network. If False, the model outputs the 32-dim
            embedding vector (or pooled features).
        weights: Path to a weights file to load.
        input_tensor: Optional Keras tensor to use as input.
        input_shape: Optional shape tuple, e.g., (Channels, Time).
            If None, defaults to (30, 1025) based on Dataset 1 in paper.
            Do not include the channel dimension '1' at the end.
        pooling: Optional pooling mode for feature extraction when `include_top` is `False`.
            - `None`: means that the output of the model will be the 2D tensor output
              of the last dense embedding layer (before classification).
            - `avg`: means that global average pooling will be applied to the
              flattened features (not applicable here as structure flattens early).
              Kept for API consistency but ignored in this specific architecture logic.
            - `max`: Same as above.
        classes: Number of classes to classify images into, only specified if
            `include_top` is True.
        classifier_activation: Activation function to use on the "top" layer.
            Ignored unless `include_top=True`.
        name: The name of the model.

    Returns:
        A `keras.Model` instance.
    """
    # 1. Input handling
    if input_tensor is None:
        if input_shape is None:
            # Default to D1 params: 30 channels, 1025 samples
            input_shape = (30, 1025)
        eeg_input = keras.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            eeg_input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            eeg_input = input_tensor

    # Explicit Reshape: (Channels, Samples) -> (Channels, Samples, 1)
    # This aligns with the "intuition" requirement while satisfying Conv2D needs.
    x = layers.Reshape(target_shape=input_shape + (1,), name="expand_dim")(eeg_input)

    # Optional Initial Downsampling (for high sampling rate datasets)
    if initial_downsample:
        x = layers.AveragePooling2D(pool_size=(1, 2), name="initial_avg_pool")(x)

    # 2. Body Architecture (Feature Extractor)
    # Block 1: 128 filters
    x = BrainNetBlock(
        filters=128, kernel_size=(1, 15), dropout_rate=dropout_rate, name="block_1"
    )(x)

    # Block 2: 32 filters
    x = BrainNetBlock(
        filters=32, kernel_size=(1, 15), dropout_rate=dropout_rate, name="block_2"
    )(x)

    # Block 3: 16 filters
    x = BrainNetBlock(
        filters=16, kernel_size=(1, 15), dropout_rate=dropout_rate, name="block_3"
    )(x)

    # Block 4: 8 filters
    x = BrainNetBlock(
        filters=8, kernel_size=(1, 15), dropout_rate=dropout_rate, name="block_4"
    )(x)

    # Block 5: 4 filters
    x = BrainNetBlock(
        filters=4, kernel_size=(1, 15), dropout_rate=dropout_rate, name="block_5"
    )(x)

    # Flatten
    x = layers.Flatten(name="flatten")(x)

    # Embedding Layer (The "Brain Template" - 32 units)
    # Note: Original code uses 'lecun_normal' and no activation for the embedding.
    x = layers.Dense(
        32,
        activation=None,
        kernel_initializer=_paper_initializer(),
        name="embedding_dense",
    )(x)

    # 3. Head / Top
    if include_top:
        # Original paper uses Triplet Loss here.
        # Since strategy is removed, we add a standard classification head.
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=_paper_initializer(),
            name="predictions",
        )(x)
    else:
        # Handling pooling argument for consistency, although BrainNet
        # structure naturally flattens before the embedding.
        # If the user asks for pooling on the 32-dim vector, it's trivial
        # (it's already a vector), but we implement standard logic just in case.
        if pooling == "avg":
            # x is (Batch, 32), GlobalAveragePooling1D expects (Batch, Steps, Features)
            # Effectively identity here if we consider it a feature vector,
            # but usually applied to Conv feature maps.
            # Given BrainNet topology, 'x' here is already the global descriptor.
            pass
        elif pooling == "max":
            pass

    # 4. Model Construction
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = eeg_input

    model = keras.Model(inputs, x, name=name)

    # 5. Load Weights
    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == "__main__":
    # Test instantiation
    model = BrainNet(
        input_shape=(30, 1025),  # Note: No trailing 1
        classes=40,
        initial_downsample=True,  # Simulating D1/D2 config
        include_top=True,
    )
    model.summary()
