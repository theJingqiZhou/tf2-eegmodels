"""TaskOriNet model definition in Keras (Keras Applications Style).

This module implements the architecture described in the EEG-AD preprint
(`arXiv:2207.01391v1`) with a keras.applications-like API.

The model is a two-branch ResNet-based architecture designed for EEG anomaly detection.
It uses 1x7 convolutions to capture temporal features and a secondary branch to
capture cross-channel correlations.

Reference:
    Zheng et al., "Task-oriented Self-supervised Learning for Anomaly Detection in Electroencephalography"
    https://arxiv.org/abs/2207.01391

    Original PyTorch Implementation:
    https://github.com/ironing/Task-oriented-SSL-EEG-AD/blob/main/model.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal

import keras
import tensorflow as tf
from keras import activations, backend, initializers, layers, ops


def _paper_initializer() -> initializers.HeNormal:
    """Return the weight initializer used in the original paper (Kaiming Normal)."""
    return initializers.HeNormal()


def _basic_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name: str | None = None,
) -> tf.Tensor:
    """A ResNet BasicBlock adapted for EEG (1D temporal convolution).

    Corresponds to `BasicBlock2` in the original PyTorch code.
    Uses (1, 7) kernels instead of (3, 3).
    """
    if name is None:
        name = f"basic_block_{keras.backend.get_uid('basic_block')}"

    shortcut = x
    in_filters = x.shape[-1]

    # Main path
    # Conv1: 1x7, stride=(1, s), padding=same
    x_out = layers.Conv2D(
        filters,
        kernel_size=(1, 7),
        strides=(1, stride),
        padding="same",
        use_bias=False,
        kernel_initializer=_paper_initializer(),
        name=f"{name}_conv1",
    )(x)
    x_out = layers.BatchNormalization(name=f"{name}_bn1")(x_out)
    x_out = layers.Activation("relu", name=f"{name}_relu1")(x_out)

    # Conv2: 1x7, stride=(1, 1), padding=same
    x_out = layers.Conv2D(
        filters,
        kernel_size=(1, 7),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer=_paper_initializer(),
        name=f"{name}_conv2",
    )(x_out)
    x_out = layers.BatchNormalization(name=f"{name}_bn2")(x_out)

    # Shortcut path (Projection)
    # Applied if stride > 1 or channel count changes
    if stride != 1 or in_filters != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=(1, 7),
            strides=(1, stride),
            padding="same",
            use_bias=False,
            kernel_initializer=_paper_initializer(),
            name=f"{name}_downsample_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_downsample_bn")(shortcut)

    x_out = layers.Add(name=f"{name}_add")([x_out, shortcut])
    x_out = layers.Activation("relu", name=f"{name}_out_relu")(x_out)
    return x_out


def _make_layer(
    x: tf.Tensor,
    filters: int,
    blocks: int,
    stride: int = 1,
    name: str = "layer",
) -> tf.Tensor:
    """Creates a ResNet layer (stage) consisting of multiple BasicBlocks."""
    # The first block handles the stride and filter expansion
    x = _basic_block(x, filters, stride=stride, name=f"{name}_block0")

    # Subsequent blocks have stride 1
    for i in range(1, blocks):
        x = _basic_block(x, filters, stride=1, name=f"{name}_block{i}")
    return x


def TaskOriNet(
    # Domain-specific args
    base_filters: int = 16,
    # Standard keras.applications args
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal[None] = None,
    classes: int = 3,
    classifier_activation: str | Callable = "softmax",
    name: str = "taskorinet",
) -> keras.Model:
    """Instantiates the TaskOriNet architecture.

    Args:
        base_filters: The initial number of filters (planes), corresponds to `inplane` in PyTorch code.
        include_top: Whether to include the fully-connected layer at the top of the network.
        weights: Path to a weights file to load.
        input_tensor: Optional Keras tensor to use as input.
        input_shape: Optional shape tuple (Channels, Samples). Defaults to (18, 769).
        pooling: Unused parameter, kept for API consistency.
        classes: Number of classes for classification.
                 (3 for the SSL task: Normal, Amp-Abnormal, Freq-Abnormal).
        classifier_activation: Activation function for the classification head.
        name: The name of the model.

    Returns:
        A `keras.Model` instance.
    """
    _ = pooling  # Pooling is hardcoded in the architecture logic

    # 1. Input handling
    if input_shape is None:
        # Default matching the paper/repo default: 18 channels, 3 seconds @ ~256Hz (769 samples)
        input_shape = (18, 769)

    if input_tensor is None:
        eeg_input = keras.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            eeg_input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            eeg_input = input_tensor

    # Handle shape: (Batch, Channels, Samples) -> (Batch, Channels, Samples, 1)
    # treating EEG as a "grayscale image" where Height=Channels, Width=Time
    channels_dim = input_shape[0]
    x = layers.Reshape((input_shape[0], input_shape[1], 1), name="input_reshape")(
        eeg_input
    )

    # 2. Initial Convolution
    # PyTorch: Conv2d(1, base_filters, kernel=(1, 7), stride=(1, 2), padding=(0, 3))
    x = layers.Conv2D(
        base_filters,
        kernel_size=(1, 7),
        strides=(1, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=_paper_initializer(),
        name="conv1",
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x1 = layers.Activation("relu", name="relu1")(x)  # Saved for Branch 2

    # 3. Backbone (ResNet Stages)
    # Layer 1: 3 blocks, stride 1, filters = base
    x2 = _make_layer(x1, base_filters, blocks=3, stride=1, name="layer1")

    # Layer 2: 4 blocks, stride 2, filters = base * 2
    x2 = _make_layer(x2, base_filters * 2, blocks=4, stride=2, name="layer2")

    # Layer 3: 6 blocks, stride 2, filters = base * 4
    x2 = _make_layer(x2, base_filters * 4, blocks=6, stride=2, name="layer3")

    # Layer 4: 3 blocks, stride 2, filters = base * 8
    x2 = _make_layer(x2, base_filters * 8, blocks=3, stride=2, name="layer4")

    # 4. Branch 1 Output Processing
    # PyTorch: conv2 (1x7, no stride, out=16)
    x2 = layers.Conv2D(
        16,
        kernel_size=(1, 7),
        padding="same",
        use_bias=False,
        kernel_initializer=_paper_initializer(),
        name="branch1_conv",
    )(x2)

    # PyTorch: AdaptiveAvgPool2d((18, 1))
    # This pools the Time dimension (W) completely, preserving Channel dimension (H=18)
    # Input x2 shape: (Batch, Channels, Time_reduced, 16)
    # Output required: (Batch, Channels, 1, 16) -> Flattened
    x2 = ops.mean(x2, axis=2, keepdims=False)  # Pool time dimension
    branch1_flat = layers.Flatten(name="branch1_flatten")(x2)

    # 5. Branch 2 (Shortcut) Output Processing
    # This branch takes the early feature map x1 and captures cross-channel info
    # PyTorch: conv3 input=x1, kernel=(18, 7), stride=(1, 3), out=2
    # Note: Kernel height must equal number of channels (18 in paper) to mix all channels

    if channels_dim is None:
        raise ValueError(
            "Input shape must have a defined channel dimension for TaskOriNet."
        )

    branch2 = layers.Conv2D(
        2,
        kernel_size=(channels_dim, 7),
        strides=(1, 3),
        padding="valid",  # PyTorch padding=(0,0) with kernel height=channels implies valid in H
        use_bias=False,
        kernel_initializer=_paper_initializer(),
        name="branch2_conv",
    )(x1)
    branch2_flat = layers.Flatten(name="branch2_flatten")(branch2)

    # 6. Feature Fusion
    # PyTorch: cat((out1, out2), dim=-1)
    embeds = layers.Concatenate(name="embedding_concat")([branch1_flat, branch2_flat])

    # 7. Classification Head
    if include_top:
        # PyTorch: Linear(542, num_classes)
        # Note: 542 is derived from specific input size (18, 769).
        # Keras calculates dense input size automatically.
        x_out = layers.Dense(
            classes, kernel_initializer=_paper_initializer(), name="logits_dense"
        )(embeds)

        if classifier_activation is not None:
            x_out = layers.Activation(
                activations.get(classifier_activation), name="predictions"
            )(x_out)
    else:
        x_out = embeds

    # 8. Model Construction
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = eeg_input

    model = keras.Model(inputs, x_out, name=name)

    # 9. Load Weights
    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == "__main__":
    # Test instantiation based on paper defaults
    model = TaskOriNet(input_shape=(18, 769), classes=3, include_top=True)
    model.summary()
