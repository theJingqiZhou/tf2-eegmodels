import pathlib
from typing import Callable, Literal

import keras
import tensorflow as tf
from keras import activations, backend, constraints, layers


def EEGNetV4(
    srate: int = 128,
    temporal_filters: int = 8,
    spatial_filters: int = 2,
    pointwise_filters: int = 16,
    srate_reduce_steps: tuple[int, int] = (4, 8),
    classifier_norm_value: float = 0.25,
    dropout_type: Literal["SpatialDropout2D"] | Literal["Dropout"] = "Dropout",
    dropout_rate: float = 0.5,
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg"] | Literal["max"] | None = None,
    classes: int = 4,
    classifier_activation: str | Callable = "softmax",
    name: str = "eegnetv4",
) -> keras.Model:
    valid_dropouts = {
        "SpatialDropout2D": layers.SpatialDropout2D,
        "Dropout": layers.Dropout,
    }

    dropout = valid_dropouts.get(dropout_type)
    if dropout is None:
        raise ValueError(
            "Expected `dropout_type` to be either 'SpatialDropout2D' or 'Dropout',"
            f" but got {dropout_type}!"
        )

    if input_shape is None:
        input_shape = (64, 128)
    if input_tensor is None:
        eeg_input = keras.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            eeg_input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            eeg_input = input_tensor

    channels, samples = eeg_input.shape[-2:]

    x = layers.Reshape((channels, samples, 1))(eeg_input)
    x = layers.Conv2D(
        temporal_filters,
        (1, srate // 2),
        padding="same",
        use_bias=False,
        name="block1_conv1",
    )(x)
    x = layers.BatchNormalization(name="block1_bn1")(x)
    x = layers.DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        depth_multiplier=spatial_filters,
        depthwise_constraint=constraints.max_norm(1.0),
        name="block1_conv2",
    )(x)
    x = layers.BatchNormalization(name="block1_bn2")(x)
    x = layers.ELU(name="block1_elu")(x)
    x = layers.AveragePooling2D((1, srate_reduce_steps[0]), name="block1_pool")(x)
    x = dropout(dropout_rate, name="block1_dropout")(x)

    half_second_samples = int(0.5 * (srate // srate_reduce_steps[0]))
    x = layers.SeparableConv2D(
        pointwise_filters,
        (1, half_second_samples),
        use_bias=False,
        padding="same",
        name="block2_conv",
    )(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.ELU(name="block2_elu")(x)
    x = layers.AveragePooling2D((1, srate_reduce_steps[1]), name="block2_pool")(x)
    x = dropout(dropout_rate, name="block2_dropout")(x)

    if include_top:
        x = layers.Flatten(name="flatten")(x)

        x = layers.Dense(
            classes,
            activation=activations.get(classifier_activation),
            kernel_constraint=constraints.max_norm(classifier_norm_value),
            name="predictions",
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPool2D()(x)

    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = eeg_input

    model = keras.Model(inputs, x, name=name)

    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == "__main__":
    net = EEGNetV4()
    net.summary()
