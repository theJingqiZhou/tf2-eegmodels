import os
from typing import Callable, Literal

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras import activations, backend, constraints, layers


def EEGNetV4(
    srate: int = 128,
    temporal_filters: int = 8,
    spatial_filters: int = 2,
    pointwise_filters: int = 16,
    srate_reduce_steps: tuple[int, int] = (4, 8),
    classifier_norm_value: float = 0.25,
    dropout_type: Literal["SpatialDropout2D"] | Literal["Dropout"] = "Dropout",
    dropout_rate: float = 0.5,
    model_name: str = "eegnetv4",
    include_top: bool = True,
    input_tensor: tf.Tensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg"] | Literal["max"] | None = None,
    classes: int = 4,
    classifier_activation: str | Callable = "softmax",
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
        input = keras.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    channels, samples = input.shape[-2:]

    x = layers.Reshape((channels, samples, 1))(input)
    x = layers.Conv2D(
        temporal_filters, (1, srate // 2), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        depth_multiplier=spatial_filters,
        depthwise_constraint=constraints.max_norm(1.0),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.AveragePooling2D((1, srate_reduce_steps[0]))(x)
    x = dropout(dropout_rate)(x)

    half_second_samples = int(0.5 * (srate // srate_reduce_steps[0]))
    x = layers.SeparableConv2D(
        pointwise_filters, (1, half_second_samples), use_bias=False, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.AveragePooling2D((1, srate_reduce_steps[1]))(x)
    x = dropout(dropout_rate)(x)

    if include_top:
        x = layers.Flatten()(x)

        output = layers.Dense(
            classes,
            activation=activations.get(classifier_activation),
            kernel_constraint=constraints.max_norm(classifier_norm_value),
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPool2D(name="max_pool")(x)

        output = x

    return keras.Model(inputs=input, outputs=output, name=model_name)


if __name__ == "__main__":
    net = EEGNetV4()
    net.summary()
