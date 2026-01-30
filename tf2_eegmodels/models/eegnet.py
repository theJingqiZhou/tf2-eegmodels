import os
from typing import Callable, Literal, Sequence

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras import activations, backend, layers, regularizers


def EEGNet(
    kernels: Sequence[tuple[int, int]] = [(2, 32), (8, 4)],
    strides: tuple[int, int] = (2, 4),
    l1: float = 1e-4,
    l2: float = 1e-4,
    dropout_rate: float = 0.25,
    model_name: str = "eegnet",
    include_top: bool = True,
    input_tensor: tf.Tensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg"] | Literal["max"] | None = None,
    classes: int = 4,
    classifier_activation: str | Callable = "softmax",
) -> keras.Model:
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
        16,
        (channels, 1),
        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Dropout(dropout_rate)(x)

    permute_dims = 2, 1, 3
    x = layers.Permute(permute_dims)(x)

    x = layers.Conv2D(
        4,
        kernels[0],
        padding="same",
        kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=l2),
        strides=strides,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(
        4,
        kernels[1],
        padding="same",
        kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=l2),
        strides=strides,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Dropout(dropout_rate)(x)

    if include_top:
        x = layers.Flatten()(x)

        output = layers.Dense(
            classes, activation=activations.get(classifier_activation)
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPool2D(name="max_pool")(x)

        output = x

    return keras.Model(inputs=input, outputs=output, name=model_name)


if __name__ == "__main__":
    net = EEGNet()
    net.summary()
