import pathlib
from typing import Callable, Literal, Sequence

import keras
import tensorflow as tf
from keras import activations, backend, layers, regularizers


def EEGNet(
    kernels: Sequence[tuple[int, int]] = [(2, 32), (8, 4)],
    strides: tuple[int, int] = (2, 4),
    l1: float = 1e-4,
    l2: float = 1e-4,
    dropout_rate: float = 0.25,
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg"] | Literal["max"] | None = None,
    classes: int = 4,
    classifier_activation: str | Callable = "softmax",
    name: str = "eegnet",
) -> keras.Model:
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
        16,
        (channels, 1),
        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
        name="block1_conv",
    )(x)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.ELU(name="block1_elu")(x)
    x = layers.Dropout(dropout_rate, name="block1_dropout")(x)

    permute_dims = 2, 1, 3
    x = layers.Permute(permute_dims)(x)

    x = layers.Conv2D(
        4,
        kernels[0],
        padding="same",
        kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=l2),
        strides=strides,
        name="block2_conv",
    )(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.ELU(name="block2_elu")(x)
    x = layers.Dropout(dropout_rate, name="block2_dropout")(x)

    x = layers.Conv2D(
        4,
        kernels[1],
        padding="same",
        kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=l2),
        strides=strides,
        name="block3_conv",
    )(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.ELU(name="block3_elu")(x)
    x = layers.Dropout(dropout_rate, name="block3_dropout")(x)

    if include_top:
        x = layers.Flatten(name="flatten")(x)

        x = layers.Dense(
            classes,
            activation=activations.get(classifier_activation),
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
    net = EEGNet()
    net.summary()
