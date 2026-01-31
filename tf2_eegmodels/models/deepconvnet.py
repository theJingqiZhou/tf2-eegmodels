import pathlib
from typing import Callable, Literal

import keras
import tensorflow as tf
from keras import activations, backend, constraints, layers


def DeepConvNet(
    dropout_rate: float = 0.5,
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg"] | Literal["max"] | None = None,
    classes: int = 4,
    classifier_activation: str | Callable = "softmax",
    name: str = "deepconvnet",
) -> keras.Model:
    if input_shape is None:
        input_shape = (64, 256)
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
        25,
        (1, 5),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="block1_conv1",
    )(x)
    x = layers.Conv2D(
        25,
        (channels, 1),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="block1_conv2",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name="block1_bn")(x)
    x = layers.ELU(name="block1_elu")(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name="block1_pool")(x)
    x = layers.Dropout(dropout_rate, name="block1_dropout")(x)

    x = layers.Conv2D(
        50,
        (1, 5),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="block2_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name="block2_bn")(x)
    x = layers.ELU(name="block2_elu")(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name="block2_pool")(x)
    x = layers.Dropout(dropout_rate, name="block2_dropout")(x)

    x = layers.Conv2D(
        100,
        (1, 5),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="block3_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name="block3_bn")(x)
    x = layers.ELU(name="block3_elu")(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name="block3_pool")(x)
    x = layers.Dropout(dropout_rate, name="block3_dropout")(x)

    x = layers.Conv2D(
        200,
        (1, 5),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="block4_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name="block4_bn")(x)
    x = layers.ELU(name="block4_elu")(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name="block4_pool")(x)
    x = layers.Dropout(dropout_rate, name="block4_dropout")(x)

    if include_top:
        x = layers.Flatten(name="flatten")(x)

        x = layers.Dense(
            classes,
            activation=activations.get(classifier_activation),
            kernel_constraint=constraints.max_norm(0.5),
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
    net = DeepConvNet()
    net.summary()
