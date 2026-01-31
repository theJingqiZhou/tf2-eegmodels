import pathlib
from typing import Callable, Literal

import keras
import tensorflow as tf
from keras import activations, backend, constraints, layers, ops


def ShallowConvNet(
    dropout_rate: float = 0.5,
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg"] | Literal["max"] | None = None,
    classes: int = 4,
    classifier_activation: str | Callable = "softmax",
    name: str = "shallowconvnet",
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
        40,
        (1, 13),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="conv1",
    )(x)
    x = layers.Conv2D(
        40,
        (channels, 1),
        use_bias=False,
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
        name="conv2",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name="bn")(x)
    x = layers.Lambda(ops.square, name="square")(x)
    x = layers.AveragePooling2D(pool_size=(1, 35), strides=(1, 7), name="pool")(x)
    x = layers.Lambda(
        lambda x_in: ops.log(ops.clip(x_in, x_min=1e-7, x_max=10000)), name="clip_log"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

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
    net = ShallowConvNet()
    net.summary()
