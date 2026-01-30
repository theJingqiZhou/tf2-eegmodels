import os
from typing import Callable, Literal

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras import activations, backend, constraints, layers


def _clip_log(x: tf.Tensor) -> tf.Tensor:
    return backend.log(backend.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(
    dropout_rate: float = 0.5,
    model_name: str = "shallowconvnet",
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
        40,
        (1, 13),
        input_shape=(channels, samples, 1),
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
    )(x)
    x = layers.Conv2D(
        40,
        (channels, 1),
        use_bias=False,
        kernel_constraint=constraints.max_norm(2.0, axis=(0, 1, 2)),
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = layers.Lambda(backend.square)(x)
    x = layers.AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(x)
    x = layers.Lambda(_clip_log)(x)
    x = layers.Dropout(dropout_rate)(x)

    if include_top:
        x = layers.Flatten()(x)

        output = layers.Dense(
            classes,
            activation=activations.get(classifier_activation),
            kernel_constraint=constraints.max_norm(0.5),
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPool2D(name="max_pool")(x)

        output = x

    return keras.Model(inputs=input, outputs=output, name=model_name)


if __name__ == "__main__":
    net = ShallowConvNet()
    net.summary()
