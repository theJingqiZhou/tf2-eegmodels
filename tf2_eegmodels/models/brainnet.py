import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras import backend, layers

def BrainNet(
    model_name: str = "brainnet",
    input_tensor: tf.Tensor | None = None,
    input_shape: tuple[int, int] | None = None,
) -> keras.Model:
    if input_shape is None:
        input_shape = (30, 1025)
    if input_tensor is None:
        input = keras.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    channels, samples = input.shape[-2:]

    x = layers.Reshape((channels, samples, 1))(input)
    x = layers.AveragePooling2D(pool_size=(1, 2))(x)

    x = layers.Conv2D(
        128, (1, 15), activation="selu", kernel_initializer="lecun_normal"
    )(x)
    x = layers.AveragePooling2D(pool_size=(1, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        32, (1, 15), activation="selu", kernel_initializer="lecun_normal"
    )
    x = layers.AveragePooling2D(pool_size=(1, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        16, (1, 15), activation="selu", kernel_initializer="lecun_normal"
    )
    x = layers.AveragePooling2D(pool_size=(1, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        8, (1, 15), activation="selu", kernel_initializer="lecun_normal"
    )
    x = layers.AveragePooling2D(pool_size=(1, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        4, (1, 15), activation="selu", kernel_initializer="lecun_normal"
    )
    x = layers.AveragePooling2D(pool_size=(1, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    output = layers.Dense(32, kernel_initializer="lecun_normal")(x)

    return keras.Model(inputs=input, outputs=output, name=model_name)
