"""SSVEPFormer model definition in Keras (Keras Applications Style).

This module implements the architecture described in the SSVEPFormer preprint
(`arXiv:2210.04172v1`) with a keras.applications-like API.

Reference:
    Chen et al., "A Transformer-based deep neural network model for SSVEP classification"
    https://arxiv.org/abs/2210.04172
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Literal, override

import keras
import tensorflow as tf
from keras import activations, backend, initializers, layers, ops


def _paper_initializer() -> initializers.RandomNormal:
    """Return the weight initializer used in the original SSVEPFormer paper."""
    return initializers.RandomNormal(mean=0.0, stddev=0.1)


class ComplexSpectrum(layers.Layer):
    """Complex spectrum representation layer (Fixed FFT transformation)."""

    def __init__(
        self,
        *,
        sampling_rate: float,
        band_hz: tuple[float, float],
        fft_length: int | None = None,
        target_df: float | None = None,
        prefer_pow2: bool = True,
        **kwargs: Any,
    ) -> None:
        # Force non-trainable to avoid conflict during deserialization
        kwargs.pop("trainable", None)
        super().__init__(trainable=False, **kwargs)

        if sampling_rate <= 0:
            raise ValueError(f"`sampling_rate` must be > 0, got {sampling_rate}.")
        self.sampling_rate = float(sampling_rate)

        if len(band_hz) != 2 or not (0.0 < band_hz[0] < band_hz[1]):
            raise ValueError(f"Invalid `band_hz`: {band_hz}")
        self.band_hz = (float(band_hz[0]), float(band_hz[1]))

        if (fft_length is None) == (target_df is None):
            raise ValueError("Provide exactly one of `fft_length` or `target_df`.")
        self.prefer_pow2 = bool(prefer_pow2)
        self.target_df = None if target_df is None else float(target_df)

        if fft_length is None:
            import math

            n = math.ceil(self.sampling_rate / self.target_df)
            if self.prefer_pow2:
                p = 1
                while p < n:
                    p <<= 1
                n = p
            self.fft_length = int(n)
        else:
            self.fft_length = int(fft_length)

        # State for bins
        self._bin_start: int | None = None
        self._bin_end: int | None = None
        self._n_bins: int | None = None

    @override
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        fs = self.sampling_rate
        df = fs / self.fft_length
        low, high = self.band_hz

        import math

        bin_start = int(math.ceil(low / df))
        bin_end = int(math.floor(high / df))
        max_bin = self.fft_length // 2
        bin_start = max(0, min(bin_start, max_bin))
        bin_end = max(0, min(bin_end, max_bin))

        if bin_end <= bin_start:
            raise ValueError(
                f"Empty band after Hz->bin mapping. bins=[{bin_start}, {bin_end}]."
            )

        self._bin_start = bin_start
        self._bin_end = bin_end
        self._n_bins = bin_end - bin_start + 1
        super().build(input_shape)

    @override
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        real, imag = ops.rfft(inputs, fft_length=self.fft_length)
        bs, be = self._bin_start, self._bin_end
        if bs is None or be is None:
            raise RuntimeError("Layer is not built yet.")

        # Slicing: [Batch, Channels, Bins]
        real_band = real[:, :, bs : be + 1]
        imag_band = imag[:, :, bs : be + 1]

        # Concatenate features: [Batch, Channels, 2*Bins]
        feats = ops.concatenate([real_band, imag_band], axis=-1)

        # Transpose to [Batch, Features, Channels] for downstream 1D Conv
        return ops.transpose(feats, (0, 2, 1))

    @override
    def compute_output_shape(
        self, input_shape: tuple[int | None, int | None, int | None]
    ) -> tuple[int | None, int, int | None]:
        b, c, _ = input_shape
        n_bins = self._n_bins
        if n_bins is None:
            # Fallback estimation
            fs = self.sampling_rate
            df = fs / self.fft_length
            import math

            low, high = self.band_hz
            n_bins = max(1, math.floor(high / df) - math.ceil(low / df) + 1)
        return (b, 2 * n_bins, c)

    @override
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "band_hz": self.band_hz,
                "fft_length": self.fft_length,
                "target_df": self.target_df,
                "prefer_pow2": self.prefer_pow2,
            }
        )
        return config


class ChannelCombinationBlock(layers.Layer):
    """Channel combination block (Infers C -> 2C)."""

    def __init__(self, dropout_rate: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.channel_mixer: layers.Conv1D | None = None
        self.layer_norm: layers.LayerNormalization | None = None
        self.gelu: layers.Activation | None = None
        self.dropout: layers.Dropout | None = None

    @override
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        input_channels = input_shape[-1]
        self.channel_mixer = layers.Conv1D(
            filters=2 * input_channels,
            kernel_size=1,
            padding="same",
            kernel_initializer=_paper_initializer(),
            name="channel_mixer",
        )
        self.layer_norm = layers.LayerNormalization(name="ln")
        self.gelu = layers.Activation(activation=activations.get("gelu"))
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    @override
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.channel_mixer is None:
            raise RuntimeError("Layer not built.")
        x = self.channel_mixer(inputs)
        x = self.layer_norm(x)
        x = self.gelu(x)
        return self.dropout(x, training=training)

    @override
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape((input_shape[0], input_shape[1], 2 * input_shape[2]))

    @override
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"dropout_rate": self.dropout_rate})
        return config


class ConvResidualBlock(layers.Layer):
    """CNN residual module."""

    def __init__(self, dropout_rate: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.pre_norm: layers.LayerNormalization | None = None
        self.conv: layers.Conv1D | None = None
        self.post_conv_norm: layers.LayerNormalization | None = None
        self.gelu: layers.Activation | None = None
        self.dropout: layers.Dropout | None = None

    @override
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        channels = input_shape[-1]
        self.pre_norm = layers.LayerNormalization(name="pre_norm")
        self.conv = layers.Conv1D(
            filters=channels,
            kernel_size=31,
            padding="same",
            kernel_initializer=_paper_initializer(),
            name="conv",
        )
        self.post_conv_norm = layers.LayerNormalization(name="post_norm")
        self.gelu = layers.Activation(activation=activations.get("gelu"))
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    @override
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.conv is None:
            raise RuntimeError("Layer not built.")
        residual = inputs
        x = self.pre_norm(inputs)
        x = self.conv(x)
        x = self.post_conv_norm(x)
        x = self.gelu(x)
        x = self.dropout(x, training=training)
        return x + residual

    @override
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"dropout_rate": self.dropout_rate})
        return config


class ChannelWiseMLPBlock(layers.Layer):
    """Channel-wise MLP residual module."""

    def __init__(self, dropout_rate: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.pre_norm: layers.LayerNormalization | None = None
        self.channel_dense: layers.Dense | None = None
        self.gelu: layers.Activation | None = None
        self.dropout: layers.Dropout | None = None
        self.concat_channels: layers.Concatenate | None = None

    @override
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        spectral_features = input_shape[-2]
        self.pre_norm = layers.LayerNormalization(name="pre_norm")
        self.channel_dense = layers.Dense(
            units=spectral_features,
            kernel_initializer=_paper_initializer(),
            name="channel_dense",
        )
        self.gelu = layers.Activation(activation=activations.get("gelu"))
        self.dropout = layers.Dropout(self.dropout_rate)
        self.concat_channels = layers.Concatenate(axis=-1)
        super().build(input_shape)

    @override
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.channel_dense is None:
            raise RuntimeError("Layer not built.")
        residual = inputs
        channel_count = inputs.shape[-1]
        normalized = self.pre_norm(inputs)

        per_channel_outputs = []
        for channel_index in range(channel_count):
            channel_slice = normalized[:, :, channel_index]
            projected = self.channel_dense(channel_slice)
            projected = layers.Reshape((-1, 1))(projected)
            per_channel_outputs.append(projected)

        x = self.concat_channels(per_channel_outputs)
        x = self.gelu(x)
        x = self.dropout(x, training=training)
        return x + residual

    @override
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"dropout_rate": self.dropout_rate})
        return config


class EncoderBlock(layers.Layer):
    """SSVEPFormer encoder block container."""

    def __init__(self, dropout_rate: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.cnn_residual = ConvResidualBlock(dropout_rate=dropout_rate)
        self.channelwise_mlp = ChannelWiseMLPBlock(dropout_rate=dropout_rate)

    @override
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.cnn_residual(inputs, training=training)
        return self.channelwise_mlp(x, training=training)

    @override
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"dropout_rate": self.dropout_rate})
        return config


def SSVEPFormer(
    # Domain-specific args
    srate: float = 250.0,
    band_hz: tuple[float, float] = (8.0, 64.0),
    fft_res_hz: float = 0.25,
    dropout_rate: float = 0.5,
    # Standard keras.applications args
    include_top: bool = True,
    weights: str | pathlib.Path | None = None,
    input_tensor: tf.Tensor | keras.KerasTensor | None = None,
    input_shape: tuple[int, int] | None = None,
    pooling: Literal["avg", "max"] | None = None,
    classes: int = 12,
    classifier_activation: str | Callable | None = "softmax",
    name: str = "ssvepformer",
) -> keras.Model:
    """Instantiates the SSVEPFormer architecture.

    Args:
        srate: Sampling rate of the EEG data in Hz.
        band_hz: Tuple of (low, high) cutoff frequencies for spectral features.
        fft_res_hz: Desired frequency resolution (used to calculate FFT length).
        dropout_rate: Dropout rate for internal blocks.
        include_top: Whether to include the fully-connected layer at the top of the network.
        weights: Path to a weights file to load.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: Optional shape tuple, only to be specified if `include_top` is False or if `input_tensor` is None.
            It should be (channels, samples).
        pooling: Optional pooling mode for feature extraction when `include_top` is `False`.
            - `None`: means that the output of the model will be the 3D tensor output of the last block.
            - `avg`: means that global average pooling will be applied.
            - `max`: means that global max pooling will be applied.
        classes: Optional number of classes to classify images into, only to be specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to use on the "top" layer.
            Ignored unless `include_top=True`. Set to `None` to return logits.
        name: The name of the model.

    Returns:
        A `keras.Model` instance.
    """
    # 1. Input handling
    if input_tensor is None:
        if input_shape is None:
            # Default fallback if nothing provided, though for EEG shape varies
            # widely so this is just a placeholder suggestion.
            input_shape = (8, int(srate))  # 1 second of data
        eeg_input = keras.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            eeg_input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            eeg_input = input_tensor

    # 2. Body Architecture
    x = ComplexSpectrum(
        sampling_rate=srate,
        band_hz=band_hz,
        target_df=fft_res_hz,
        name="complex_spectrum",
    )(eeg_input)

    x = ChannelCombinationBlock(dropout_rate=dropout_rate, name="channel_combination")(
        x
    )

    x = EncoderBlock(dropout_rate=dropout_rate, name="encoder_block_1")(x)
    x = EncoderBlock(dropout_rate=dropout_rate, name="encoder_block_2")(x)

    # 3. Head / Top
    if include_top:
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dropout(dropout_rate, name="top_dropout_1")(x)
        x = layers.Dense(
            6 * classes,
            kernel_initializer=_paper_initializer(),
            name="top_dense_1",
        )(x)
        x = layers.LayerNormalization(name="top_norm_1")(x)
        x = layers.Activation(activations.get("gelu"), name="top_gelu_1")(x)
        x = layers.Dropout(dropout_rate, name="top_dropout_2")(x)

        x = layers.Dense(
            classes,
            kernel_initializer=_paper_initializer(),
            name="predictions",
        )(x)

        if classifier_activation is not None:
            x = layers.Activation(
                activations.get(classifier_activation), name="predictions_activation"
            )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

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
    model = SSVEPFormer(
        srate=250,
        input_shape=(9, 250),  # 9 channels, 1 second
        classes=12,
        include_top=True,
    )
    model.summary()
