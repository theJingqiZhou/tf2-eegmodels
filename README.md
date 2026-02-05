# Introduction

This is a fork of the Army Research Laboratory (ARL) EEGModels project: A Collection of Neural Network models for EEG signal processing and classification, written in Keras 3 and Tensorflow. The aim of this project is still

- provide a set of well-validated models for EEG signal processing and classification
- facilitate reproducible research and
- enable other researchers to use and compare these models as easy as possible on their data

# Requirements

Defaults:

- python >= 3.9, < 3.13
- tensorflow >= 2.16.2 (Linux)
- tensorflow-macos == 2.16.2 (Darwin)
- keras > 3

With [gpu] option:

- Python >= 3.9, < 3.13
- tensorflow[and-cuda] >= 2.16.2 (Linux)
- tensorflow-macos == 2.16.2 (Darwin)
- tensorflow-metal >= 1.1.0 (Darwin)
- keras > 3

> Code formatter is black, and import fix is done via isort.

# Models Implemented

- EEGNet [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013). Both the original model and the revised model are implemented.
- DeepConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)

# Usage

To use this package, clone this repository to local and run `pip install <path/to/tf2-eegmodels/clone>`. Then, one can simply import any model and configure it as

```python
import keras
import tensorflow as tf
from tf2_eegmodels import models as eegmodels

srate: int = ...

channels: int = ...

samples: int = ...

# According to the original paper, the parameters of some layers are defined based on the sampling rate.
model_eegnet_v4 = eegmodels.EEGNetV4(
    srate=srate, input_shape=(channels, samples), classes=classes
)
# To replace the final softmax activation.
model_eegnet_v4_out_logits = eegmodels.EEGNetV4(
    srate=srate,
    input_shape=(channels, samples),
    classes=classes,
    classifier_activation="linear",
)

# What if we do not want to flat feature map at the top of model?
model_eegnet_v4_no_top = eegmodels.EEGNetV4(
    srate=srate, input_shape=(channels, samples), include_top=False
)
# How about reduce the final feature map via global pooling?
model_eegnet_v4_pool_top = eegmodels.EEGNetV4(
    srate=srate, input_shape=(channels, samples), include_top=False, pooling="avg"
)

eeg_tensor: tf.Tensor = ...  # In shape (channels, samples).

# We can also use actual tensor to aid model creation.
model_eegnet = eegmodels.EEGNet(
    input_tensor=eeg_tensor, input_shape=(channels, samples), classes=classes
)

eeg_keras_tensor: keras.KerasTensor = keras.Input(shape=(channels, samples))

# And of course a symbolic tensor (KerasTensor).
model_shallowconvnet = eegmodels.ShallowConvNet(
    input_tensor=eeg_keras_tensor, classes=classes
)

weights_path: str | pathlib.Path = ...

# To use pretrained weights.
model_deepconvnet = eegmodels.DeepConvNet(
    weights=weights_path, input_shape=(channels, samples), classes=classes
)
```

Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively). Then fit the model and predict on new test data.

```python
model_eegnet_v4.compile(loss="categorical_crossentropy", optimizer="adam")
fittedModel = model_eegnet_v4.fit(...)
predicted = model_eegnet_v4.predict(...)
```

# Paper Citation

If you use the EEGNet model in your research and found it helpful, please cite the following paper:

```
@article{Lawhern2018,
  author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
  title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
  year={2018}
}
```

Similarly, if you use the ShallowConvNet or DeepConvNet models and found them helpful, please cite the following paper:

```
@article{hbm23730,
author = {Schirrmeister Robin Tibor and
          Springenberg Jost Tobias and
          Fiederer Lukas Dominique Josef and
          Glasstetter Martin and
          Eggensperger Katharina and
          Tangermann Michael and
          Hutter Frank and
          Burgard Wolfram and
          Ball Tonio},
title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
journal = {Human Brain Mapping},
volume = {38},
number = {11},
pages = {5391-5420},
keywords = {electroencephalography, EEG analysis, machine learning, end‐to‐end learning, brain–machine interface, brain–computer interface, model interpretability, brain mapping},
doi = {10.1002/hbm.23730},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730}
}
```

# Original Legal Disclaimer

The original project is governed by the terms of the Creative Commons Zero 1.0 Universal (CC0 1.0) Public Domain Dedication (the Agreement). You should have received a copy of the Agreement with a copy of this software. If not, see https://github.com/USArmyResearchLab/ARLDCCSO. Your use or distribution of ARL EEGModels, in both source and binary form, in whole or in part, implies your agreement to abide by the terms set forth in the Agreement in full.

Other portions of this project are subject to domestic copyright protection under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 license.  The complete text of the license governing this material is in the file labeled LICENSE.TXT that is a part of this project's official distribution.

arl-eegmodels is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You may find the full license in the file LICENSE in this directory.
