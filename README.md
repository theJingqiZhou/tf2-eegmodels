# tf2-eegmodels

## Introduction

This is a fork of the Army Research Laboratory (ARL) EEGModels project: A Collection of Neural Network models for EEG signal processing and classification, written in **Keras 3** and **TensorFlow**. The aim of this project is to:

- provide a set of well-validated models for EEG signal processing and classification (including recent Transformers and Contrastive Learning models).
- facilitate reproducible research.
- enable other researchers to use and compare these models as easily as possible on their data.

## Requirements

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

## Models Implemented

- **EEGNet** [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013): Both the original model and the revised model (v4) are implemented.
- **DeepConvNet** [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- **ShallowConvNet** [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- **BrainNet** [[3]](https://doi.org/10.1109/PERCOM56429.2023.10099367): Siamese Network for Biometric Recognition.
- **SSVEPFormer** [[4]](https://arxiv.org/abs/2210.04172): Transformer-based model for SSVEP classification.
- **TaskOriNet** [[5]](https://arxiv.org/abs/2207.01391): Task-oriented Self-supervised Learning (ResNet-based) for Anomaly Detection.

## Usage

To use this package, clone this repository to local and run `pip install <path/to/tf2-eegmodels/clone>`. Then, one can simply import any model and configure it.

Example usages:

```python
import keras
import tensorflow as tf
from tf2_eegmodels import models as eegmodels

srate: int = 250
channels: int = 30
samples: int = 1025
classes: int = 4

# --- EEGNet V4 ---
# According to the original paper, the parameters of some layers are defined based on the sampling rate.
model_eegnet_v4 = eegmodels.EEGNetV4(
    srate=srate, input_shape=(channels, samples), classes=classes
)

# --- DeepConvNet with Pretrained Weights ---
# weights_path: str | pathlib.Path = ...
# model_deepconvnet = eegmodels.DeepConvNet(
#     weights=weights_path, input_shape=(channels, samples), classes=classes
# )

# --- BrainNet (Siamese/Biometric) ---
# Note: Input shape handling in BrainNet automatically expands dims
model_brainnet = eegmodels.BrainNet(
    input_shape=(channels, samples),
    classes=classes,
    initial_downsample=False, # Set True for high sampling rates (~1000Hz)
)

# --- SSVEPFormer (Transformer) ---
model_ssvepformer = eegmodels.SSVEPFormer(
    srate=srate,
    input_shape=(channels, samples),
    classes=classes,
    fft_res_hz=0.5,
)

# --- TaskOriNet (Self-Supervised / Anomaly Detection) ---
model_taskorinet = eegmodels.TaskOriNet(
    input_shape=(channels, samples),
    classes=3, # Typically 3 classes for the SSL task
)
```

Compile the model with the associated loss function and optimizer (e.g., categorical cross-entropy and Adam optimizer). Then fit the model and predict on new test data.

```python
model_eegnet_v4.compile(loss="categorical_crossentropy", optimizer="adam")
# fittedModel = model_eegnet_v4.fit(...)
# predicted = model_eegnet_v4.predict(...)
```

## Paper Citation

If you use the **EEGNet** model in your research and found it helpful, please cite the following paper:

```bibtex
@article{Lawhern2018,
    author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
    title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
    journal={Journal of Neural Engineering},
    volume={15},
    number={5},
    pages={056013},
    url={[http://stacks.iop.org/1741-2552/15/i=5/a=056013](http://stacks.iop.org/1741-2552/15/i=5/a=056013)},
    year={2018},
}
```

If you use the **ShallowConvNet** or **DeepConvNet** models, please cite:

```bibtex
@article{hbm23730,
    author = {Schirrmeister Robin Tibor and Springenberg Jost Tobias and Fiederer Lukas Dominique Josef and Glasstetter Martin and Eggensperger Katharina and Tangermann Michael and Hutter Frank and Burgard Wolfram and Ball Tonio},
    title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
    journal = {Human Brain Mapping},
    volume = {38},
    number = {11},
    pages = {5391-5420},
    doi = {10.1002/hbm.23730},
    url = {[https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730](https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730)},
    year = {2017},
}
```

If you use **BrainNet**, please cite:

```bibtex
@inproceedings{Fallahi2023,
    author={Fallahi, Arezou and et al.},
    title={BrainNet: Improving Brainwave-based Biometric Recognition with Siamese Networks},
    booktitle={2023 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
    year={2023},
    doi={10.1109/PERCOM56429.2023.10099367},
}
```

If you use **SSVEPFormer**, please cite:

```bibtex
@article{Chen2022,
    author={Chen, Pan and et al.},
    title={A Transformer-based deep neural network model for SSVEP classification},
    journal={arXiv preprint arXiv:2210.04172},
    year={2022},
    url={[https://arxiv.org/abs/2210.04172](https://arxiv.org/abs/2210.04172)},
}
```

If you use **TaskOriNet**, please cite:

```bibtex
@article{Zheng2022,
    author={Zheng, W. and et al.},
    title={Task-oriented Self-supervised Learning for Anomaly Detection in Electroencephalography},
    journal={arXiv preprint arXiv:2207.01391},
    year={2022},
    url={[https://arxiv.org/abs/2207.01391](https://arxiv.org/abs/2207.01391)},
}
```

## Original Legal Disclaimer

The original project is governed by the terms of the Creative Commons Zero 1.0 Universal (CC0 1.0) Public Domain Dedication (the Agreement). You should have received a copy of the Agreement with a copy of this software. If not, see https://github.com/USArmyResearchLab/ARLDCCSO. Your use or distribution of ARL EEGModels, in both source and binary form, in whole or in part, implies your agreement to abide by the terms set forth in the Agreement in full.

Other portions of this project are subject to domestic copyright protection under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 license.  The complete text of the license governing this material is in the file labeled LICENSE.TXT that is a part of this project's official distribution.

arl-eegmodels is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You may find the full license in the file LICENSE in this directory.