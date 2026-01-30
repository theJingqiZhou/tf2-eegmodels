# Introduction
This is a fork of the Army Research Laboratory (ARL) EEGModels project: A Collection of Neural Network models for EEG signal processing and classification, written in Keras 2 and Tensorflow. The aim of this project is to

- provide a set of well-validated models for EEG signal processing and classification
- facilitate reproducible research and
- enable other researchers to use and compare these models as easy as possible on their data

# Requirements

- Python == 3.11
- tensorflow == 2.18.X

# Models Implemented

- EEGNet [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013). Both the original model and the revised model are implemented.
- DeepConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)


# Usage

To use this package, place the contents of this folder in your PYTHONPATH environment variable. Then, one can simply import any model and configure it as


```python

from tf2_eegmodels import models

model  = models.EEGNet(input_shape = ..., classes = ...)

model2 = models.ShallowConvNet(input_shape = ..., classes = ...)

model3 = models.DeepConvNet(input_shape = ..., classes = ...)

```

Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively). Then fit the model and predict on new test data.

```python

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
fittedModel    = model.fit(...)
predicted      = model.predict(...)

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
