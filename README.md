# Torch-Yin
This package implements the Yin pitch estimation algorithm described by
[A De Cheveigné, et al](https://asa.scitation.org/doi/10.1121/1.1458024)
for the [PyTorch](https://pytorch.org/) deep learning framework. It is based
on the excellent NumPy implementation by
[Patrice Guyot](https://github.com/patriceguyot/Yin), which has been extended
for full vectorization and to support batched computation.

## Status
[![PyPI](https://badge.fury.io/py/torch-yin.svg)](https://badge.fury.io/py/torch-yin)
![Tests](https://github.com/brentspell/torch-yin/actions/workflows/test.yml/badge.svg)
[![Coveralls](https://coveralls.io/repos/github/brentspell/torch-yin/badge.svg?branch=main)](https://coveralls.io/repos/github/brentspell/torch-yin/badge.svg?branch=main)

### Install with pip
```bash
pip install torch-yin
```

## Usage
Here we estimate the fundamental frequency of a simple 1 second sinusoid at
440Hz:

```python
import torch
import torchyin

FS = 48000

y = torch.sin(2 * torch.pi * 440 / FS * torch.arange(FS))

pitch = torchyin.estimate(y, sample_rate=FS)

pitch[0]
```

```python
tensor(440.3669)
```

Pitch can also be calculated for batches of signals, shaped `[batch, samples]`.
In this example, we create a batch of signals for the 88 standard piano keys
using broadcasting, shaped `[88, 48000]`.

```python
import torch
import torchyin

FS = 48000

f = 2 ** ((torch.arange(88) - 48) / 12) * 440
t = torch.arange(FS)
y = torch.sin(2 * torch.pi * f.unsqueeze(1) / FS * t.unsqueeze(0))

pitch = torchyin.estimate(
    y,
    sample_rate=FS,
    pitch_min=20,
    pitch_max=5000,
)

pitch[:, 0]
```

```python
tensor([  27.5072,   29.1439,   30.8682,   32.6975,   34.6570,   36.6973,
          38.8979,   41.2017,   43.6364,   46.2428,   48.9796,   51.8919,
          54.9828,   58.2524,   61.6967,   65.3951,   69.2641,   73.3945,
          77.7958,   82.4742,   87.2727,   92.4855,   97.9592,  103.8961,
         110.0917,  116.5049,  123.3933,  130.7902,  138.7283,  146.7890,
         155.3398,  164.9485,  174.5455,  185.3282,  195.9184,  207.7922,
         220.1835,  233.0097,  247.4227,  262.2951,  277.4566,  294.4785,
         311.6883,  328.7671,  350.3650,  369.2308,  393.4426,  413.7931,
         440.3669,  466.0194,  494.8453,  521.7391,  551.7241,  585.3658,
         623.3766,  657.5342,  695.6522,  738.4615,  786.8852,  827.5862,
         872.7272,  941.1765,  979.5918, 1043.4783, 1116.2791, 1170.7317,
        1230.7693, 1333.3334, 1411.7648, 1500.0000, 1548.3871, 1655.1724,
        1777.7778, 1846.1539, 2000.0000, 2086.9565, 2181.8184, 2400.0000,
        2526.3159, 2666.6667, 2823.5295, 3000.0000, 3200.0002, 3428.5715,
        3428.5715, 3692.3079, 4000.0000, 4363.6367])
```

For more information and detailed parameter descriptions, please check
out [this blog post](https://brentspell.com/2022/torch-yin/), see the
[module documentation](https://github.com/brentspell/torch-yin/blob/main/torchyin/yin.py),
or run `help(torchyin)`.

## Development

### Setup
The following script creates a virtual environment using
[pyenv](https://github.com/pyenv/pyenv) for the project and installs
dependencies.

```bash
pyenv install 3.9.10
pyenv virtualenv 3.9.10 torch-yin
pip install -r requirements.txt
```

You can then run tests, etc. follows:

```bash
pytest --cov=torchyin
black .
flake8 .
mypy torchyin
```

These can also be used with the [pre-commiit](https://pypi.org/project/pre-commit/)
library to run all checks at commit time.

### Deployment
The project uses setup.py for installation and is deployed to
[PyPI](https://pypi.org/project/torch-yin). The source distribution can be
built for deployment with the following command:

```bash
python setup.py clean --all
rm -r ./dist
python setup.py sdist
```

The distribution can then be uploaded to PyPI using twine.

```bash
twine upload --repository-url=https://upload.pypi.org/legacy/ dist/*
```

For deployment testing, the following command can be used to upload to the
PyPI test repository:

```bash
twine upload --repository-url=https://test.pypi.org/legacy/ dist/*
```

## License
Copyright © 2022 Brent M. Spell

Licensed under the MIT License (the "License"). You may not use this
package except in compliance with the License. You may obtain a copy of the
License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
