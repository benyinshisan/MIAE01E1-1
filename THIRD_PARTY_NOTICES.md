# Third-party notices

Third-party components used in this repository, with their licences.

## Code adapted into this repository

| Component | Commit / version | Licence | Used in |
|---|---|---|---|
| [cleanrl](https://github.com/vwxyzjn/cleanrl) | `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` | MIT | `envs.py`, `buffer.py`, `network.py` (structure) |
| [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | — | MIT | upstream of cleanrl's Atari wrappers and replay buffer; structure reference for `network.py` |

## Runtime dependencies (installed via pip, not redistributed)

| Package | Version | Licence |
|---|---|---|
| `ale-py` | 0.10.1 | GPLv2 |
| `gymnasium` | 1.0.0 | MIT |
| `numpy` | 2.5.4 | BSD-3-Clause |
| `torch` | 2.4.1 | BSD-3-Clause |
| `opencv-python-headless` | 5.0.0.93 | Apache-2.0 |
| `tensorboard` | 2.21.0 | Apache-2.0 |

## Licence texts

### cleanrl — MIT

```text
MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### stable-baselines3 — MIT

```text
The MIT License

Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Paper

Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* **Human-level control through deep
reinforcement learning.** *Nature* **518**, 529–533 (2015).
DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
