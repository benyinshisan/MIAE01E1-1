# Third-party notices

This repository contains three kinds of material. They are licensed differently, so they are listed separately.

1. **Our own code** — written for this course replication (`miae01_dqn/`, `tests/`, `docs/`). Provided for course assessment.
2. **Code derived from upstream projects** — MIT-licensed, with the original copyright notices retained. Listed in §1 and §2 below.
3. **Runtime dependencies** — installed by pip/conda, **not** redistributed in this repository. Listed in §3.

---

## 1. cleanrl

Reference implementation used as the starting point.

- Repository: https://github.com/vwxyzjn/cleanrl
- Commit: `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`
- Files referenced: `cleanrl/dqn_atari.py`, `cleanrl_utils/atari_wrappers.py`, `cleanrl_utils/buffers.py`
- License: MIT

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

---

## 2. Stable-Baselines3

cleanrl's `atari_wrappers.py` and `buffers.py` — the two files our `envs.py` and `buffer.py` derive from — are themselves adapted from Stable-Baselines3. The SB3 copyright notice is therefore retained alongside cleanrl's.

- Repository: https://github.com/DLR-RM/stable-baselines3
- Upstream files: `stable_baselines3/common/atari_wrappers.py`, `stable_baselines3/common/buffers.py`
- License: MIT

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

Per-file provenance is recorded in the header comment of each derived file and in the README's "Files modified" table.

---

## 3. Runtime dependencies (not redistributed here)

Installed from `environment.yml` / `requirements.txt`; no source is copied into this repository.

| Package | Version (pinned) | License |
|---|---|---|
| `ale-py` | 0.10.1 | **GPLv2** — see note below |
| `gymnasium` | 1.0.0 | MIT |
| `numpy` | 2.5.4 | BSD-3-Clause |
| `torch` | 2.4.1 (local) / Colab build | BSD-3-Clause |
| `opencv-python-headless` | 5.0.0.93 | Apache-2.0 |
| `tensorboard` | 2.21.0 | Apache-2.0 |

**Note on `ale-py` (GPLv2).** The Arcade Learning Environment is GPLv2, and the Atari ROMs we use are the ones bundled inside the `ale-py` wheel. We do **not** vendor or redistribute `ale-py` or the ROMs in this repository; users install the package themselves. If this code were ever distributed as a binary bundle including ALE, the GPL terms of that dependency would need to be revisited. For the purposes of this course submission the dependency is installed, not shipped.

---

## 4. Paper

Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* **Human-level control through deep reinforcement learning.** *Nature* **518**, 529–533 (2015). DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)

The paper's original implementation was never publicly released; its "Code availability" note points to a DeepMind-internal address that is not publicly accessible.
