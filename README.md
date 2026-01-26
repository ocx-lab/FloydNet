# FloydNet
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)

Official implementation of [FloydNet](https://openreview.net/pdf?id=aUsx1G6RVQ).

![Figure Pivotal Attention Mechanism for 2-Floyd/3-Floyd.](misc/pivotalattn2&3.png)

This repository serves two audiences:
- **Engineering users**: Reusable PyTorch components (functional attention APIs and Transformer-style blocks) under `src/`.
- **Research users**: Scripts/configs to reproduce paper experiments (TSP, Graph Isomorphism, BREC) under `example/`.

---

## Introduction

FloydNet is the official PyTorch implementation.
The repository provides:

1. **Reusable components**: a drop-in attention/Transformer-block interface intended for integration into existing projects.
2. **Reproduction code**: end-to-end training/evaluation pipelines to reproduce the benchmarks reported in the paper.

For algorithmic details, hyperparameter choices, and analysis, please refer to the paper (TODO: link).

---

## Repository Structure

- `src/floydnet/`  
  **Library code for reuse**  
  Contains the functional attention API and module/block implementations.

- `example/`  
  **Experiment reproduction code**  
  Includes benchmark-specific scripts, configs, and data preparation utilities.

---

### Installation

#### Option A: Install from PyPI
```bash
pip install floydnet
```

#### Option B: Install from source
```bash
git clone git@github.com:ocx-lab/FloydNet.git
cd FloydNet
pip install -e .
```

> Requirements: Python `>= 3.9`, PyTorch `>= 2.1` (see `pyproject.toml`).

### Public API

FloydNet re-exports the public API from `src/floydnet/__init__.py`, so you can import from the top-level package:

- **Functional API**:
  - `pivotal_attention` (see `src/floydnet/functional.py`)
- **Module / block API**:
  - `PivotalAttentionBlock` (see `src/floydnet/transformer.py`)

```python
from floydnet import pivotal_attention, PivotalAttentionBlock
```

### Minimal usage example

```python
import torch
from floydnet import pivotal_attention, PivotalAttentionBlock

# -------------------------
# Module API (Transformer-style block)
# Input is a 2D grid: (B, N, N, C)
# -------------------------
B, N, C = 2, 16, 64
x = torch.randn(B, N, N, C)

m = PivotalAttentionBlock(embed_dim=C, num_heads=8, dropout=0.0)
out = m(x)  # (B, N, N, C)
print(out.shape)

# -------------------------
# Functional API
# All inputs are 5D: (B, H, N, N, D)
# -------------------------
B, H, N, D = 2, 8, 16, 64
q_ik = torch.randn(B, H, N, N, D)
k_ij = torch.randn(B, H, N, N, D)
k_jk = torch.randn(B, H, N, N, D)
v_ij = torch.randn(B, H, N, N, D)
v_jk = torch.randn(B, H, N, N, D)

y = pivotal_attention(q_ik, k_ij, k_jk, v_ij, v_jk)  # (B, H, N, N, D)
print(y.shape)
```

---

## Reproducing Paper Results

This section targets **research users** who want to reproduce the experiments in the paper.

See `example/README.md` For detailed description.

### Environment setup

We recommend using `uv` to create an isolated environment for the reproduction code under `example/`.

```bash
cd /path/to/FloydNet

# 1) Create a uv virtual environment with Python 3.12
uv venv --python 3.12

# 2) Activate it
source .venv/bin/activate

# 3) Install extra dependencies for reproducing paper experiments
uv pip install -r example/requirements.txt

# 4) Install FloydNet (editable) for local development / imports
uv pip install -e .
```

## Changelog (latest)

- Full release with training and evaluation scripts for Graph Count, BREC, and TSP.
- Added `pivotal_attention3` functional API for 3-Floyd attention.
- Added additional configuration options in `PivotalAttentionBlock`.

The full changelog is in [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO},
  url       = {TODO}
}
```

(Alternatively, see [CITATION.cff](CITATION.cff).)

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).