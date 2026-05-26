<div align="center">

# SpecPL: Disentangling Spectral Granularity for Prompt Learning

**Jingtao Zhou\*, Xirui Kang\*, Feiyang Huang\*, Lai-Man Po†**  
(*City University of Hong Kong*)  
<small>\* Equal Contribution &nbsp;&nbsp; † Corresponding Author</small>

<br>

[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](#)
[![Paper](https://img.shields.io/badge/arXiv-2605.04504-b31b1b.svg)](https://arxiv.org/abs/2605.04504)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation of "SpecPL: Disentangling Spectral Granularity for Prompt Learning" (ICML 2026).** Paper: [arXiv:2605.04504](https://arxiv.org/abs/2605.04504).

</div>

## Overview

SpecPL is a granularity-aware plug-in for CLIP-based prompt learning. It uses a frozen VAE as a spatial-spectral proxy to separate low-frequency semantic cues from high-frequency discriminative details, then uses these signals to guide prompt learning during training. SpecPL adds no inference-time overhead after training.

This repository currently supports SpecPL integrations for:

| Family | Vanilla trainer | SpecPL trainer | Entry point |
| :-- | :-- | :-- | :-- |
| CoOp | `trainers/coop.py` (`CoOp`) | `trainers/coop_specpl.py` (`CoOpSpecPL`) | `train.py` |
| CoCoOp | `trainers/cocoop.py` (`CoCoOp`) | `trainers/cocoop_specpl.py` (`CoCoOpSpecPL`) | `train.py` |
| MaPLe | `trainers/maple.py` (`MaPLe`) | `trainers/maple_specpl.py` (`MaPLeSpecPL`) | `train.py` |
| MMRL | `MMRL/trainers/mmrl.py` (`MMRL`) | `MMRL/trainers/mmrl_specpl.py` (`MMRLSpecPL`) | `MMRL/train.py` |

Final benchmark numbers will be added after the camera-ready release.

## Requirements

We recommend Python 3.9 and PyTorch 2.1.2 with CUDA 11.8.

```bash
conda create -n specpl python=3.9 -y
conda activate specpl

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
python setup.py develop
cd ..
```

## Repository Layout

MMRL keeps its own entry point and configs because it uses a different trainer stack. CoOp, CoCoOp, and MaPLe share the top-level entry point.

```text
SpecPL-Prompt-Learning/
├── train.py                         # CoOp / CoCoOp / MaPLe entry point
├── configs/
│   ├── datasets/
│   └── trainers/
├── datasets/
├── trainers/                        # CoOp / CoCoOp / MaPLe + SpecPL
│   ├── coop.py
│   ├── coop_specpl.py
│   ├── cocoop.py
│   ├── cocoop_specpl.py
│   ├── maple.py
│   └── maple_specpl.py
├── scripts/
│   ├── coop/
│   ├── cocoop/
│   └── maple/
├── docs/
└── MMRL/
    ├── train.py                     # MMRL entry point
    ├── configs/
    ├── datasets/
    ├── trainers/
    │   ├── mmrl.py
    │   └── mmrl_specpl.py
    └── scripts/mmrl/
```

## Paths And Environment Variables

All public scripts use placeholder paths and can be configured through environment variables:

```bash
export DATA_ROOT=path/to/data        # dataset root containing the prompt-learning datasets
export CLIP_ROOT=path/to/clip        # optional cache directory for OpenAI CLIP weights
export HF_ENDPOINT=https://...       # optional HuggingFace mirror, if needed
```

If `CLIP_ROOT` is not set, CLIP weights are cached under `~/.cache/clip`. The VAE teacher used by SpecPL defaults to `REPA-E/e2e-qwenimage-vae` and follows the standard HuggingFace cache behavior. For offline use, set `VAE_PRETRAINED_ID` in the corresponding config to a local model path.

## Running Base-to-Novel Experiments

All maintained Base-to-Novel scripts take a dataset name as the first argument and a seed as the second argument.

```bash
export DATA_ROOT=path/to/data
export CLIP_ROOT=path/to/clip
```

### CoOp

```bash
sh scripts/coop/vanilla_base2new_train.sh imagenet 1
sh scripts/coop/vanilla_base2new_test.sh  imagenet 1

sh scripts/coop/specpl_base2new_train.sh  imagenet 1
sh scripts/coop/specpl_base2new_test.sh   imagenet 1
```

### CoCoOp

```bash
sh scripts/cocoop/vanilla_base2new_train.sh oxford_pets 1
sh scripts/cocoop/vanilla_base2new_test.sh  oxford_pets 1

sh scripts/cocoop/specpl_base2new_train.sh  oxford_pets 1
sh scripts/cocoop/specpl_base2new_test.sh   oxford_pets 1
```

### MaPLe

```bash
sh scripts/maple/vanilla_base2new_train.sh caltech101 1
sh scripts/maple/vanilla_base2new_test.sh  caltech101 1

sh scripts/maple/specpl_base2new_train.sh  caltech101 1
sh scripts/maple/specpl_base2new_test.sh   caltech101 1
```

### MMRL

Run MMRL commands from the `MMRL/` directory:

```bash
cd MMRL

sh scripts/mmrl/vanilla_base2new_train.sh fgvc_aircraft 1
sh scripts/mmrl/vanilla_base2new_test.sh  fgvc_aircraft 1

sh scripts/mmrl/specpl_base2new_train.sh  fgvc_aircraft 1
sh scripts/mmrl/specpl_base2new_test.sh   fgvc_aircraft 1
```

For full reproduction details, see [`docs/SpecPL.md`](docs/SpecPL.md).

## Result Parsing

The maintained result parser is available in the MMRL sub-codebase:

```bash
cd MMRL
python parse_test_res.py output_specpl/base2new/test_new/<DATASET>/shots_16/<TRAINER>/<CFG>/
```

Top-level CoOp/CoCoOp/MaPLe runs follow the same output layout, so you can either parse copied outputs inside `MMRL/` or use the parser as a reference for your own result aggregation.

## Citation

If this project is useful for your research, please cite:

```bibtex
@inproceedings{zhou2026specpl,
    title     = {SpecPL: Disentangling Spectral Granularity for Prompt Learning},
    author    = {Zhou, Jingtao and Kang, Xirui and Huang, Feiyang and Po, Lai-Man},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year      = {2026}
}

@misc{zhou2026specpldisentanglingspectralgranularity,
    title         = {SpecPL: Disentangling Spectral Granularity for Prompt Learning},
    author        = {Jingtao Zhou and Xirui Kang and Feiyang Huang and Lai-Man Po},
    year          = {2026},
    eprint        = {2605.04504},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url           = {https://arxiv.org/abs/2605.04504}
}
```

## Acknowledgements

This implementation builds on [CoOp / CoCoOp](https://github.com/KaiyangZhou/CoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [MMRL](https://github.com/Kingcong/MMRL), [OpenAI CLIP](https://github.com/openai/CLIP), and [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

## Contact

Please open an issue for questions about installation, training, or reproduction.
