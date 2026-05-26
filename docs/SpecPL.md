# SpecPL Reproduction Guide

This guide covers the maintained Base-to-Novel scripts for `CoOp`, `CoCoOp`, `MaPLe`, `MMRL`, and their `+ SpecPL` variants on the standard prompt-learning datasets.

CoOp, CoCoOp, and MaPLe are run from the top-level codebase. MMRL is run from the `MMRL/` sub-codebase because it has its own configs, trainer registry, and entry point.

## 1. Preliminary

### 1.1 Environment

Follow the Requirements section in the project [`README.md`](../README.md). In short:

```bash
conda create -n specpl python=3.9 -y
conda activate specpl
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch && python setup.py develop && cd ..
```

### 1.2 Datasets And Cache Paths

Follow the dataset preparation instructions of [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). Place all datasets under one root, then export it before running scripts:

```bash
export DATA_ROOT=path/to/data
```

The scripts also read these optional environment variables:

```bash
export CLIP_ROOT=path/to/clip        # OpenAI CLIP weight cache; defaults to ~/.cache/clip
export HF_ENDPOINT=https://...       # optional HuggingFace mirror for restricted networks
```

The examples intentionally use `path/to/...` placeholders. Replace them with local paths on your machine.

### 1.3 Pre-trained CLIP And VAE Teacher

* CLIP ViT-B/16 is downloaded automatically on first use. Set `CLIP_ROOT` if you want to control the cache directory.
* SpecPL trainers instantiate `VAEAttributeTeacher` from `clip/vae.py`. By default it loads `REPA-E/e2e-qwenimage-vae` from HuggingFace. For offline use, set `VAE_PRETRAINED_ID` in the corresponding config to a local path.

## 2. File Layout & Convention

| Family | Vanilla file            | Vanilla `TRAINER` | `+ SpecPL` file                | `+ SpecPL` `TRAINER` |
| :----- | :---------------------- | :---------------: | :----------------------------- | :------------------: |
| CoOp   | `trainers/coop.py`      | `CoOp`            | `trainers/coop_specpl.py`      | `CoOpSpecPL`         |
| CoCoOp | `trainers/cocoop.py`    | `CoCoOp`          | `trainers/cocoop_specpl.py`    | `CoCoOpSpecPL`       |
| MaPLe  | `trainers/maple.py`     | `MaPLe`           | `trainers/maple_specpl.py`     | `MaPLeSpecPL`        |
| MMRL   | `MMRL/trainers/mmrl.py` | `MMRL`            | `MMRL/trainers/mmrl_specpl.py` | `MMRLSpecPL`         |

The vanilla and SpecPL files register **different** trainer names, so both are imported at startup and the active method is selected at runtime via `--trainer`:

```python
# train.py  (top-level, CoOp / CoCoOp / MaPLe)
import trainers.coop          # registers TRAINER `CoOp`
import trainers.cocoop        # registers TRAINER `CoCoOp`
import trainers.maple         # registers TRAINER `MaPLe`

import trainers.coop_specpl   # registers TRAINER `CoOpSpecPL`
import trainers.cocoop_specpl # registers TRAINER `CoCoOpSpecPL`
import trainers.maple_specpl  # registers TRAINER `MaPLeSpecPL`
```

```python
# MMRL/train.py
import trainers.mmrl          # registers TRAINER `MMRL`
import trainers.mmrl_specpl   # registers TRAINER `MMRLSpecPL`
```

The vanilla and SpecPL variants in each family **share the same config file** (e.g. `configs/trainers/CoOp/vit_b16.yaml`); only `--trainer` and the output-directory tag (`output_vanilla` vs `output_specpl`) differ between the `vanilla_*.sh` and `specpl_*.sh` scripts.

---

## 3. Base-to-Novel Experiments

All Base-to-Novel scripts follow the same calling convention:

```bash
sh scripts/<method>/<vanilla|specpl>_base2new_train.sh <DATASET> <SEED>
sh scripts/<method>/<vanilla|specpl>_base2new_test.sh  <DATASET> <SEED>
```

The `<vanilla|specpl>` prefix selects both the registered `TRAINER` (`CoOp` vs `CoOpSpecPL`, etc.) and the output-directory tag (`output_vanilla/...` vs `output_specpl/...`). No edits to `train.py` are required.

### 3.1 CoOp / CoOp + SpecPL

```bash
# Vanilla
for SEED in 1 2 3; do
  sh scripts/coop/vanilla_base2new_train.sh imagenet $SEED
  sh scripts/coop/vanilla_base2new_test.sh  imagenet $SEED
done

# + SpecPL
for SEED in 1 2 3; do
  sh scripts/coop/specpl_base2new_train.sh imagenet $SEED
  sh scripts/coop/specpl_base2new_test.sh  imagenet $SEED
done
```

* Default config: `configs/trainers/CoOp/vit_b16.yaml` (16 shots, 10 epochs for `imagenet`; non-ImageNet epoch counts follow the script-specific `EPC` setting).
* SpecPL-specific defaults are added in `train.py` under `TRAINER.COOP`.

### 3.2 CoCoOp / CoCoOp + SpecPL

```bash
# Vanilla
sh scripts/cocoop/vanilla_base2new_train.sh oxford_pets 1
sh scripts/cocoop/vanilla_base2new_test.sh  oxford_pets 1

# + SpecPL
sh scripts/cocoop/specpl_base2new_train.sh  oxford_pets 1
sh scripts/cocoop/specpl_base2new_test.sh   oxford_pets 1
```

* Default config: `configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml`.

### 3.3 MaPLe / MaPLe + SpecPL

```bash
# Vanilla
sh scripts/maple/vanilla_base2new_train.sh caltech101 1
sh scripts/maple/vanilla_base2new_test.sh  caltech101 1

# + SpecPL
sh scripts/maple/specpl_base2new_train.sh  caltech101 1
sh scripts/maple/specpl_base2new_test.sh   caltech101 1
```

* Default config: `configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml` (deep prompting on the first 9 layers, 2 context tokens).

### 3.4 MMRL / MMRL + SpecPL  (`MMRL/`)

```bash
cd MMRL

# Vanilla
sh scripts/mmrl/vanilla_base2new_train.sh fgvc_aircraft 1
sh scripts/mmrl/vanilla_base2new_test.sh  fgvc_aircraft 1

# + SpecPL
sh scripts/mmrl/specpl_base2new_train.sh  fgvc_aircraft 1
sh scripts/mmrl/specpl_base2new_test.sh   fgvc_aircraft 1
```

* Default config: `configs/trainers/MMRL/vit_b16.yaml` (or `vit_b16_imagenet.yaml` when `DATASET=imagenet`).
* MMRL uses the `TASK B2N` flag during training to enable the Base-to-Novel split (kept as-is in the scripts).

## 4. Cross-Dataset & Few-Shot (optional)

The cross-dataset and few-shot scripts inherited from the upstream baselines are kept for reference. The maintained public examples in this repository are the Base-to-Novel scripts listed above. To adapt a reference script to SpecPL, use the SpecPL trainer name, for example `--trainer CoOpSpecPL`, and keep the same dataset/config layout.

## 5. Parsing Results

```bash
# The maintained parser is in the MMRL sub-codebase.
cd MMRL
python parse_test_res.py output_specpl/base2new/test_new/<DATASET>/shots_16/<TRAINER>/<CFG>/
```

This is the standard `Dassl` parser; it reports per-seed Top-1 accuracy plus mean/std so you can directly compute the Harmonic Mean across base & novel splits.

## 6. FAQ

**Q1. Are the vanilla and SpecPL trainers loaded simultaneously?**
Yes. The vanilla file (e.g. `trainers/coop.py`) registers `CoOp` while the SpecPL file (`trainers/coop_specpl.py`) registers `CoOpSpecPL`. Both modules are imported by `train.py`, and the active trainer is selected at runtime via `--trainer`.

**Q2. Where does the spectral disentanglement happen?**
Inside the `+ SpecPL` trainers, a frozen VAE (`clip/vae.py:VAEAttributeTeacher`) is used as the spatial-spectral proxy. Its outputs are split into low- and high-frequency bands and used to supervise the prompt-learning components. See Section 3 of the paper.

**Q3. Can I run on a different backbone (e.g. ViT-L/14)?**
Yes. Provide an appropriate YAML under `configs/trainers/<TRAINER>/` and update `CFG` in the corresponding script. The SpecPL plug-in is backbone-agnostic.
