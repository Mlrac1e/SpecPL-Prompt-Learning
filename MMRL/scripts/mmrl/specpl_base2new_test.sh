#!/bin/bash
# MMRL + SpecPL — Base-to-Novel evaluation (novel split).
# Usage: sh scripts/mmrl/specpl_base2new_test.sh <DATASET> <SEED>
# Loads `trainers/mmrl_specpl.py`; the registered trainer is `MMRLSpecPL`.
# Both vanilla MMRL and MMRL+SpecPL share the same config file under
# `configs/trainers/MMRL/`.

# Set DATA_ROOT and CLIP_ROOT before running, e.g.:
#   export DATA_ROOT=path/to/data
#   export CLIP_ROOT=path/to/model
export HF_ENDPOINT=https://hf-mirror.com
DATA=${DATA_ROOT:-"path/to/data"}
TRAINER=MMRLSpecPL
CFG_TRAINER=MMRL
TAG=specpl

DATASET=$1
SEED=$2


if [ "$DATASET" = "imagenet" ]; then
    CFG=vit_b16_imagenet
else
    CFG=vit_b16
fi

SHOTS=16
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output_${TAG}/base2new/train_base/${COMMON_DIR}
DIR=output_${TAG}/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
else
    echo "Run this job and save the output to ${DIR}"
fi

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${CFG_TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TASK B2N

