#!/bin/bash
# CoCoOp + SpecPL — Base-to-Novel training.
# Usage: sh scripts/cocoop/specpl_base2new_train.sh <DATASET> <SEED>
# Loads `trainers/cocoop_specpl.py`; the registered trainer is `CoCoOpSpecPL`.
# Both vanilla CoCoOp and CoCoOp+SpecPL share the same config file under
# `configs/trainers/CoCoOp/`.

# Set DATA_ROOT and CLIP_ROOT before running, e.g.:
#   export DATA_ROOT=path/to/data
#   export CLIP_ROOT=path/to/clip
DATA=${DATA_ROOT:-path/to/data}
TRAINER=CoCoOpSpecPL
CFG_TRAINER=CoCoOp
TAG=specpl

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16
SUB=base

DIR=output_${TAG}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
