#!/bin/bash
# CoOp + SpecPL — Base-to-Novel training.
# Usage: sh scripts/coop/specpl_base2new_train.sh <DATASET> <SEED>
# Loads `trainers/coop_specpl.py`; the registered trainer is `CoOpSpecPL`.
# Both vanilla CoOp and CoOp+SpecPL share the same config file under
# `configs/trainers/CoOp/`.

# Set DATA_ROOT and CLIP_ROOT before running, e.g.:
#   export DATA_ROOT=path/to/data
#   export CLIP_ROOT=path/to/model
export HF_ENDPOINT=https://hf-mirror.com
DATA=${DATA_ROOT:-"path/to/data"}
TRAINER=CoOpSpecPL
CFG_TRAINER=CoOp
TAG=specpl

DATASET=$1
SEED=$2

CFG=vit_b16
SHOTS=16
SUB=base


if [ "$DATASET" = "imagenet" ]; then
    EPC=10
else
    EPC=100
fi

DIR=output_${TAG}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${EPC}/seed${SEED}
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
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    OPTIM.MAX_EPOCH ${EPC}
