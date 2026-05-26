#!/bin/bash
# Vanilla MMRL — Base-to-Novel evaluation (novel split).
# Usage: sh scripts/mmrl/vanilla_base2new_test.sh <DATASET> <SEED>
# Loads `trainers/mmrl.py`; the registered trainer is `MMRL`.

# Set DATA_ROOT and CLIP_ROOT before running, e.g.:
#   export DATA_ROOT=path/to/data
#   export CLIP_ROOT=path/to/clip
DATA=${DATA_ROOT:-path/to/data}
TRAINER=MMRL
TAG=vanilla

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
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TASK B2N
