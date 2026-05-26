#!/bin/bash
# Vanilla MaPLe — Base-to-Novel evaluation (novel split).
# Usage: sh scripts/maple/vanilla_base2new_test.sh <DATASET> <SEED>
# Loads `trainers/maple.py`; the registered trainer is `MaPLe`.

# Set DATA_ROOT and CLIP_ROOT before running, e.g.:
#   export DATA_ROOT=path/to/data
#   export CLIP_ROOT=path/to/clip
DATA=${DATA_ROOT:-path/to/data}
TRAINER=MaPLe
TAG=vanilla

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5
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
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
