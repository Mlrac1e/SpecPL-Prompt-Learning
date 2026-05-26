#!/bin/bash

#cd ../..

# custom config
DATA=${DATA_ROOT:-path/to/data}
TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=vit_b16
SHOTS=16
EPC=10

DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output_XD/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
    --load-epoch ${EPC} \
    --eval-only
fi