#!/bin/bash
# custom config
DATA=${DATA_ROOT:-path/to/data}
TRAINER=CoOp
CTP=end
CFG=vit_b16
SHOTS=16
CSC=False


DATASET=imagenet
NCTX=16
EPO=10


DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${EPC}/seed${SEED}
python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all \
        OPTIM.MAX_EPOCH ${EPO}