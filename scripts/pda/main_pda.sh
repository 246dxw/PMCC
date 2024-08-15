#!/bin/bash

# custom config
DATA= # ********** your directory ***********

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7

LOCATION=middle
DEEPLAYER=None
TP=True

# text prompt
# TDEEP=False
# VP=False
# VDEEP=False
# SHARE=False

# multi-modal prompt
TDEEP=True
VP=True
VDEEP=True
SHARE=True

DIR=output/pda/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/tdeep${TDEEP}_vdeep${VDEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}, so skip this job"
else
  echo "Run this job and save the output to ${DIR}"
  
  python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/PDA/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.PDA.TP ${TP}\
    TRAINER.PDA.T_DEEP ${TDEEP} \
    TRAINER.PDA.N_CTX ${NTOK} \
    TRAINER.PDA.VP ${VP} \
    TRAINER.PDA.V_DEEP ${VDEEP}\
    TRAINER.PDA.NUM_TOKENS ${NTOK} \
    TRAINER.PDA.LOCATION ${LOCATION} \
    TRAINER.PDA.DEEP_LAYERS ${DEEPLAYER} \
    TRAINER.PDA.DEEP_SHARED ${SHARE} 
    
fi