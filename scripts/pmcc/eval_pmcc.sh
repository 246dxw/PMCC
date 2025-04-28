#!/bin/bash

# custom config
DATA=/data/dxw/data # ********** your directory ***********

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7
ALPHA=$8
BETA=$9

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
SHARE=False

# bash scripts/pda/eval_pmcc.sh officehome b32_ep10_officehome PMCC ViT-B/16 2 a-c 0

DIR=/data/dxw/output/pda/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/tdeep${TDEEP}_vdeep${VDEEP}_${LOCATION}_a1.0b1.0/${DOMAINS}_ntok${NTOK}

python train.py \
  --gpu ${GPU} \
  --backbone ${BACKBONE} \
  --domains ${DOMAINS} \
  --root ${DATA} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/PMCC/${CFG}.yaml \
  --output-dir ${DIR} \
  --model-dir ${DIR} \
  --alpha ${ALPHA} \
  --beta ${BETA} \
  --eval-only \
  TRAINER.PMCC.TP ${TP}\
  TRAINER.PMCC.T_DEEP ${TDEEP} \
  TRAINER.PMCC.N_CTX ${NTOK} \
  TRAINER.PMCC.VP ${VP} \
  TRAINER.PMCC.V_DEEP ${VDEEP}\
  TRAINER.PMCC.NUM_TOKENS ${NTOK} \
  TRAINER.PMCC.LOCATION ${LOCATION} \
  TRAINER.PMCC.DEEP_LAYERS ${DEEPLAYER} \
  TRAINER.PMCC.DEEP_SHARED ${SHARE} 
  