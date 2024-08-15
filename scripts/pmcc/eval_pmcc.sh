#!/bin/bash

# custom config
DATA=/Workplace_sdb/dxw/data # ********** your directory ***********

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

# bash scripts/pda/eval_pmcc.sh officehome b32_ep10_officehome PMCC ViT-B/16 2 a-c 0

DIR=/Workplace_sdb/dxw/output/pda/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/tdeep${TDEEP}_vdeep${VDEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

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
  