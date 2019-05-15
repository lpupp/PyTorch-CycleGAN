#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --gres gpu:Tesla-K80:1

BASE_PATH=/home/cluster/lgaega
GIT_PATH=${BASE_PATH}/PyTorch-CycleGAN
DATA_BASE_PATH=${BASE_PATH}/data/CycleGAN

source ~/tensorflow/bin/activate

RUN_DOMAIN=fashion
#RUN_DOMAIN=jewels

RUN_NAME=shoes2dresses
#RUN_NAME=bracelets2rings
#RUN_NAME=earrings2bracelets
#RUN_NAME=rings2earrings

OUTPUT_DIR=${DATA_BASE_PATH}/output/${RUN_DOMAIN}/${RUN_NAME}
DATAROOT=${DATA_BASE_PATH}/${RUN_DOMAIN}/${RUN_NAME}

mkdir ${DATA_BASE_PATH}/output/
mkdir ${DATA_BASE_PATH}/output/${RUN_DOMAIN}
mkdir ${OUTPUT_DIR}

# Hyper-params
IMG_SIZE=256
BS=1
LR=0.0002

srun python ${GIT_PATH}/train.py --dataroot ${DATA_BASE_PATH}/${RUN_DOMAIN}/${RUN_NAME} --cuda --n_cpu 8 --n_epochs 300 --batch_size ${BS} --lr ${LR} --size ${IMG_SIZE} --output_dir ${OUTPUT_DIR} >> ${OUTPUT_DIR}/${RUN_NAME}_${IMG_SIZE}_${BS}_${LR}.txt
