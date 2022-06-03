#!/bin/bash

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="CS_VAE_tbtest_weights"
echo "name=$NAME"

PROJECT_ROOT=./vae-disentanglement

export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=celeba
LOGS_DIR=$PROJECT_ROOT/train-logs

#    --loss_terms AuxClassification BetaTCVAE

python $PROJECT_ROOT/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=ConceptStructuredVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=10 \
    --w_kld=1 \
    --num_workers=1 \
    --batch_size=64 \
    --max_epoch=3 \
    --in_channels=3 \
    --gpus 0  \
    --visdom_on=False \
    --lr_G=0.0001 \
    --adjacency_matrix=$PROJECT_ROOT/adjacency_matrices/$DATASET_NAME.pkl \
    --interm_unit_dim=2 \
    --correlation_strength=0.2 


