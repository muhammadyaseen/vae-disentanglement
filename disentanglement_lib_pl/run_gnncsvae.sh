#!/bin/bash

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="GNNCSVAE_structure_test"
echo "name=$NAME"

PROJECT_ROOT=./vae-disentanglement

export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=dsprites_correlated
LOGS_DIR=$PROJECT_ROOT/train-logs

python $PROJECT_ROOT/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=GNNBasedConceptStructuredVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --decoder=SimpleConv64CommAss \
    --w_kld=1 \
    --num_workers=0 \
    --batch_size=64 \
    --max_epoch=2 \
    --in_channels=1 \
    --gpus 0  \
    --visdom_on=False \
    --lr_G=0.0001 \
    --adjacency_matrix=$PROJECT_ROOT/adjacency_matrices/$DATASET_NAME.pkl \
    --interm_unit_dim=3 \
    --correlation_strength=0.2 \
    --loss_terms AuxClassification \
    --pin_memory=False \
    --z_dim 5 
