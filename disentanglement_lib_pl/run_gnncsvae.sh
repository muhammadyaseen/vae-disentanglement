#!/bin/bash

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="pendulum_plots_test"
echo "name=$NAME"

PROJECT_ROOT=../
echo $PROJECT_ROOT

export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=pendulum
LOGS_DIR=$PROJECT_ROOT/train-logs

python $PROJECT_ROOT/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=GNNBasedConceptStructuredVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --decoder=SimpleConv64CommAss \
    --w_kld=1.0 \
    --w_recon=10.0 \
    --num_workers=0 \
    --batch_size=64 \
    --max_epoch=100 \
    --in_channels=3 \
    --gpus 0  \
    --visdom_on=False \
    --lr_G=0.0001 \
    --adjacency_matrix=$PROJECT_ROOT/adjacency_matrices/$DATASET_NAME.pkl \
    --z_dim 2 \
    --l_dim 5 \
    --use_loss_weights=False
