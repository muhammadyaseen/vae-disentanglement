#!/bin/bash

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="BetaVAE_Toy"
echo "name=$NAME"

PROJECT_ROOT=./vae-disentanglement

export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=toydata
LOGS_DIR=$PROJECT_ROOT/train-logs

python $PROJECT_ROOT/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=BetaVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --decoder=SmallFCDecoder \
    --encoder=SmallFCEncoder \
    --w_kld=0.1 \
    --w_recon=5 \
    --z_dim 6 \
    --kl_warmup_epochs=0 \
    --num_workers=1 \
    --batch_size=64 \
    --max_epoch=100 \
    --in_channels=1 \
    --gpus 0  \
    --visdom_on=False \
    --lr_G=0.001 \
    --adjacency_matrix=$PROJECT_ROOT/adjacency_matrices/$DATASET_NAME.pkl \
    --interm_unit_dim=3 \
    --correlation_strength=0.2 \
    --continue_training=True \
    --ckpt_path=$LOGS_DIR/BetaVAE_Toy/version_1/checkpoints/epoch=49-step=699.ckpt
    

