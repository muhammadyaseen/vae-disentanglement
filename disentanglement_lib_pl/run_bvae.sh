#!/bin/sh

NAME="BetaVAE_dsprites_corr02"
echo "name=$NAME"

PROJECT_ROOT=./vae-disentanglement

export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=dsprites_correlated
LOGS_DIR=$PROJECT_ROOT/train-logs

# --max_c=25 \
# --iterations_c=100000 \
# --controlled_capacity_increase=True



python $PROJECT_ROOT/disentanglement_lib_pl/experiment_runner.py \
--name=$NAME \
--ckpt_dir=$LOGS_DIR \
--expr_name=$NAME \
--alg=BetaVAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=$DATASET_NAME \
--encoder=SimpleGaussianConv64CommAss \
--decoder=SimpleConv64CommAss \
--z_dim=10 \
--w_kld=1 \
--num_workers=1 \
--batch_size=64 \
--max_epoch=200 \
--in_channels=1 \
--gpus 0 \
--visdom_on=False \
--lr_G=0.0001 \
--correlation_strength=0.2
