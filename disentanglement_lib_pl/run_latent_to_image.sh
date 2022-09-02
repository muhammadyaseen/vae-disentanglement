#!/bin/sh

NAME="LatentToImage_test"
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
--alg=LatentToImage \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=$DATASET_NAME \
--encoder=SimpleGaussianConv64CommAss \
--decoder=SimpleConv64CommAss \
--z_dim=4 \
--l_dim=4 \
--num_workers=2 \
--batch_size=64 \
--max_epoch=100 \
--in_channels=3 \
--gpus 0 \
--visdom_on=False \
--lr_G=0.0001
 
