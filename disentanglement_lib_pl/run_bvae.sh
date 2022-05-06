#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="dsprites_cond_b1"

echo "name=$NAME"

export DISENTANGLEMENT_LIB_DATA=../datasets/
export DATASET_NAME=dsprites_cond

python main.py \
--name=$NAME \
--ckpt_dir=../pl-dt-test \
--expr_name=$NAME \
--alg=BetaVAE \
--dset_dir=../datasets  \
--dset_name=$DATASET_NAME \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=4 \
--w_kld=1 \
--num_workers=8 \
--batch_size=128 \
--max_epoch=100 \
--in_channels=1 \
--gpus="2" \
--visdom_on=True 
#--loss_terms=BetaTCVAE \
#--w_tc=1 
