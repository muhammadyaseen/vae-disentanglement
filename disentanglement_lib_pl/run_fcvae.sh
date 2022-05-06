#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="laddervae-dsprites"

echo "name=$NAME"

export DISENTANGLEMENT_LIB_DATA=../datasets/
export DATASET_NAME=dsprites

python fcmain.py \
--name=$NAME \
--ckpt_dir=../pl-dt-test \
--expr_name=$NAME \
--alg=LadderVAE \
--dset_dir=../datasets  \
--dset_name=$DATASET_NAME \
#--encoder=SimpleFCNNEncoder \
#--decoder=SimpleFCNNDencoder \
--z_dim=2 \
#--w_kld=2 \
--num_workers=8 \
--batch_size=144 \
--max_epoch=1000 \
--in_channels=1 \
#--in_dim=4 \
--gpus="2" \
--visdom_on=True 
#--loss_terms=BetaTCVAE \
#--w_tc=1 
