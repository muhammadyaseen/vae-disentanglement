#! /bin/sh

visdom > /workspace/vae-disentanglement/visdom.log & 

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="laddervae-celeba-l0-test"

echo "name=$NAME"

export DISENTANGLEMENT_LIB_DATA=../datasets/
export DATASET_NAME=celeba

python main_laddervae.py \
--name=$NAME \
--ckpt_dir=../pl-dt-test \
--expr_name=$NAME \
--alg=LadderVAE \
--dset_dir=../datasets  \
--dset_name=$DATASET_NAME \
--z_dim 10 10 \
--num_workers=8 \
--batch_size=144 \
--max_epoch=1000 \
--in_channels=3 \
--gpus="2" \
--visdom_on=True \
--l_zero_reg=True

