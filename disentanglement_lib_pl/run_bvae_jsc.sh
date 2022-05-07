#! /bin/sh

export PROJECT_FOLDER=$PROJECT/vae-disentanglement/
export DISENTANGLEMENT_LIB_DATA=$PROJECT_FOLDER/datasets/
export DATASET_NAME=dsprites_correlated

# start visdom server
visdom > $PROJECT_FOLDER/visdom.log &

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="AnnealedVAE_dsprites_corr"

echo "name=$NAME"

python main_bvae.py \
--name=$NAME \
--ckpt_dir=$PROJECT_FOLDER/train-logs \
--expr_name=$NAME \
--alg=BetaVAE \
--dset_dir=../datasets  \
--dset_name=$DATASET_NAME \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=10 \
--w_kld=1000 \
--num_workers=8 \
--batch_size=128 \
--max_epoch=100 \
--in_channels=1 \
--gpus="2" \
--visdom_on=True \
--lr_G=5e-4 \
--save_every_epoch=True \
--max_c=25 \
--iterations_c=100000 \
--controlled_capacity_increase=True