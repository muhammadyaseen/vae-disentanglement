#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME="AnnealedVAE_dsprites_corr"
echo "name=$NAME"

PROJECT=../
export DISENTANGLEMENT_LIB_DATA=$PROJECT/datasets/
DATASET_NAME=dsprites_correlated
VISDOM_PORT=8097
VISDOM_ENV=$PROJECT/train-logs/$NAME/visdom

# Check for visdom port
echo "Checking Visdom port..."
ss -tulpn | grep $VISDOM_PORT
VISDOM_PROCESS=$(ss -tulpn | grep $VISDOM_PORT | sed -r -n -e 's/.*pid=([0-9]+).*$/\1/p;q')
if [ -z "$VISDOM_PROCESS" ]
then
    echo "Visdom port is free"
else
    echo "Process $VISDOM_PROCESS is using $VISDOM_PORT. Will attempt to kill it"
    kill $VISDOM_PROCESS
    echo "After killing process"
    ss -tulpn | grep $VISDOM_PORT
fi

# start visdom server
echo "Creating Visdom env dir at: $VISDOM_ENV"
mkdir -p $VISDOM_ENV
visdom -env_path $VISDOM_ENV -logging_level WARN &


# --max_c=25 \
# --iterations_c=100000 \
# --controlled_capacity_increase=True

python main_bvae.py \
--name=$NAME \
--ckpt_dir=$PROJECT/train-logs \
--expr_name=$NAME \
--alg=BetaVAE \
--dset_dir=../datasets  \
--dset_name=$DATASET_NAME \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=10 \
--w_kld=2 \
--num_workers=8 \
--batch_size=64 \
--max_epoch=100 \
--in_channels=1 \
--gpus  \
--visdom_on=True \
--lr_G=0.0001 \
--save_every_epoch=True \
--visdom_port=$VISDOM_PORT