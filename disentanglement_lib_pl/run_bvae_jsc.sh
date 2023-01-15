#!/bin/bash
#SBATCH --account=hai_slc_vaes
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --output=hai_slc_vaes-gpu-out-and-err.%j
#SBATCH --error=hai_slc_vaes-gpu-out-and-err.%j
#SBATCH --job-name=hai_slc_vaes-Beta-VAE
#SBATCH --mail-user=muhammad.yaseen@cispa.de
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --time=02:00:00

NAME="BetaVAE_pendulum_comp_runs"
echo "name=$NAME"

# This path will work anywhere in JUWELS-Booster
PROJECT_ROOT=/vae-disentanglement
export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=betavae_pendulum
LOGS_DIR=$PROJECT_ROOT/train-logs

# The path after .sif refers to the path within containers
srun \
    apptainer exec --nv --bind $PROJECT/vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif python /vae-disentanglement/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=BetaVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --encoder=SimpleGaussianConv64CommAss \
    --decoder=SimpleConv64CommAss \
    --visdom_on=False \
    --z_dim 4 \
    --l_dim 4 \
    --w_kld=25.0 \
    --w_recon=1.0 \
    --num_workers=48 \
    --batch_size=64 \
    --max_epoch=200 \
    --in_channels=3 \
    --gpus 0 \
    --visdom_on=False \
    --lr_G=0.0001 \

