#!/bin/bash
#SBATCH --account=hai_cs_vaes
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --output=hai_cs_vaes-gpu-out-and-err.%j
#SBATCH --error=hai_cs_vaes-gpu-out-and-err.%j
#SBATCH --job-name=hai_cs_vaes-Beta-VAE
#SBATCH --mail-user=muhammad.yaseen@cispa.de
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --time=02:00:00

NAME="BetaVAE_dsprites_corr02"
echo "name=$NAME"

# This path will work anywhere in JUWELS-Booster
PROJECT_ROOT=/vae-disentanglement
export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=dsprites_correlated
LOGS_DIR=$PROJECT_ROOT/train-logs

# --max_c=25 \
# --iterations_c=100000 \
# --controlled_capacity_increase=True

# The path after .sif refers to the path within containers
srun --account=hai_cs_vaes --gres=gpu:4 --partition=develbooster --nodes=1 \
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
    --z_dim=10 \
    --w_kld=1 \
    --num_workers=48 \
    --batch_size=64 \
    --max_epoch=26 \
    --in_channels=1 \
    --gpus 0 1 2 3  \
    --visdom_on=False \
    --lr_G=0.0001 \
    --correlation_strength=0.2
