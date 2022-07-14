#!/bin/bash
#SBATCH --account=hai_cs_vaes
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --output=%j-job-out-and-err.txt
#SBATCH --error=%j-job-out-and-err.txt
#SBATCH --job-name=hai_cs_vaes-CS-VAE
#SBATCH --mail-user=muhammad.yaseen@cispa.de
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --time=06:59:00

NAME="w_dsprites_GNNCSVAE_2d_supreg_mix"
echo "name=$NAME"

# This path will work anywhere in JUWELS-Booster
PROJECT_ROOT=/vae-disentanglement
export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=dsprites_correlated
LOGS_DIR=$PROJECT_ROOT/train-logs

# The path after .sif refers to the path within containers
#srun --account=hai_cs_vaes --gres=gpu:4 --partition=develbooster --nodes=1 \
srun \
    apptainer exec --nv --bind $PROJECT/vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif python /vae-disentanglement/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=GNNBasedConceptStructuredVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --decoder=SimpleConv64CommAss \
    --w_kld=0.15 \
    --w_recon=0.8 \
    --num_workers=48 \
    --batch_size=64 \
    --max_epoch=50 \
    --in_channels=1 \
    --gpus 0  \
    --visdom_on=False \
    --lr_G=0.00001 \
    --adjacency_matrix=$PROJECT_ROOT/adjacency_matrices/$DATASET_NAME.pkl \
    --correlation_strength=0.2 \
    --z_dim 2 \
    --loss_terms aux_classification

