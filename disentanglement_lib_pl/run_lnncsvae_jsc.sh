#!/bin/bash
#SBATCH --account=hai_slc_vaes
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --output=%j-job-out-and-err.txt
#SBATCH --error=%j-job-out-and-err.txt
#SBATCH --job-name=hai_slc_vaes-SLC-VAE
#SBATCH --mail-user=muhammad.yaseen@cispa.de
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --time=02:00:00

NAME="waterflow_latentnn"
echo "name=$NAME"

# This path will work anywhere in JUWELS-Booster
PROJECT_ROOT=/vae-disentanglement
export DISENTANGLEMENT_LIB_DATA=$PROJECT_ROOT/datasets/
DATASET_NAME=waterflow
LOGS_DIR=$PROJECT_ROOT/train-logs

# The path after .sif refers to the path within containers
srun \
    apptainer exec --nv --bind $PROJECT/vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif python /vae-disentanglement/disentanglement_lib_pl/experiment_runner.py \
    --name=$NAME \
    --ckpt_dir=$LOGS_DIR \
    --expr_name=$NAME \
    --alg=LatentNN_CSVAE \
    --dset_dir=$DISENTANGLEMENT_LIB_DATA  \
    --dset_name=$DATASET_NAME \
    --decoder=SimpleConv64CommAss \
    --w_kld=1.0 \
    --w_recon=1.0 \
    --w_sup_reg=1.0 \
    --w_cov_loss=0.0 \
    --num_workers=48 \
    --batch_size=64 \
    --max_epoch=200 \
    --in_channels=3 \
    --gpus 0 \
    --visdom_on=False \
    --lr_G=0.0001 \
    --adjacency_matrix=$PROJECT_ROOT/adjacency_matrices/$DATASET_NAME.pkl \
    --z_dim 1 \
    --l_dim 4 \
    --use_loss_weights=False \
    --loss_terms aux_classification 


