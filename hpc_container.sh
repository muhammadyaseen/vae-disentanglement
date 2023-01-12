TOKEN_NAME=dvae_red_reg
TOKEN_SECRET=glpat-8Hr9g1zt9RaxkUWoP9yZ

# for some reason the following doesn't work.
srun -p gpu --gres=gpu:A100:1 --container-image=projects.cispa.saarland:5005#c01muya:vae-disentanglement --container-mounts=./temp:/ctemp --pty bash

ssh muhammad1@juwels-booster.fz-juelich.de
jutil env activate -p hai_slc_vaes

# Container related stuff

# Need to do this because the default directory where we login doesn't have enough space
export APPTAINER_CACHEDIR=$SCRATCH/at_cache
export APPTAINER_TMPDIR=$SCRATCH/at_tmpdir

# connect skopio to our registeries
skopeo login docker.io
skopeo login projects.cispa.saarland:5005
# connect apptainer to our registeries
apptainer remote login docker.io
apptainer remote login projects.cispa.saarland:5005

# upload image to Docker Hub
skopeo copy docker://projects.cispa.saarland:5005/c01muya/vae-disentanglement:latest docker://docker.io/myaseende/vae-disentanglement:latest
# pull image from Docker Hub
apptainer pull <container_name>.sif docker://docker.io/myaseende/vae-disentanglement:latest
apptainer pull vae-disent-v1.1-tensorboard.sif docker://docker.io/myaseende/vae-disentanglement:v1.1-tensorboard
# e.g apptainer pull file-out.sif docker://alpine:latest

srun -N1 --partition=develbooster --account=hai_slc_vaes --gres=gpu:1 --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci
srun -N1 --partition=develbooster --account=hai_slc_vaes --gres=gpu:1 $SCRATCH/container/vae-disent.oci --pty bash

# For interactive jobs, first we allocate nodes/resources and then we can run a container
salloc --partition=develbooster --gres=gpu:1 --account=hai_slc_vaes --time=00:30:00
srun -N1 --partition=develbooster --account=hai_slc_vaes --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci

# https://apptainer.org/docs/user/main/cli/apptainer_shell.html
# To get a shell inside the container
srun -N1 --partition=develbooster --account=hai_slc_vaes --pty \
    apptainer shell --nv ../container-file/vae-disent-v1.1-tensorboard.sif 

srun -N1 --partition=develbooster --account=hai_slc_vaes --pty  \
    apptainer shell --nv --bind ./vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif

# when running from within vae_disentanglement dir
srun --nodes=1 --gres=gpu:2 --partition=develbooster --account=hai_slc_vaes \
    apptainer exec --nv --bind ./:/vae-disentanglement \
    ../container-file/vae-disent-v1.1-tensorboard.sif bash /vae-disentanglement/disentanglement_lib_pl/run_bvae_jsc.sh

# Running visdom
module load Python
~/.local/bin/visdom
~/.local/bin/tensorboard --logdir vae-disentanglement/train-logs/AnnealedVAE_dsprites_corr/AnnealedVAE_dsprites_corr --port 6006

# building 
docker build -t myaseende/vae-disentanglement:v1.1-tensorboard .
# Running on my laptop
docker run --rm -it -p 6006:6006 -v ./vae-disentanglement:/vae-disentanglement myaseende/vae-disentanglement:v1.1-tensorboard /bin/bash
# Pushing on my laptop
docker image push myaseende/vae-disentanglement:v1.1-tensorboard

#From $PROJECT directory
sbatch vae-disentanglement/disentanglement_lib_pl/run_csvae_jsc.sh

# Apptainer commands can also be directly ran on Login nodes
# This is useful when you just want to check something quick in the container 
# and don't want to allocate resources / budget for it
apptainer exec --nv --bind ./vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif cat /etc/os-release

apptainer exec --nv --bind $PROJECT/vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif 'pip list | grep light'


# Moving files via scp
# First move to laptop
scp muhammad1@juwels-booster.fz-juelich.de:/p/project/hai_slc_vaes/vae-disentanglement/train-logs/w_lr4_dsprites_GNNCSVAE_2d_supreg_mix//version_0/checkpoints/model.ckpt ./<etc>
scp muhammad1@juwels-booster.fz-juelich.de:/p/project/hai_slc_vaes/vae-disentanglement/train-logs/dim_debug_GNNCSVAE/version_1/checkpoints/epoch=42-step=495360.ckpt ./
# Then to Cispa
scp <etc> c01muya@gpu03:/home/c01muya/vae-disentanglement/train-logs/jscmodel_bvae_w1corr02_dsprites


# Uploading to Tensorboard.dev
tensorboard dev upload --logdir ./vae-disentanglement/train-logs/GNN_CS_VAE_dsprites_correlated_structure_test_cont \
    --name "GNN_CSVAE_dsprites_correlated" \
    --description "Results after training GNN based CSVAE on dsprites_correlated over ~60 epochs w/o any Annealing or Weighting"


scp muhammad1@juwels-booster.fz-juelich.de:/p/project/hai_slc_vaes/vae-disentanglement/train-logs/pendulum_sw_latentnn/latentnn_kld5_psw/checkpoints/epoch=167-step=38304.ckpt ./latentnn_167eps_kld5_pendulum_sw.ckpt

 #10.17.0.0/16, 134.96.225.0/24, 134.96.235.0/24, 134.96.238.32/27, 134.96.238.64/26, 134.96.238.128/25, 195.37.156.0/22

scp muhammad1@juwels-booster.fz-juelich.de:/p/project/hai_slc_vaes/vae-disentanglement/train-logs/dsprites_latentnn_w_kld4_z10/version_3_continue_from_272eps/checkpoints/*.ckpt ./
scp muhammad1@juwels-booster.fz-juelich.de:/p/project/hai_slc_vaes/vae-disentanglement/train-logs/celeba_latentnn_w_kld1_z30/version_8_from_204eps/checkpoints/*.ckpt ./


# To rerun stuff add the following args
#  --continue_training=True \
#  --ckpt_path=$LOGS_DIR/$NAME/version_?/checkpoints/<file>.ckpt
# The training will start from prev saved epoch state and continue till `max_epochs`
# A new checkpoint will be saved, so make sure to backup old one if you still need it

