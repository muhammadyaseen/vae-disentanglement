TOKEN_NAME=dvae_red_reg
TOKEN_SECRET=glpat-8Hr9g1zt9RaxkUWoP9yZ

# for some reason the following doesn't work.
srun -p gpu --gres=gpu:A100:1 --container-image=projects.cispa.saarland:5005#c01muya:vae-disentanglement --container-mounts=./temp:/ctemp --pty bash

skopeo copy docker://projects.cispa.saarland:5005/c01muya/vae-disentanglement:latest docker://docker.io/myaseende/vae-disentanglement:latest

skopeo login docker.io
skopeo login projects.cispa.saarland:5005
apptainer remote login docker.io
apptainer remote login projects.cispa.saarland:5005

ssh muhammad1@juwels-booster.fz-juelich.de
export APPTAINER_CACHEDIR=$SCRATCH/at_cache
export APPTAINER_TMPDIR=$SCRATCH/at_tmpdir

srun -N1 --partition=develbooster --account=hai_vae_cs --gres=gpu:1 --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci
srun -N1 --partition=develbooster --account=hai_vae_cs --gres=gpu:1 $SCRATCH/container/vae-disent.oci --pty bash


salloc --partition=develbooster --gres=gpu:1 --account=hai_cs_vaes --time=00:30:00
srun -N1 --partition=develbooster --account=hai_vae_cs --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci
# https://apptainer.org/docs/user/main/cli/apptainer_shell.html

#get correlated data set up to load
#read up experimental settings and params from paper
#figure out how to use gpus in the container env
#figure out the slurm dir structure for placing code and putting results.

