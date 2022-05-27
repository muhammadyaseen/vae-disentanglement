TOKEN_NAME=dvae_red_reg
TOKEN_SECRET=glpat-8Hr9g1zt9RaxkUWoP9yZ

# for some reason the following doesn't work.
srun -p gpu --gres=gpu:A100:1 --container-image=projects.cispa.saarland:5005#c01muya:vae-disentanglement --container-mounts=./temp:/ctemp --pty bash

ssh muhammad1@juwels-booster.fz-juelich.de
jutil env activate -p hai_cs_vaes

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

srun -N1 --partition=develbooster --account=hai_cs_vaes --gres=gpu:1 --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci
srun -N1 --partition=develbooster --account=hai_cs_vaes --gres=gpu:1 $SCRATCH/container/vae-disent.oci --pty bash

# For interactive jobs, first we allocate nodes/resources and then we can run a container
salloc --partition=develbooster --gres=gpu:1 --account=hai_cs_vaes --time=00:30:00
srun -N1 --partition=develbooster --account=hai_cs_vaes --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci

srun -N1 --partition=develbooster --account=hai_cs_vaes --pty apptainer shell --network-args "portmap=8080:8080/tcp" --nv ../container-file/vae-disent-v1.1-tensorboard.sif 
srun -N1 --partition=develbooster --account=hai_cs_vaes --pty apptainer shell --nv ../container-file/vae-disent-v1.1-tensorboard.sif 
srun -N1 --partition=develbooster --account=hai_cs_vaes --pty apptainer shell --network none \
    --network-args "portmap=8080:8080" \
    --nv ../container-file/vae-disent-v1.1-tensorboard.sif 

srun -N1 --partition=develbooster --account=hai_cs_vaes --pty  \
    apptainer exec --bind ./vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif python /vae-disentanglement/disentanglement_lib_pl/visdomtest.py --visdom_port 8097 --visdom_host localhost 

srun -N1 --partition=develbooster --account=hai_cs_vaes --pty  \
    apptainer exec --bind ./vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif bash /vae-disentanglement/disentanglement_lib_pl/run_visdom_test.sh


srun -N1 --partition=develbooster --account=hai_cs_vaes --pty  \
    apptainer shell --nv --bind ./vae-disentanglement:/vae-disentanglement \
    ./container-file/vae-disent-v1.1-tensorboard.sif

# when running from within vae_disentanglement dir
srun --nodes=1 --gres=gpu:2 --partition=develbooster --account=hai_cs_vaes \
    apptainer exec --nv --bind ./:/vae-disentanglement \
    ../container-file/vae-disent-v1.1-tensorboard.sif bash /vae-disentanglement/disentanglement_lib_pl/run_bvae_jsc.sh


#sbatch    
# https://apptainer.org/docs/user/main/cli/apptainer_shell.html

#get correlated data set up to load
#read up experimental settings and params from paper
#figure out how to use gpus in the container env
#figure out the slurm dir structure for placing code and putting results.

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



