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
# e.g apptainer pull file-out.sif docker://alpine:latest

srun -N1 --partition=develbooster --account=hai_vae_cs --gres=gpu:1 --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci
srun -N1 --partition=develbooster --account=hai_vae_cs --gres=gpu:1 $SCRATCH/container/vae-disent.oci --pty bash

# For interactive jobs, first we allocate nodes/resources and then we can run a container
salloc --partition=develbooster --gres=gpu:1 --account=hai_cs_vaes --time=00:30:00
srun -N1 --partition=develbooster --account=hai_vae_cs --pty apptainer shell --nv $SCRATCH/container/vae-disent.oci

srun -N1 --partition=develbooster --account=hai_vae_cs --pty apptainer shell --network-args "portmap=8080:8080/tcp" --nv ../container-file/vae-disentanglement_latest.sif 
srun -N1 --partition=develbooster --account=hai_vae_cs --pty apptainer shell --nv ../container-file/vae-disentanglement_latest.sif 
srun -N1 --partition=develbooster --account=hai_vae_cs --pty apptainer shell --network none \
    --network-args "portmap=8080:8080" \
    --nv ../container-file/vae-disentanglement_latest.sif 

# https://apptainer.org/docs/user/main/cli/apptainer_shell.html

#get correlated data set up to load
#read up experimental settings and params from paper
#figure out how to use gpus in the container env
#figure out the slurm dir structure for placing code and putting results.

