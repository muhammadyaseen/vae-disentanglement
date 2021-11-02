import sys
import torch
import os

from common.utils import setup_logging, initialize_seeds, set_environment_variables
from common.arguments import get_args
import models

from vae_experiment import VAEExperiment

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):

    tb_logger = TensorBoardLogger(
        save_dir = _args.ckpt_dir,
        name = _args.expr_name
    )

    # load the model associated with args.alg
    model_cl = getattr(models, _args.alg)
    model = model_cl(_args)

    # load checkpoint
    # TODO: this will probably have to be replaced with tensorboard based loading
    if _args.ckpt_load:
        model.load_checkpoint(_args.ckpt_load, 
                                load_iternum=_args.ckpt_load_iternum, 
                                load_optim=_args.ckpt_load_optim)

    experiment_config = dict(        
        in_channels=_args.in_channels,
        image_size=_args.image_size,
        LR=_args.lr_G,
        weight_decay=0.0,
        
        dataset=_args.dset_name,
        datapath=_args.dset_dir,
        droplast=True,        
        batch_size=_args.batch_size,
        num_workers=_args.num_workers,
        pin_memory=_args.pin_memory,

        seed=_args.seed,
        evaluation_metrics=_args.evaluation_metrics
    )

    experiment_config['visual_args'] = dict(
        dataset = _args.dset_name,
        scalar_metrics = ['loss','recon', 'kld_loss'],
        disent_metrics = None,
    )
    experiment_config['visdom_args'] = dict() 

    experiment = VAEExperiment(model, experiment_config)

    trainer_config = dict(
        gpus=_args.gpus,
        max_epochs=20
    )

    pl_trainer = Trainer(default_root_dir=f"{tb_logger.save_dir}",
                     min_epochs=1,
                     logger=tb_logger,
                     flush_logs_every_n_steps=50,
                     limit_train_batches=1.,
                     limit_val_batches=1.,
                     num_sanity_val_steps=2,
                     callbacks = None,
                     accelerator='dp',
                     **trainer_config)

    pl_trainer.fit(experiment)


if __name__ == "__main__":

    _args = get_args(sys.argv[1:])

    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    main(_args)
