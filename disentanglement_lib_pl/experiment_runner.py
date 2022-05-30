import sys
import torch
import os
import pdb

from common.utils import setup_logging, initialize_seeds, set_environment_variables
from common.arguments import get_args
import models

from bvae_experiment import BVAEExperiment
from laddervae_experiment import LadderVAEExperiment
from csvae_experiment import ConceptStructuredVAEExperiment

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

EXPERIMENT_CLASS = {
    'LadderVAE': LadderVAEExperiment,
    'BetaVAE': BVAEExperiment,
    'ConceptStructuredVAE': ConceptStructuredVAEExperiment
}

def get_dataset_specific_params(cmdline_args):

    if cmdline_args.dset_name == 'dsprites_correlated': 
        return dict(correlation_strength=cmdline_args.correlation_strength)
    else:
        return {}

def get_scalar_metrics_for_alg(cmdline_args):
    
    base_metrics = ['loss','recon', 'kld_loss']
    if cmdline_args.alg == 'ConceptStructuredVAE':
        return base_metrics
    
    elif cmdline_args.alg == 'LadderVAE':
        return base_metrics + ['kld_z1', 'kld_z2'] + ['l_zero_reg'] if cmdline_args.l_zero_reg else []
    
    elif cmdline_args.alg == 'BetaVAE':
        return base_metrics + ['C'] if cmdline_args.controlled_capacity_increase else []
    
    else:
        raise ValueError(f"Unsupported algorithm: {cmdline_args.alg}")

def get_experiment_config_for_alg(cmdline_args):
    
    base_experiment_config = dict(
        name = cmdline_args.expr_name,        
        in_channels = cmdline_args.in_channels,
        image_size = cmdline_args.image_size,
        LR = cmdline_args.lr_G,
        weight_decay = 0.0,
        dataset = cmdline_args.dset_name,
        datapath = cmdline_args.dset_dir,
        droplast = True,        
        batch_size = cmdline_args.batch_size,
        num_workers = cmdline_args.num_workers,
        pin_memory = cmdline_args.pin_memory,
        seed = cmdline_args.seed,
        evaluation_metrics = cmdline_args.evaluation_metrics,
        visdom_on = cmdline_args.visdom_on,
        save_dir = os.path.join(cmdline_args.ckpt_dir, cmdline_args.expr_name),
        max_epochs = cmdline_args.max_epoch
    )

    if cmdline_args.alg == 'ConceptStructuredVAE':
        base_experiment_config.update({

        })

    elif cmdline_args.alg == 'LadderVAE':
        base_experiment_config.update({
            'l_zero_reg': cmdline_args.l_zero_reg
        })

    elif cmdline_args.alg == 'BetaVAE':
        base_experiment_config.update({
            'max_c': cmdline_args.max_c
        })

    else:
        raise ValueError(f"Unsupported algorithm: {cmdline_args.alg}")

    return base_experiment_config

def get_trainer_params(cmdline_args, logger):

    base_trainer_params = dict(
            default_root_dir=logger.save_dir,
            min_epochs=1,
            logger=logger,
            limit_train_batches=1.0,
            limit_val_batches=0.05,
            num_sanity_val_steps=2,
            callbacks = None,
            gpus=cmdline_args.gpus,
            max_epochs=cmdline_args.max_epoch
    )
        
    # version specific params (local vs. cluster)
    import pytorch_lightning as pl
    if pl.__version__ == '1.6.3': # Container/Cluster
        base_trainer_params.update({
            'strategy': 'ddp',
            'accelerator': 'gpu',
            'devices': 4,
            'enable_progress_bar': False,
        })
    
    if pl.__version__ == '1.4.4': # Local
        base_trainer_params.update({
            'accelerator': 'dp',
            'progress_bar_refresh_rate': 0
        })

    return base_trainer_params

def main(_args):

    tb_logger = TensorBoardLogger(
        save_dir = _args.ckpt_dir,
        name = _args.expr_name
    )

    # load the model associated with args.alg
    model_cl = getattr(models, _args.alg)
    model = model_cl(_args)
   
    # load checkpoint
    if _args.ckpt_load:
        model.load_checkpoint(_args.ckpt_load, 
                                load_iternum=_args.ckpt_load_iternum, 
                                load_optim=_args.ckpt_load_optim)


    # Get expr metadata and hyperparams etc
    experiment_config = get_experiment_config_for_alg(_args)

    experiment_config['visual_args'] = dict(
        dataset = _args.dset_name,
        scalar_metrics = get_scalar_metrics_for_alg(_args),        
        disent_metrics = _args.evaluation_metrics,
    )
    
    experiment_config['visdom_args'] = dict(
        save_every_epoch=_args.save_every_epoch
    ) 

    dataset_params = get_dataset_specific_params(_args)

    # Instantiate main experiment class (PytorchLightning Module)
    experiment = EXPERIMENT_CLASS[_args.alg](model, experiment_config, dataset_params)

    
    #pdb.set_trace()

    trainer_params = get_trainer_params(_args, tb_logger)
    trainer = Trainer(**trainer_params)
    trainer.fit(experiment)


if __name__ == "__main__":

    cmdline_args = get_args(sys.argv[1:])

    setup_logging(cmdline_args.verbose)
    initialize_seeds(cmdline_args.seed)

    main(cmdline_args)
