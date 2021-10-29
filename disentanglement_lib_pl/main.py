import sys
import torch
import logging
import os
import yaml

from common.utils import setup_logging, initialize_seeds, set_environment_variables
from common.arguments import get_args
import models

from vae_experiment import VAEExperiment

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):


    # TODO: do we need it when we have visdom? should have some way to choose...
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

    # run test or train
    # if not _args.test:
    #     model.train()
    # else:
    #     model.test()

    config = dict()
    config['exp_params'] = dict(
        in_channels=_args.in_channels,
        img_size=64,    # fixed ???
        LR=_args.lr_G,
        weight_decay=0.0,
        
        # TODO: integrate data loading...
        dataset=_args.dset_name,
        datapath=_args.dset_dir,
        droplast=True,        
        batch_size=_args.batch_size,
        num_workers=_args.num_workers,
        pin_memory=_args.pin_memory,
        seed=_args.seed,
        image_size=_args.image_size,
        visdom_port=_args.visdom_port,
        eval_metrics=_args.evaluation_metric
    )

    # TODO: need to move this from yaml to cmdline args
    # for now we construct the dict out of cmdline args instead of reading values from YAML file
    experiment = VAEExperiment(model, config['exp_params'])

    config['trainer_params'] = dict(
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
                     **config['trainer_params'])

    pl_trainer.fit(experiment)

    # Run evaluation locally
    # The local_evaluation is implemented by aicrowd in the global namespace, so importing it suffices.
    #  todo: implement a modular version of local_evaluation
    # noinspection PyUnresolvedReferences
    # TODO: figure out how to integrate this gracefully
    # from aicrowd import local_evaluation


if __name__ == "__main__":

    _args = get_args(sys.argv[1:])
    #_args = None
    
    # Read-in config data
    # with open(_args.filename, 'r') as cfg_file:
    #     try:
    #         _args = yaml.safe_load(cfg_file)
    #     except yaml.YAMLError as ex:
    #         print(ex)
    
    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    # set the environment variables for dataset directory and name, and check if the root dataset directory exists.
    set_environment_variables(_args.dset_dir, _args.dset_name)
    assert os.path.exists(os.environ.get('DISENTANGLEMENT_LIB_DATA', '')), \
        'Root dataset directory does not exist at: \"{}\"'.format(_args.dset_dir)

    main(_args)
