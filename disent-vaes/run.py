import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args, _ = parser.parse_known_args()

    # Read-in config data
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEExperiment(model, config['exp_params'])
    pl_trainer = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     logger=tt_logger,
                     flush_logs_every_n_steps=100,
                     limit_train_batches=1.,
                     limit_val_batches=1.,
                     num_sanity_val_steps=5,
                     callbacks = None,
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")

    pl_trainer.fit(experiment)

    # Save reconstructions of images in val set
    experiment.save_sample_images()
