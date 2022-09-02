import os
import time
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim

from models.vae import VAE
from common import constants as c
from common import data_loader
#from common.visdom_visualiser import VisdomVisualiser
from evaluation import evaluation_utils


class BaseLatentToImageExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: nn.Module,
                 params: dict,
                 dataset_params: dict) -> None:
        
        super(BaseLatentToImageExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.sample_loader = None
        self.current_device = None
        self.visdom_on = params['visdom_on']
        self.save_dir = params['save_dir']
        self.max_epochs = params['max_epochs']
        self.dataset_params = dataset_params
        
    def forward(self, x_input, **kwargs):
        return self.model.forward(x_input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        super(BaseLatentToImageExperiment, self).training_step(batch, batch_idx, optimizer_idx)
        
        x_true, labels = batch
        self.current_device = x_true.device
        fwd_pass_results = self.forward(x_true, labels=labels, current_device=self.current_device)

        fwd_pass_results.update({
            'x_true': x_true,
            'true_latents': labels,
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'max_epochs': self.max_epochs
        })
        
        train_step_outputs = self.model.loss_function(loss_type='mse', **fwd_pass_results)

        return train_step_outputs
        
    def training_step_end(self, train_step_output):

        # This aggregation is required when we use multiple GPUs
        train_step_output[c.TOTAL_LOSS] = train_step_output[c.TOTAL_LOSS].mean()
        train_step_output[c.RECON] = train_step_output[c.RECON].mean()
        
    def on_train_epoch_start(self):
        
        timestamp = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time.time()))
        print(f"Epoch {self.current_epoch} / {self.max_epochs} started at {timestamp}")
    
    def on_train_epoch_end(self):
        
        timestamp = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time.time()))
        print(f"Epoch {self.current_epoch} / {self.max_epochs} ended at {timestamp}")
    
    def training_epoch_end(self, train_step_outputs):
        
        # TODO: figure out a way to do model / architecture specific or dataset specific 
        # logging w/o if-else jungle
        timestamp = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time.time()))
        print(f"training_epoch_end() called for epoch: {self.current_epoch} / {self.max_epochs} at {timestamp}")     
        
        torch.set_grad_enabled(False)
        self.model.eval()
        
        # log passed in params Once for reproducibility
        if self.current_epoch == 0:
            for param_name, param_val in self.params.items():
                self.logger.experiment.add_text(f"Script_Params/{param_name}", str(param_val), self.current_epoch)
            
            for param_name, param_val in self.dataset_params.items():
                self.logger.experiment.add_text(f"Script_Params/{param_name}", str(param_val), self.current_epoch)


        # 1. Save avg loss in this epoch
        avg_loss = torch.stack([tso[c.TOTAL_LOSS] for tso in train_step_outputs]).mean()
        avg_recon_loss = torch.stack([tso[c.RECON] for tso in train_step_outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Total Loss (Train)", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Loss/Reconstruction Loss (Train)", avg_recon_loss, self.current_epoch)

        # 2. save recon images and generated images, histogram of latent layer activations
        recon_grid = self._get_reconstructed_images()
        padded_epoch = str(self.current_epoch).zfill(len(str(self.max_epochs)))
        image_file_name = f"recon.epoch.{padded_epoch}.jpg"
        recon_images_path = os.path.join(self.logger.log_dir, "recon_images", image_file_name)
        vutils.save_image(recon_grid, recon_images_path)
        self.logger.experiment.add_image("Reconstructed Images", recon_grid, self.current_epoch)
        
        torch.set_grad_enabled(True)
        self.model.train()

    def on_train_end(self):
        print("Training finished.")
        
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        x_true, labels = batch
        self.current_device = x_true.device
        fwd_pass_results  = self.forward(x_true, labels=labels, current_device=self.current_device)

        fwd_pass_results.update({
            'x_true': x_true,
            'true_latents': labels,
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'max_epochs': self.max_epochs
        })
        
        val_losses = self.model.loss_function(loss_type='mse', **fwd_pass_results)
        
        return val_losses

    def validation_step_end(self, val_step_output):
                
        # This aggregation is required when we use multiple GPUs
        val_step_output[c.TOTAL_LOSS] = val_step_output[c.TOTAL_LOSS].mean()
        val_step_output[c.RECON] = val_step_output[c.RECON].mean()

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):

        optims = []
        scheds = []

        # TODO: add betas explicitly
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        # In non-pl version, it was like this.
        # self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        optims.append(optimizer)

        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            # TODO: figure out how to integrate scheduler that were used in non-pl code
            # self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
            #                      args.w_recon_scheduler, args.w_recon_scheduler_args)

            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        
        return data_loader.get_dataloader(self.params['dataset'],
                                            self.params['datapath'],
                                            shuffle=True,
                                            batch_size=self.params['batch_size'], 
                                            droplast=self.params['droplast'],
                                            num_workers=self.params['num_workers'],
                                            include_labels=None,
                                            pin_memory=self.params['pin_memory'],
                                            seed=self.params['seed'],
                                            image_size=self.params['image_size'],
                                            split="train",
                                            train_pct=0.90,
                                            **self.dataset_params
                                            )

    def val_dataloader(self):
        
        
        self.sample_loader = data_loader.get_dataloader(self.params['dataset'],
                                            self.params['datapath'],
                                            shuffle=True,
                                            batch_size=self.params['batch_size'],
                                            droplast=self.params['droplast'],
                                            num_workers=self.params['num_workers'],
                                            include_labels=None,
                                            pin_memory=self.params['pin_memory'],
                                            seed=self.params['seed'],
                                            image_size=self.params['image_size'],
                                            split="test",
                                            train_pct=0.10,
                                            **self.dataset_params
                                            )
        return self.sample_loader

    def _get_reconstructed_images(self):

        # Get sample reconstruction image
        current_device = next(self.model.parameters()).device
        test_input, test_label = next(iter(self.sample_loader))
        test_input = test_input.to(current_device)
        test_label = test_label.to(current_device)
        
        fwd_pass_results = self.model.forward(test_input, current_device=current_device, labels = test_label)
        
        recons = fwd_pass_results['x_recon']
        inputs_and_reconds_side_by_side = torch.cat([test_input, recons], dim = 3)
        img_input_vs_recon = vutils.make_grid(inputs_and_reconds_side_by_side, normalize=True, value_range=(0.0,1.0))
        
        return img_input_vs_recon
