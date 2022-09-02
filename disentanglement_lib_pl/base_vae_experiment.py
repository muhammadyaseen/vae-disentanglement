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
from common import utils
from evaluation import evaluation_utils


class BaseVAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: nn.Module,
                 params: dict,
                 dataset_params: dict) -> None:
        
        super(BaseVAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.sample_loader = None
        self.current_device = None
        self.visdom_on = params['visdom_on']
        self.save_dir = params['save_dir']
        self.max_epochs = params['max_epochs']
        self.dataset_params = dataset_params
        
        #if self.visdom_on:
        #    self.visdom_visualiser = VisdomVisualiser(params)

    def forward(self, x_input, **kwargs):
        return self.model.forward(x_input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        if batch_idx % 1000 == 0:
            print(f"Batch: {batch_idx} / {len(self.trainer.train_dataloader)}")

    def training_step_end(self, train_step_output):

        # This aggregation is required when we use multiple GPUs
        train_step_output[c.TOTAL_LOSS] = train_step_output[c.TOTAL_LOSS].mean()
        train_step_output[c.RECON] = train_step_output[c.RECON].mean()
        train_step_output[c.KLD_LOSS] = train_step_output[c.KLD_LOSS].mean()
            
        if isinstance(self.model, VAE) and self.model.controlled_capacity_increase:
            self.logger.experiment.add_scalar("C", self.model.c_current, self.global_step)

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
        avg_kld_loss = torch.stack([tso[c.KLD_LOSS] for tso in train_step_outputs]).mean()
        avg_recon_loss = torch.stack([tso[c.RECON] for tso in train_step_outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Total Loss (Train)", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Loss/Reconstruction Loss (Train)", avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Loss/Total KLD Loss (Train)", avg_kld_loss, self.current_epoch)


        # 2. save recon images and generated images, histogram of latent layer activations
        recon_grid = self._get_reconstructed_images()
        padded_epoch = str(self.current_epoch).zfill(len(str(self.max_epochs)))
        image_file_name = f"recon.epoch.{padded_epoch}.jpg"
        recon_images_path = os.path.join(self.logger.log_dir, "recon_images", image_file_name)
        vutils.save_image(recon_grid, recon_images_path)
        self.logger.experiment.add_image("Reconstructed Images", recon_grid, self.current_epoch)
        
        self.logger.experiment.add_image("Sampled Images", self._get_sampled_images(36), self.current_epoch)
        
        # 3. Evaluate disent metrics
        if self.params["evaluation_metrics"]:
            evaluation_results = evaluation_utils.evaluate_disentanglement_metric(self.model, 
                                                eval_results_dir=".",
                                                metric_names=self.params["evaluation_metrics"],
                                                dataset_name=self.params['dataset']
                            )
            # TODO: Use Tensorboard to visualize disent metrics
        
        torch.set_grad_enabled(True)
        self.model.train()

    def on_train_end(self):
        print("Training finished.")
        if self.visdom_on:
            self._save_visdom_environment()
        
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
        
        val_losses = self.model.loss_function(**fwd_pass_results)
        
        return val_losses

    def validation_step_end(self, val_step_output):
                
        # This aggregation is required when we use multiple GPUs
        val_step_output[c.TOTAL_LOSS] = val_step_output[c.TOTAL_LOSS].mean()
        val_step_output[c.RECON] = val_step_output[c.RECON].mean()
        val_step_output[c.KLD_LOSS] = val_step_output[c.KLD_LOSS].mean()

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
                                            shuffle=False,
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

    def _get_sampled_images(self, how_many: int):

        current_device = next(self.model.parameters()).device
        sampled_images = self.model.sample(how_many, current_device=current_device)
        grid_of_samples = vutils.make_grid(sampled_images, normalize=True, nrow=12, value_range=(0.0,1.0))
        return grid_of_samples

    def _get_reconstructed_images(self):

        # Get sample reconstruction image
        current_device = next(self.model.parameters()).device
        test_input, test_label = next(iter(self.sample_loader))
        test_input = test_input.to(current_device)

        fwd_pass_results = self.model.forward(test_input, current_device=current_device, labels = test_label)
        recons = fwd_pass_results['x_recon']
        inputs_and_reconds_side_by_side = torch.cat([test_input, recons], dim = 3)
        img_input_vs_recon = vutils.make_grid(inputs_and_reconds_side_by_side, normalize=True, value_range=(0.0,1.0))
        
        return img_input_vs_recon
        
    def _get_latent_layer_activations(self):

        # TODO: probably we should save hist over WHOLE val dataset and
        # not just a single batch

        test_input, _ = next(iter(self.sample_loader))
        test_input = test_input.to(self.curr_device)
        activations_mu, activations_logvar = self.model.encode(test_input)

        return {'mu': activations_mu, 'logvar': activations_logvar}

    def _save_visdom_environment(self):
        
        if self.visdom_on:
            print("Saving visdom environments to logs folder: (%s)" % self.save_dir)
            print("Available: ")
            for i, env_name in enumerate(self.visdom_visualiser.visdom_instance.get_env_list()):
                print(f"[{i}] - {env_name}")
                if env_name in self.visdom_visualiser.environmets:
                    self.visdom_visualiser.visdom_instance.save([env_name])
                    print("saved...")
        
