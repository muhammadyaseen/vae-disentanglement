
from numpy.lib.function_base import select
import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim
from torchvision import transforms

from models.ladder_vae import LadderVAE
from common import constants as c
from common import data_loader
from common.visdom_visualiser import VisdomVisualiser
from evaluation import evaluation_utils

class LadderVAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: LadderVAE,
                 params: dict) -> None:
        
        super(LadderVAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.num_val_imgs = 0
        self.num_train_imgs = 0
        self.sample_loader = None
        self.curr_device = None
        self.visdom_on = params['visdom_on']
        self.save_dir = params['save_dir']
        self.l_zero_reg = params['l_zero_reg']
        
        if self.visdom_on:
            self.visdom_visualiser = VisdomVisualiser(params)

    def forward(self, x_input, **kwargs):
        
        return self.model.forward(x_input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        x_true1, label1 = batch

        self.curr_device = x_true1.device
        fwd_pass_results = self.forward(x_true1, label=label1)

        fwd_pass_results.update({
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx
        })
        
        losses = self.model.loss_function(loss_type='cross_ent', **fwd_pass_results)

        # TODO: find out a way to cleanly visualize mu and logvar of multiple layers
        #losses['mu_batch'] = mu.detach().cpu()
        #losses['logvar_batch'] = logvar.detach().cpu()

        return losses

    def training_step_end(self, train_step_output):
        pass
    
    def training_epoch_end(self, train_step_outputs):
        
        # TODO: figure out a way to do model / architecture specific or dataset specific 
        # logging w/o if-else jungle
        
        torch.set_grad_enabled(False)
        self.model.eval()
        scalar_metrics = dict()

        # 1. Save avg loss in this epoch
        avg_loss = torch.stack([tso[c.TOTAL_LOSS] for tso in train_step_outputs]).mean()
        avg_kld_loss = torch.stack([tso[c.KLD_LOSS] for tso in train_step_outputs]).mean()
        avg_kld_z1_loss = torch.stack([tso["kld_z1"] for tso in train_step_outputs]).mean()
        avg_kld_z2_loss = torch.stack([tso["kld_z2"] for tso in train_step_outputs]).mean()
        avg_recon_loss = torch.stack([tso[c.RECON] for tso in train_step_outputs]).mean()
        
        
        self.logger.experiment.add_scalar("Total Loss (Train)", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Recon Loss (Train)", avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar("KLD Loss (Train)", avg_kld_loss, self.current_epoch)
        self.logger.experiment.add_scalar("KLD Loss z1 (Train)", avg_kld_z1_loss, self.current_epoch)
        self.logger.experiment.add_scalar("KLD Loss z2 (Train)", avg_kld_z2_loss, self.current_epoch)

        if self.l_zero_reg:
            reg_loss = torch.stack([tso["l_zero_reg"] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar("Reg L-0 Loss", reg_loss, self.current_epoch)
            scalar_metrics["l_zero_reg"] = reg_loss

        # 2. save recon images and generated images, histogram of latent layer activations
        self.logger.experiment.add_image("Sampled Images", self._get_sampled_images(36), self.current_epoch)
        #self.logger.experiment.add_histogram("Latent Activations", self._get_latent_layer_activations()['mu'], self.current_epoch)       
        recon_grid = self._get_reconstructed_images()
        self.logger.experiment.add_image("Reconstructed Images", recon_grid, self.current_epoch)
        
        # 3. Evaluate disent metrics
        if self.params["evaluation_metrics"]:
            evaluation_results = evaluation_utils.evaluate_disentanglement_metric(self.model, 
                                                eval_results_dir=".",
                                                metric_names=self.params["evaluation_metrics"],
                                                dataset_name=self.params['dataset']
                            )

            # TODO: Use Tensorboard to visualize disent metrics
        
        scalar_metrics[c.TOTAL_LOSS] = avg_loss
        scalar_metrics[c.RECON] = avg_recon_loss
        scalar_metrics[c.KLD_LOSS] = avg_kld_loss
        scalar_metrics['kld_z1'] = avg_kld_z1_loss
        scalar_metrics['kld_z2'] = avg_kld_z2_loss

        torch.set_grad_enabled(True)
        self.model.train()

    def on_train_end(self):

        print("Training finished.")
        if self.visdom_on:
            self._save_visdom_environment()

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        x_true, labels = batch
        self.curr_device = x_true.device
        fwd_pass_results  = self.forward(x_true, labels = labels)

        fwd_pass_results.update({
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx
        })
        
        val_losses = self.model.loss_function(loss_type='cross_ent', **fwd_pass_results)
        
        return val_losses

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):

        optims = []
        scheds = []

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
                                            train_pct=0.90
                                            )

    def val_dataloader(self):
        
        self.sample_loader = data_loader.get_dataloader(self.params['dataset'],
                                            self.params['datapath'],
                                            shuffle=False,
                                            batch_size=64, 
                                            droplast=self.params['droplast'],
                                            num_workers=self.params['num_workers'],
                                            include_labels=None,
                                            pin_memory=self.params['pin_memory'],
                                            seed=self.params['seed'],
                                            image_size=self.params['image_size'],
                                            split="test",
                                            train_pct=0.90
                                            )
        return self.sample_loader

    def _get_sampled_images(self, how_many: int):

        curr_device = next(self.model.parameters()).device
        sampled_images = self.model.sample(how_many, curr_device)#.cpu().data
        grid_of_samples = vutils.make_grid(sampled_images, normalize=True, nrow=12, value_range=(0.0,1.0))
        return grid_of_samples

    def _get_reconstructed_images(self):

        # Get sample reconstruction image
        curr_device = next(self.model.parameters()).device
        test_input, test_label = next(iter(self.sample_loader))
        test_input = test_input.to(curr_device)
        
        fwd_pass_results = self.model.forward(test_input, labels = test_label)
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