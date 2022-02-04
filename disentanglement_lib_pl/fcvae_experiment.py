
from numpy.lib.function_base import select
import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim
from torchvision import transforms

from models.fc_vae import FC_VAE
from common import constants as c
from common import data_loader
from common.visdom_visualiser import VisdomVisualiser
from evaluation import evaluation_utils

class FCVAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: FC_VAE,
                 params: dict) -> None:
        
        super(FCVAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.num_val_imgs = 0
        self.num_train_imgs = 0
        self.sample_loader = None
        self.curr_device = None
        self.visdom_on = params['visdom_on']
        self.save_dir = params['save_dir']

        if self.visdom_on:
            self.visdom_visualiser = VisdomVisualiser(params)

    def forward(self, x_input, **kwargs):
        
        return self.model.forward(x_input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        x_true1, label1 = batch
        self.curr_device = x_true1.device
        x_recon, mu, z, logvar = self.forward(x_true1, label=label1)

        loss_fn_args = dict(x_recon=x_recon, 
                            x_true=x_true1, 
                            mu=mu, 
                            logvar=logvar, 
                            z=z,
                            optimizer_idx=optimizer_idx,
                            batch_idx = batch_idx)
        
        losses = self.model.loss_function(loss_type='mse', **loss_fn_args)

        losses['mu_batch'] = mu.detach().cpu()
        losses['logvar_batch'] = logvar.detach().cpu()

        return losses

    def training_step_end(self, train_step_output):
        pass
    
    def training_epoch_end(self, train_step_outputs):
        
        # TODO: figure out a way to do model / architecture specific or dataset specific 
        # logging w/o if-else jungle
        
        torch.set_grad_enabled(False)
        self.model.eval()

        # 1. Save avg loss in this epoch
        avg_loss = torch.stack([tso[c.TOTAL_LOSS] for tso in train_step_outputs]).mean()
        avg_kld_loss = torch.stack([tso[c.KLD_LOSS] for tso in train_step_outputs]).mean()
        avg_recon_loss = torch.stack([tso[c.RECON] for tso in train_step_outputs]).mean()

        self.logger.experiment.add_scalar("Total Loss (Train)", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Recon Loss (Train)", avg_recon_loss, self.current_epoch)
        self.logger.experiment.add_scalar("KLD Loss (Train)", avg_kld_loss, self.current_epoch)


        # if isinstance(self.model, BetaVAE_Vanilla) and self.model.c_max is not None:
        #     self.logger.experiment.add_scalar("C", self.model.c_current, self.model.num_iter)

        # 2. save recon images and generated images, histogram of latent layer activations
        self.logger.experiment.add_histogram("Latent Activations", self._get_latent_layer_activations()['mu'], self.current_epoch)       
        
        # 3. Evaluate disent metrics
        if self.params["evaluation_metrics"]:
            evaluation_results = evaluation_utils.evaluate_disentanglement_metric(self.model, 
                                                eval_results_dir=".",
                                                metric_names=self.params["evaluation_metrics"],
                                                dataset_name=self.params['dataset']
                            )
            if self.visdom_on:
                self.visdom_visualiser.visualize_disentanglement_metrics(evaluation_results, self.current_epoch)

        scalar_metrics = dict()
        scalar_metrics[c.TOTAL_LOSS] = avg_loss
        scalar_metrics[c.RECON] = avg_recon_loss
        scalar_metrics[c.KLD_LOSS] = avg_kld_loss

        if 'BetaTCVAE' in self.model.loss_terms:
            avg_tc_loss = torch.stack([tso['vae_betatc'] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar("TC Loss (Train)", avg_tc_loss, self.current_epoch)
            scalar_metrics['vae_betatc'] = avg_tc_loss


        if self.visdom_on:
            #self.visdom_visualiser.visualize_reconstruction(x_inputs,x_recons, self.current_epoch)
            self.visdom_visualiser.visualize_scalar_metrics(scalar_metrics, self.current_epoch)
            self.visdom_visualiser.visualize_multidim_metrics( {
                                            "mu_batch" : torch.stack([tso['mu_batch'] for tso in train_step_outputs]),
                                            "var_batch" : torch.stack([tso['logvar_batch'].exp() for tso in train_step_outputs]),
                                        }, self.current_epoch)
        torch.set_grad_enabled(True)
        self.model.train()

    def on_train_end(self):

        print("Training finished.")
        if self.visdom_on:
            print("Saving visdom environments to logs folder: (%s)" % self.save_dir)
            print("Available: ")
            for i, env_name in enumerate(self.visdom_visualiser.visdom_instance.get_env_list()):
                print(f"[{i}] - {env_name}")
                if env_name in self.visdom_visualiser.environmets:
                    self.visdom_visualiser.visdom_instance.save([env_name])
                    print("saved...")

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        x_true, labels = batch
        #print(x_true.shape)
        self.curr_device = x_true.device
        #print("before val fwd")
        x_recon, mu, z, logvar  = self.forward(x_true, labels = labels)
        #print("after val fwd")

        loss_fn_args = dict(x_recon=x_recon, 
                            x_true=x_true, 
                            mu=mu, 
                            logvar=logvar, 
                            z=z,
                            optimizer_idx=optimizer_idx,
                            batch_idx = batch_idx)
        
        val_losses = self.model.loss_function(loss_type='mse', **loss_fn_args)
        
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
                                            image_size=64,
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
                                            image_size=64,
                                            split="test",
                                            train_pct=0.90
                                            )
        return self.sample_loader

        
    def _get_latent_layer_activations(self):

        # TODO: probably we should save hist over WHOLE val dataset and
        # not just a single batch

        test_input, _ = next(iter(self.sample_loader))
        test_input = test_input.to(self.curr_device)
        activations_mu, activations_logvar = self.model.encode(test_input)

        return {'mu': activations_mu, 'logvar': activations_logvar}
        

