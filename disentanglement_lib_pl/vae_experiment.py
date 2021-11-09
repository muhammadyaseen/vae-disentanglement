
import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim
from torchvision import transforms

from models.vae import VAE
from common import constants as c
from common import data_loader
from common.visdom_visualiser import VisdomVisualiser
from evaluation import evaluation_utils

class VAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: VAE,
                 params: dict) -> None:
        
        super(VAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.num_val_imgs = 0
        self.num_train_imgs = 0
        self.sample_loader = None
        self.curr_device = None
        self.visdom_visualiser = VisdomVisualiser(params['visual_args'], params['visdom_args'])

    def forward(self, x_input, **kwargs):
        
        return self.model.forward(x_input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        x_true1, label1 = batch
        self.curr_device = x_true1.device
        x_recon, z, mu, logvar = self.forward(x_true1, label=label1)

        loss_fn_args = dict(x_recon=x_recon, 
                            x_true=x_true1, 
                            mu=mu, 
                            logvar=logvar, 
                            z=z,
                            optimizer_idx=optimizer_idx,
                            batch_idx = batch_idx)
        
        losses = self.model.loss_function(**loss_fn_args)

        return losses

    def training_epoch_end(self, train_step_outputs):
        
        # TODO: figure out a way to do model / architecture specific or dataset specific 
        # loggin w/o if-else jungle
        
        torch.set_grad_enabled(False)
        self.model.eval()

        # 0. Add graph / architecture
        # TODO: for some reason this causes TracerWarning. Commenting it for now
        # if self.current_epoch == 0:
        #     rand_input = torch.rand((1, self.params['in_channels'],
        #                              self.params['image_size'],
        #                              self.params['image_size']))
        #     rand_input = rand_input.to(self.model.device)
        #     self.logger.experiment.add_graph(self.model, rand_input)

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
        self.logger.experiment.add_image("Sampled Images", self._get_sampled_images(36), self.current_epoch)
        
        recon_grid, x_recons, x_inputs = self._get_reconstructed_images()
        self.logger.experiment.add_image("Reconstructed Images", recon_grid, self.current_epoch)
        
        self.logger.experiment.add_histogram("Latent Activations", self._get_latent_layer_activations()['mu'], self.current_epoch)
        
        # 3. Evaluate disent metrics
        if self.params["evaluation_metrics"]:
            evaluation_results = evaluation_utils.evaluate_disentanglement_metric(self.model, 
                                                eval_results_dir=".",
                                                metric_names=self.params["evaluation_metrics"],
                                                dataset_name=self.params['dataset']
                            )
            self.visdom_visualiser.visualise_disentanglement_metrics(evaluation_results, self.current_epoch)

        scalar_metrics = dict()
        scalar_metrics[c.TOTAL_LOSS] = avg_loss
        scalar_metrics[c.RECON] = avg_recon_loss
        scalar_metrics[c.KLD_LOSS] = avg_kld_loss


        # reconloss, kldloss, totalloss, mu,var,capacity etc
        self.visdom_visualiser.visualize_reconstruction(x_inputs,x_recons, self.current_epoch)
        self.visdom_visualiser.visualize_scalar_metrics(scalar_metrics, self.current_epoch)
        
        torch.set_grad_enabled(True)
        self.model.train()

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        x_true, labels = batch
        self.curr_device = x_true.device
        #print("before val fwd")
        x_recon, z, mu, logvar  = self.forward(x_true, labels = labels)
        #print("after val fwd")

        loss_fn_args = dict(x_recon=x_recon, 
                            x_true=x_true, 
                            mu=mu, 
                            logvar=logvar, 
                            z=z,
                            optimizer_idx=optimizer_idx,
                            batch_idx = batch_idx)
        
        val_losses = self.model.loss_function(**loss_fn_args)
        
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
                                            image_size=self.params['image_size']
                                            )

    def val_dataloader(self):
        
        # for now we just return the same data
        # TODO: implement some kind of disjoing split
        self.sample_loader = self.train_dataloader()
        return self.sample_loader

    def _get_sampled_images(self, how_many: int):

        sampled_images = self.model.sample(how_many, self.curr_device).cpu().data
        grid_of_samples = vutils.make_grid(sampled_images, normalize=True, nrow=12, value_range=(0.0,1.0))
        return grid_of_samples

    def _get_reconstructed_images(self):

        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_loader))
        test_input = test_input.to(self.curr_device)

        recons, _, _, _ = self.model.forward(test_input, labels = test_label)
        recons = recons.cpu().data
        recons_grid = vutils.make_grid(recons, normalize=True, nrow=12, value_range=(0.0,1.0))
        
        return recons_grid, test_input.cpu().data, recons
        
    def _get_latent_layer_activations(self):

        # TODO: probably we should save hist over WHOLE val dataset and
        # not just a single batch

        test_input, _ = next(iter(self.sample_loader))
        test_input = test_input.to(self.curr_device)
        activations_mu, activations_logvar = self.model.encode(test_input)

        return {'mu': activations_mu, 'logvar': activations_logvar}
        

