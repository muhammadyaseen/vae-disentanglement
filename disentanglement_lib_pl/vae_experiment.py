import torch
from torch import optim
# from models import BaseVAE
# from models import BetaVAE_Vanilla
# from models.types_ import *
from common import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from models.vae import VAE
# from models.base.base_disentangler import BaseDisentangler
import visdom
from common.utils import VisdomDataGatherer

class VAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: VAE,
                 params: dict) -> None:
        
        super(VAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.num_val_imgs = 0
        self.num_train_imgs = 0
        self.sample_loader = None
        #self.global_step = 0
        
        # Visdom Visualization
        self.visdom_port = self.params['visdom_port']
        self.viz_name = "{} on {}".format(self.params['datapath'], self.params['dataset'])
        #self.visdom_instance = visdom.Visdom(port=self.visdom_port)
        self.visdom_gatherer = VisdomDataGatherer()
        self.eval_metrics = self.params['eval_metrics'] if self.params['eval_metrics'] is not None else []
        self.scalar_windows = ['recon_loss', 'total_loss', 'kld_loss', 'mu', 'var'] + self.eval_metrics
        self.visdom_scalar_windows = dict()
        for win in self.scalar_windows:
            self.visdom_scalar_windows[win] = None

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, x_input, **kwargs):
        
        #return self.model(input, **kwargs)
        # TODO: check if this is the right call
        return self.model.forward(x_input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        #self.global_step += 1
        x_true1, label1 = batch
        self.curr_device = x_true1.device

        # this is self.forward in PL which calls self.model(input)
        x_recon, z, mu, logvar = self.forward(x_true1, label=label1)

        loss_fn_args = dict(x_recon=x_recon, 
                            x_true=x_true1, 
                            mu=mu, 
                            logvar=logvar, 
                            z=z,
                            optimizer_idx=optimizer_idx,
                            batch_idx = batch_idx)
        
        losses = self.model.loss_function(**loss_fn_args)

        # TODO: This is where all mini-batch level visualization can be done
        # that were being done in log_save() method

        # end of mini-batch
        return losses

    def training_epoch_end(self, train_step_outputs):
        
        torch.set_grad_enabled(False)
        self.model.eval()

        # 0. Add graph / architecture
        if self.current_epoch == 0:
            rand_input = torch.rand((1, self.params['in_channels'],
                                     self.params['img_size'],
                                     self.params['img_size']))
            rand_input = rand_input.to(self.curr_device)
            self.logger.experiment.add_graph(self.model, rand_input)

        # 1. Save avg loss in this epoch
        avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        self.logger.experiment.add_scalar("Loss (Train)", avg_loss, self.current_epoch)

        # if isinstance(self.model, BetaVAE_Vanilla) and self.model.c_max is not None:
        #     self.logger.experiment.add_scalar("C", self.model.c_current, self.model.num_iter)

        # 2. save recon images and generated images
        self._log_reconstructed_images()
        self._log_sampled_images()

        # 3. histogram of latent layer activations
        self._log_latent_layer_activations()


        # TODO: when should we eval disent metrics ???
        # if self.evaluation_metric and is_time_for(self.iter, self.evaluate_iter):
        #    self.evaluate_results = evaluate_disentanglement_metric(self, metric_names=self.evaluation_metric)
        
        # TODO: show metrics on visdom
        # if is_time_for(self.iter, self.visdom_update_iter):
        #     self.update_visdom_visualisations(kwargs[c.INPUT_IMAGE],
        #                                       kwargs[c.RECON_IMAGE],
        #                                       kwargs['mu_batch'],
        #                                       kwargs['logvar_batch'],
        #                                       kwargs[c.LOSS])

        torch.set_grad_enabled(True)
        self.model.train()

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        x_true, labels = batch
        self.curr_device = x_true.device
        x_recon, z, mu, logvar  = self.forward(x_true, labels = labels)
        
        loss_fn_args = dict(x_recon=x_recon, 
                            x_true=x_true, 
                            mu=mu, 
                            logvar=logvar, 
                            z=z,
                            optimizer_idx=optimizer_idx,
                            batch_idx = batch_idx)
        
        val_losses = self.model.loss_function(**loss_fn_args)

        # TODO: Validation level visualization can be done here
        # self.visualize_recon(x_true, x_recon, test=True)
        # self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
        #                            data=(x_true, label), test=True)
        
        return val_losses

    def validation_end(self, outputs):

        """
        called at the end of every validation step

        :param outputs:
        :return:
        """
        print("validation_end called")
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        #self.sample_images() !! undefined ?
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

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])

        elif self.params['dataset'] in ['onedim', 'dsprites', 'continum', 'threeshapes', 'threeshapesnoisy']:
            transform = None

        else:
            raise ValueError('Undefined dataset type')
        return transform

    def _log_sampled_images(self):

        sampled_images = self.model.sample(36, self.curr_device).cpu().data
        grid_of_samples = vutils.make_grid(sampled_images, normalize=True, nrow=12, value_range=(0.0,1.0))
        self.logger.experiment.add_image("Sampled Images", grid_of_samples, global_step=self.global_step)

    def _log_reconstructed_images(self):

        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_loader))
        test_input = test_input.to(self.curr_device)

        recons, _, _, _ = self.model.forward(test_input, labels = test_label)
        recons = recons.cpu().data
        recons_grid = vutils.make_grid(recons, normalize=True, nrow=12, value_range=(0.0,1.0))
        self.logger.experiment.add_image("Reconstructed Images", recons_grid,
                                            global_step=self.global_step)

        del test_input, test_label, recons

    def _log_latent_layer_activations(self):

        # TODO: probably we should save hist over WHOLE val dataset and
        # not just a single batch

        test_input, _ = next(iter(self.sample_loader))
        test_input = test_input.to(self.curr_device)
        activations_mu, activations_logvar = self.model.encode(test_input)
        self.logger.experiment.add_histogram("Latent Activations", activations_mu, self.global_step)

