import math
import torch
from torch import optim
from models import BaseVAE
from models import BetaVAE_Vanilla
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

from one_dim_data_loader import OneDimLatentDataset
from dsprites_loader import DSpritesDataset

class VAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEExperiment, self).__init__()

        self.model = vae_model
        #print("Model dev: ", self.model.device)
        self.params = params
        self.curr_device = torch.device("cuda:1")
        self.hold_graph = False
        self.num_val_imgs = 0
        self.num_train_imgs = 0

        #self.logger.experiment.add_graph(vae_model, torch.rand((1,3,64,64)))

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):

        real_img, labels = batch
        self.curr_device = real_img.device
        #print("training_step:" , self.curr_device)

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        return train_loss

    def training_epoch_end(self, train_step_outputs):

        # this function is called after the epoch has completed

        # 0.
        if self.current_epoch == 0:
            rand_input = torch.rand((1, self.params['in_channels'],
                                     self.params['img_size'],
                                     self.params['img_size']))
            rand_input = rand_input.to(self.curr_device)
            self.logger.experiment.add_graph(self.model, rand_input)

        # 1. Save avg loss in this epoch
        avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        self.logger.experiment.add_scalar("Loss (Train)", avg_loss, self.current_epoch)

        if isinstance(self.model, BetaVAE_Vanilla) and self.model.c_max is not None:
            self.logger.experiment.add_scalar("C", self.model.c_max, self.model.num_iter)

        # 2. save recon images and generated images
        self._log_reconstructed_images()
        self._log_sampled_images()

        # 3. histogram of latent layer activations
        self._log_latent_layer_activations()

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):



        real_img, labels = batch
        self.curr_device = real_img.device
        #print("validation_step:" , self.curr_device)
        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

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


    def _log_sampled_images(self):

        samples = self.model.sample(36, self.curr_device)

        grid_of_samples = vutils.make_grid(samples.cpu().data, normalize=True, nrow=12)
        self.logger.experiment.add_image("Sampled Images", grid_of_samples, global_step=self.current_epoch)

        # vutils.save_image(samples.cpu().data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"sampled_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)
        del samples

    def _log_reconstructed_images(self):

        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        print("curr dev:", self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        recons_grid = vutils.make_grid(recons.cpu().data, normalize=True, nrow=12)
        self.logger.experiment.add_image("Reconstructed Images", recons_grid, global_step=self.current_epoch)

        # vutils.save_image(recons.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        del test_input, test_label, recons

    def _log_latent_layer_activations(self):

        # TODO: probably we should save hist over WHOLE val dataset and
        # not just a single batch

        test_input, _ = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        activations_mu, activations_logvar = self.model.encode(test_input)
        self.logger.experiment.add_histogram("Latent Activations", activations_mu, self.current_epoch)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
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

        transform = self.data_transforms()
        dataset = None

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)

        elif self.params['dataset'] == 'onedim':
            dataset = OneDimLatentDataset(root=self.params['data_path'],
                                          split="train")

        elif self.params['dataset'] == 'dsprites':
            dataset = DSpritesDataset(root=self.params['data_path'], split="train", transform=transform)

        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset) if dataset is not None else 0
        print("# train : ", self.num_train_imgs)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True
                         ,num_workers=self.params['num_workers'])

    def val_dataloader(self):
        transform = self.data_transforms()
        val_dataset = None

        if self.params['dataset'] == 'celeba':
            val_dataset = CelebA(root = self.params['data_path'], split = "test", transform=transform, download=True)

        elif self.params['dataset'] == 'onedim':
            val_dataset = OneDimLatentDataset(root=self.params['data_path'], split = "test")

        elif self.params['dataset'] == 'dsprites':
            val_dataset = DSpritesDataset(root=self.params['data_path'], split="test", transform=transform)

        else:
            raise ValueError('Undefined dataset type')

        self.sample_dataloader = DataLoader(val_dataset, batch_size= self.params['batch_size'],
                                            shuffle = False, drop_last=True
                                            ,num_workers=self.params['num_workers'])

        self.num_val_imgs = len(val_dataset) if val_dataset is not None else 0
        print("# val : ", self.num_val_imgs)

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])

        elif self.params['dataset'] in ['onedim', 'dsprites']:
            transform = None

        else:
            raise ValueError('Undefined dataset type')
        return transform


    def save_latent_codes(self, save_dir):

        circle_mus, circle_logvars, square_mus, square_logvars = [], [], [], []

        print("len: ", len(self.sample_dataloader))
        # get the batch
        for test_input, test_label in self.sample_dataloader:

            test_label = test_label.detach().cpu().numpy()

            # get codes for the batch
            code_mus, code_logvars = self.model.encode(test_input)

            code_mus = code_mus.detach().cpu().numpy()
            code_logvars = code_logvars.detach().cpu().numpy()

            # accumulate mu and log_vars according to the 'class' they belong to
            for i, (mu, logvar) in enumerate(zip(code_mus, code_logvars)):

                print(i, test_label[i][0], mu, logvar, type(mu), type(logvar))

                if test_label[i][0] == 0:
                    circle_mus.append(mu.item())
                    circle_logvars.append(logvar.item())

                if test_label[i][0] == 1:
                    square_mus.append(mu.item())
                    square_logvars.append(logvar.item())
        import os
        import json
        json.dump({
            'circle_mus': circle_mus,
            'circle_logvars':circle_logvars,
            'square_mus': square_mus,
            'square_logvars':square_logvars
        }, open(os.path.join(save_dir,'latent_codes.pt'),'w'))



