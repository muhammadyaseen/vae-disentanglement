import math
import torch
from torch import optim
from models import BaseVAE
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
        self.curr_device = None
        self.hold_graph = False
        self.num_val_imgs = 0
        self.num_train_imgs = 0

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):

        real_img, labels = batch
        self.curr_device = real_img.device
        #print("train_step dev:" , self.curr_device)

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):



        real_img, labels = batch
        self.curr_device = real_img.device
        #print(real_img.shape)
        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def save_sample_images(self):

        # Get sample reconstruction image
        #print("Curr device: ", self.curr_device)
        test_input, test_label = next(iter(self.sample_dataloader))
        #test_input = test_input.to(self.curr_device)
        #test_label = test_label.to(self.curr_device)

        #print(self.model)

        recons = self.model.generate(test_input, labels = test_label)
        print(recons.shape)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(36,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except Exception as ex:
            print("Passing...", ex)
            pass


        del test_input, recons #, samples


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

    #@data_loader
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
            dataset = DSpritesDataset(root=self.params['data_path'], split="train")

        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset) if dataset is not None else 0

        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)

    #@data_loader
    def val_dataloader(self):
        transform = self.data_transforms()
        val_dataset = None

        if self.params['dataset'] == 'celeba':
            val_dataset = CelebA(root = self.params['data_path'], split = "test", transform=transform, download=True)

        elif self.params['dataset'] == 'onedim':
            val_dataset = OneDimLatentDataset(root=self.params['data_path'], split = "test")

        elif self.params['dataset'] == 'dsprites':
            val_dataset = DSpritesDataset(root=self.params['data_path'], split="train")

        else:
            raise ValueError('Undefined dataset type')

        self.sample_dataloader = DataLoader(val_dataset, batch_size= 144, shuffle = False, drop_last=True)

        self.num_val_imgs = len(val_dataset) if val_dataset is not None else 0

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

        elif self.params['dataset'] == 'onedim':
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
