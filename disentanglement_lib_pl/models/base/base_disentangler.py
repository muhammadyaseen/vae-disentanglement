import os
import logging

import torch
import torchvision.utils


import common.constants as c


DEBUG = False


class BaseDisentangler(object):

    def __init__(self, args):

        # Misc
        self.name = args.name
        self.alg = args.alg
        self.loss_terms = args.loss_terms
        
        # TODO: this logic shouldn't be here...
        # self.lr_scheduler = None
        # self.w_recon_scheduler = None
        # self.optim_G = None

        # Output directory
        # TODO: this logic shouldn't be here...
        # self.train_output_dir = os.path.join(args.train_output_dir, self.name)
        # self.test_output_dir = os.path.join(args.test_output_dir, self.name)
        # self.file_save = args.file_save
        # self.gif_save = args.gif_save
        # os.makedirs(self.train_output_dir, exist_ok=True)
        # os.makedirs(self.test_output_dir, exist_ok=True)

        # Latent space
        self.z_dim = args.z_dim
        self.l_dim = args.l_dim
        self.num_labels = args.num_labels

        # Loss weights
        self.w_recon = args.w_recon

        # Solvers
        # TODO: this logic shouldn't be here...
        # self.beta1 = args.beta1
        # self.beta2 = args.beta2
        # self.lr_G = args.lr_G
        # self.lr_D = args.lr_D
        # self.max_epoch = int(args.max_epoch)

        # Data
        # TODO: this logic shouldn't be here...
        # self.dset_dir = args.dset_dir
        # self.dset_name = args.dset_name
        # self.batch_size = args.batch_size
        # self.image_size = args.image_size
        # self.num_channels = args.in_channels

        # TODO: this data loader logic shouldn't be here.
        # Find a respectable home for this
        # from common.data_loader import get_dataloader
        # self.data_loader = get_dataloader(args.dset_name, args.dset_dir, args.batch_size, args.seed, args.num_workers,
        #                                   args.image_size, args.include_labels, args.pin_memory, not args.test,
        #                                   not args.test)

        # only used if some supervision was imposed such as in Conditional VAE
        # TODO: Should this logic be here ??? find a good place for it..
        # if self.data_loader.dataset.has_labels():
        #     self.num_classes = self.data_loader.dataset.num_classes()
        #     self.total_num_classes = sum(self.data_loader.dataset.num_classes(False))
        #     self.class_values = self.data_loader.dataset.class_values()

        # self.num_channels = self.data_loader.dataset.num_channels()
        # self.num_batches = len(self.data_loader)

        # logging.info('Number of samples: {}'.format(len(self.data_loader.dataset)))
        # logging.info('Number of batches per epoch: {}'.format(self.num_batches))
        # logging.info('Number of channels: {}'.format(self.num_channels))

        # logging
        self.info_cumulative = {}
        self.iter = 0
        self.epoch = 0
        self.evaluate_results = dict()

        # logging iterations
        # self.traverse_iter = args.traverse_iter if args.traverse_iter else self.num_batches
        # self.evaluate_iter = args.evaluate_iter if args.evaluate_iter else self.num_batches
        # self.ckpt_save_iter = args.ckpt_save_iter if args.ckpt_save_iter else self.num_batches
        # self.schedulers_iter = args.schedulers_iter if args.schedulers_iter else self.num_batches
        # self.visdom_update_iter = args.visdom_update_iter if args.visdom_update_iter else self.num_batches

        # traversing the latent space
        self.traverse_min = args.traverse_min
        self.traverse_max = args.traverse_max
        self.traverse_spacing = args.traverse_spacing
        self.traverse_z = args.traverse_z
        self.traverse_l = args.traverse_l
        self.traverse_c = args.traverse_c
        self.white_line = None


        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.net_dict = dict()
        self.optim_dict = dict()

        # model is the only attribute that all sub-classes should have
        self.model = None
        
        # FactorVAE args
        # TODO: we shouldn't be needing device info here...
        # move this to some other place
        # self.ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device, requires_grad=False)
        # self.zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device, requires_grad=False)
        # self.num_layer_disc = args.num_layer_disc
        # self.size_layer_disc = args.size_layer_disc

        # FactorVAE & BetaTCVAE args
        self.w_tc = args.w_tc

        # InfoVAE args
        self.w_infovae = args.w_infovae

        # DIPVAE args
        self.w_dipvae = args.w_dipvae

        # DIPVAE args
        self.lambda_od = args.lambda_od
        self.lambda_d_factor = args.lambda_d_factor
        self.lambda_d = self.lambda_d_factor * self.lambda_od

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if len(images.size()) == 3:
            images = images.unsqueeze(0)
        return self.model.encode(images)

    def encode_stochastic(self, **kwargs):
        raise NotImplementedError

    def decode(self, **kwargs):
        latent = kwargs['latent']
        if len(latent.size()) == 1:
            latent = latent.unsqueeze(0)
        return self.model.decode(latent)

    @staticmethod
    def set_z(z, latent_id, val):
        z[:, latent_id] = val

    def set_l(self, l, label_id, latent_id, val):
        l[:, label_id * self.l_dim + latent_id] = val

    def loss_fn(self, **kwargs):
        raise NotImplementedError

    def sample(self,
               num_samples:int,
               current_device: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.z_dim)
        z = z.to(current_device)
        return self.decode(z)

"""
Ignoring schedularing stuff for now

    def schedulers_step(self, validation_loss=None, step_num=None):
        self.lr_scheduler_step(validation_loss)
        self.w_recon_scheduler_step(step_num)

    def lr_scheduler_step(self, validation_loss):
        if self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(validation_loss)
        else:
            self.lr_scheduler.step()

    def w_recon_scheduler_step(self, step_num):
        if self.w_recon_scheduler is None:
            return
        self.w_recon = self.w_recon_scheduler.step(step_num)

    def setup_schedulers(self, lr_scheduler, lr_scheduler_args, w_recon_scheduler, w_recon_scheduler_args):
        self.lr_scheduler = get_scheduler(self.optim_G, lr_scheduler, lr_scheduler_args)
        self.w_recon_scheduler = get_scheduler(self.w_recon, w_recon_scheduler, w_recon_scheduler_args)
"""
