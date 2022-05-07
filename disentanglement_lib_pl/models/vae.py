import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize
from common import constants as c


class VAE(nn.Module):
    """
    Auto-Encoding Variational Bayes
    by Kingma and Welling
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, args):
        
        super().__init__()

        # Misc
        # self.name = args.name
        self.alg = args.alg
        self.loss_terms = args.loss_terms

        # Misc params related to data
        self.z_dim = args.z_dim[0]
        self.l_dim = args.l_dim
        self.num_labels = args.num_labels
        self.num_channels = args.in_channels
        self.image_size = args.image_size
        self.batch_size = args.batch_size

        # Weight of recon loss
        self.w_recon = args.w_recon

        # Beta-vae and Annealed Beta-VAE args
        self.w_kld = args.w_kld
        self.controlled_capacity_increase = args.controlled_capacity_increase
        self.max_c = torch.tensor(args.max_c, dtype=torch.float)
        self.iterations_c = torch.tensor(args.iterations_c, dtype=torch.float)
        self.current_c = torch.tensor(0.0)

        # As a little joke
        assert self.w_kld == 1.0 or self.alg != 'VAE', 'in vanilla VAE, w_kld should be 1.0. ' \
                                                       'Please use BetaVAE if intended otherwise.'
        
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

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # model
        self.encoder = encoder(self.z_dim, self.num_channels, self.image_size)
        self.decoder = decoder(self.z_dim, self.num_channels, self.image_size)

        # FactorVAE
        # TODO: we can probably keep permD here, but should move optim_PermD to 
        # a place where we config other optimizers...
        # if c.FACTORVAE in self.loss_terms:
        #     from models.factorvae import factorvae_init
        #     self.PermD, self.optim_PermD = factorvae_init(args.discriminator[0], self.z_dim, self.num_layer_disc,
        #                                                   self.size_layer_disc, self.lr_D, self.beta1, self.beta2)

    def _kld_loss_fn(self, mu, logvar, **kwargs):
        if not self.controlled_capacity_increase:
            kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        else:
            """
            Based on: Understanding disentangling in β-VAE
            https://arxiv.org/pdf/1804.03599.pdf
            """
            # TODO: change `self.iter` to instead use `pl.LightningModule.{global_step | current_epoch}`
            # Dunno if doing that is ok for multi-gpu etc
            global_iter = kwargs['global_step']
            capacity = torch.min(self.max_c, self.max_c * torch.tensor(global_iter) / self.iterations_c)
            self.current_c = capacity.detach()
            kld_loss = (kl_divergence_mu0_var1(mu, logvar) - capacity).abs() * self.w_kld
        return kld_loss

    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu, logvar = kwargs['mu'], kwargs['logvar']
        global_step = kwargs['global_step']
        bs = self.batch_size
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0

        if loss_type == 'cross_ent':
            output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        
        if loss_type == 'mse':
            output_losses[c.RECON] = F.mse_loss(x_recon, x_true, reduction='sum') / bs * self.w_recon

        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        output_losses[c.KLD_LOSS] = self._kld_loss_fn(mu, logvar, global_step=global_step)
        output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]

        if c.FACTORVAE in self.loss_terms:
            from models.factorvae import factorvae_loss_fn
            output_losses['vae_tc_factor'], output_losses['discriminator_tc'] = factorvae_loss_fn(
                self.w_tc, self.model, self.PermD, self.optim_PermD, self.ones, self.zeros, **kwargs)
            output_losses[c.TOTAL_LOSS] += output_losses['vae_tc_factor']

        if c.DIPVAE_I in self.loss_terms:
            from models.dipvae import dipvaei_loss_fn
            output_losses['vae_dipi'] = dipvaei_loss_fn(self.w_dipvae, self.lambda_od, self.lambda_d, **kwargs)
            output_losses[c.TOTAL_LOSS] += output_losses['vae_dipi']

        if c.DIPVAE_II in self.loss_terms:
            from models.dipvae import dipvaeii_loss_fn
            output_losses['vae_dipii'] = dipvaeii_loss_fn(self.w_dipvae, self.lambda_od, self.lambda_d, **kwargs)
            output_losses[c.TOTAL_LOSS] += output_losses['vae_dipii']

        if c.BetaTCVAE in self.loss_terms:
            from models.betatcvae import betatcvae_loss_fn
            output_losses['vae_betatc'] = betatcvae_loss_fn(self.w_tc, **kwargs)
            output_losses[c.TOTAL_LOSS] += output_losses['vae_betatc']

        if c.INFOVAE in self.loss_terms:
            from models.infovae import infovae_loss_fn
            output_losses['vae_mmd'] = infovae_loss_fn(self.w_infovae, self.z_dim, self.device, **kwargs)
            output_losses[c.TOTAL_LOSS] += output_losses['vae_mmd']

        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        #print(" in loss fn ", mu.shape)
        return output_losses

    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z, **kwargs))

    def forward(self, x_true, **kwargs):
        
        mu, logvar = self.encode(x=x_true)
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z=z)
        return x_recon, mu, z, logvar

    def sample(self, num_samples, current_device):
        
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(current_device)
        return self.decode(z)