import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models.base.base_disentangler import BaseDisentangler
from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize
from common import constants as c


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        return self.decode(z)


class VAE(BaseDisentangler):
    """
    Auto-Encoding Variational Bayes
    by Kingma and Welling
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, args):
        
        super().__init__(args)

        # beta-vae hyper-parameters
        self.w_kld = args.w_kld
        self.controlled_capacity_increase = args.controlled_capacity_increase
        self.max_c = torch.tensor(args.max_c, dtype=torch.float)
        self.iterations_c = torch.tensor(args.iterations_c, dtype=torch.float)

        # As a little joke
        assert self.w_kld == 1.0 or self.alg != 'VAE', 'in vanilla VAE, w_kld should be 1.0. ' \
                                                       'Please use BetaVAE if intended otherwise.'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # model
        self.model = VAEModel(encoder(self.z_dim, self.num_channels, self.image_size),
                              decoder(self.z_dim, self.num_channels, self.image_size))

        # FactorVAE
        if c.FACTORVAE in self.loss_terms:
            from models.factorvae import factorvae_init
            self.PermD, self.optim_PermD = factorvae_init(args.discriminator[0], self.z_dim, self.num_layer_disc,
                                                          self.size_layer_disc, self.lr_D, self.beta1, self.beta2)

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(x=images)
        return mu

    def encode_stochastic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(x=images)
        return reparametrize(mu, logvar)

    def _kld_loss_fn(self, mu, logvar):
        if not self.controlled_capacity_increase:
            kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        else:
            """
            Based on: Understanding disentangling in β-VAE
            https://arxiv.org/pdf/1804.03599.pdf
            """
            capacity = torch.min(self.max_c, self.max_c * torch.tensor(self.iter) / self.iterations_c)
            kld_loss = (kl_divergence_mu0_var1(mu, logvar) - capacity).abs() * self.w_kld
        return kld_loss

    def loss_function(self, **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu, logvar = kwargs['mu'], kwargs['logvar']

        bs = self.batch_size
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        #output_losses[c.TOTAL_VAE] = input_losses.get(c.TOTAL_VAE, 0)
        output_losses[c.TOTAL_LOSS] = 0

        output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        output_losses['kld'] = self._kld_loss_fn(mu, logvar)
        output_losses[c.TOTAL_LOSS] += output_losses['kld']

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

        return output_losses

    def vae_base_forward(self, x_true, **kwargs):
        
        mu, logvar = self.model.encode(x=x_true1, c=label1)
        z = reparametrize(mu, logvar)
        x_recon = self.model.decode(z=z, c=label1)
        
        return {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar}
