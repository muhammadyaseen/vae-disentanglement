import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize, kl_divergence_mu0_var1_per_node
from common import constants as c
from common import utils

class VAE(nn.Module):
    """
    Auto-Encoding Variational Bayes
    by Kingma and Welling
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, network_args, **kwargs):
        
        super().__init__()

        # Misc
        # self.name = args.name
        self.alg = network_args.alg
        self.loss_terms = network_args.loss_terms
        self.dataset = network_args.dset_name
        self.loss_type = utils.get_loss_type_for_dataset(self.dataset)

        # Misc params related to data
        self.z_dim = network_args.z_dim[0]
        self.l_dim = network_args.l_dim
        #self.num_labels = network_args.num_labels
        self.num_channels = network_args.in_channels
        self.image_size = network_args.image_size
        self.batch_size = network_args.batch_size

        # Weight of recon loss
        self.w_recon = network_args.w_recon

        # Beta-vae and Annealed Beta-VAE args
        self.w_kld = network_args.w_kld
        self.controlled_capacity_increase = network_args.controlled_capacity_increase
        self.max_capacity = torch.tensor(network_args.max_capacity, dtype=torch.float)
        self.iterations_c = torch.tensor(network_args.iterations_c, dtype=torch.float)
        self.current_c = torch.tensor(0.0)
        self.kl_warmup_epochs = network_args.kl_warmup_epochs
        
        # FactorVAE & BetaTCVAE args
        self.w_tc = network_args.w_tc

        # InfoVAE args
        self.w_infovae = network_args.w_infovae

        # DIPVAE args
        self.w_dipvae = network_args.w_dipvae

        # DIPVAE args
        self.lambda_od = network_args.lambda_od
        self.lambda_d_factor = network_args.lambda_d_factor
        self.lambda_d = self.lambda_d_factor * self.lambda_od

        # encoder and decoder
        encoder_name = network_args.encoder[0]
        decoder_name = network_args.decoder[0]
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

        loss_per_node = dict()

        if not self.controlled_capacity_increase:
            #kld_loss = kl_divergence_mu0_var1(mu, logvar) 
            kld_loss = kl_divergence_mu0_var1_per_node(mu, logvar)
            for node_idx, node_kld_loss in enumerate(kld_loss):
                loss_per_node[f'KLD_z_{node_idx}'] = node_kld_loss.detach()
        else:
            """
            Based on: Understanding disentangling in β-VAE
            https://arxiv.org/pdf/1804.03599.pdf
            """
            # TODO: change `self.iter` to instead use `pl.LightningModule.{global_step | current_epoch}`
            # Dunno if doing that is ok for multi-gpu etc
            global_iter = kwargs['global_step']
            capacity = torch.min(self.max_capacity, self.max_capacity * torch.tensor(global_iter) / self.iterations_c)
            self.current_c = capacity.detach()
            kld_loss = (kl_divergence_mu0_var1(mu, logvar) - capacity).abs()
        
        return kld_loss.sum() * self.w_kld, loss_per_node

    def loss_function(self, **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu, logvar = kwargs['posterior_mu'], kwargs['posterior_logvar']
        global_step = kwargs['global_step']
        bs = self.batch_size
        current_epoch = kwargs['current_epoch']
        
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0

        if self.loss_type == c.BIN_CROSS_ENT_LOSS:
            output_losses[c.RECON] = (F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs) * self.w_recon
        
        if self.loss_type == c.MSE_LOSS:
            output_losses[c.RECON] = (F.mse_loss(x_recon, x_true, reduction='sum') / bs) * self.w_recon

        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        if current_epoch > self.kl_warmup_epochs:
            output_losses[c.KLD_LOSS], kld_loss_per_layer = self._kld_loss_fn(mu, logvar, global_step=global_step)
            output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]
            output_losses.update(kld_loss_per_layer)
        else:
            output_losses[c.KLD_LOSS] = torch.Tensor([0.0]).to(device=x_recon.device)
        
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

        return output_losses

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        
        if self.loss_type == c.BIN_CROSS_ENT_LOSS:
            return torch.sigmoid(self.decoder(z))
        else:
                return self.decoder(z)

    def forward(self, x_true, **kwargs):
        
        fwd_pass_results = dict()

        mu, logvar = self.encode(x=x_true, **kwargs)
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z=z, **kwargs)
        
        fwd_pass_results.update({
            'x_recon': x_recon,
            'posterior_mu' : mu,
            'posterior_logvar': logvar,
            'z': z
        })
        
        return fwd_pass_results

    def sample(self, num_samples, current_device):
        
        z = torch.randn(num_samples, self.z_dim, device=current_device)
        return self.decode(z)