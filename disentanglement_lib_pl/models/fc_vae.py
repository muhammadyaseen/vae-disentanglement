import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize
from common import constants as c


class FC_VAE(nn.Module):
    
    """
    FCNN_VAE
    """

    def __init__(self, args):
        
        super().__init__()

        # Misc
        # self.name = args.name
        self.alg = args.alg
        self.loss_terms = args.loss_terms

        # Misc params related to data
        self.z_dim = args.z_dim
        self.l_dim = args.l_dim
        self.num_labels = args.num_labels
        self.in_dim = args.in_dim
        self.h_dims = args.h_dims
        self.batch_size = args.batch_size

        # Weight of recon loss
        self.w_recon = args.w_recon

        # Beta-vae args
        self.w_kld = args.w_kld
        self.controlled_capacity_increase = args.controlled_capacity_increase
        self.max_c = torch.tensor(args.max_c, dtype=torch.float)
        self.iterations_c = torch.tensor(args.iterations_c, dtype=torch.float)

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # model
        self.encoder = encoder(self.z_dim, self.in_dim, self.h_dims)
        self.decoder = decoder(self.z_dim, self.in_dim, self.h_dims)

    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, z, **kwargs):
        return self.decoder(z, **kwargs)

    def forward(self, x_true, **kwargs):
        
        mu, logvar = self.encode(x=x_true)
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z=z)

        return x_recon, mu, z, logvar

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(current_device)
        return self.decode(z)

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

    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu, logvar = kwargs['mu'], kwargs['logvar']

        bs = self.batch_size
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0

        if loss_type == 'cross_ent':
            output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        
        if loss_type == 'mse':
            output_losses[c.RECON] = F.mse_loss(x_recon, x_true, reduction='sum') / bs * self.w_recon

        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        output_losses[c.KLD_LOSS] = self._kld_loss_fn(mu, logvar)
        output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]

        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        #print(" in loss fn ", mu.shape)
        return output_losses
