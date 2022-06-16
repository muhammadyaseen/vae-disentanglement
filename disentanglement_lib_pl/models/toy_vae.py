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

        # model
        self.encoder = self._init_toy_encoder()
        self.decoder = self._init_toy_decoder()

    def _kld_loss_fn(self, mu, logvar, **kwargs):
        if not self.controlled_capacity_increase:
            kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        else:
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

        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        #print(" in loss fn ", mu.shape)
        return output_losses

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x_true, **kwargs):
        
        fwd_pass_results = dict()

        mu, logvar = self.encode(x=x_true, **kwargs)
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z=z, **kwargs)
        
        fwd_pass_results.update({
            'x_recon': x_recon,
            'mu' : mu,
            'logvar': logvar,
            'z': z
        })
        
        return fwd_pass_results



    def sample(self, num_samples, current_device):
        
        z = torch.randn(num_samples, self.z_dim, device=current_device)
        return self.decode(z)