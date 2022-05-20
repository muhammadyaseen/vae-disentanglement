from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize
from common import constants as c


class ConceptStructuredVAE(nn.Module):
    """
    Concept Structured VAEs
    """

    def __init__(self, args):
        
        super(ConceptStructuredVAE, self).__init__()

    def forward(self, x_true, **kwargs):
        pass
   
    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu, logvar = kwargs['mu'], kwargs['logvar']
        global_step = kwargs['global_step']
        bs = self.batch_size
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0
    
        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        return output_losses
    
    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z, **kwargs))

    def _kld_loss_fn(self, mu, logvar, **kwargs):

        kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        return kld_loss
