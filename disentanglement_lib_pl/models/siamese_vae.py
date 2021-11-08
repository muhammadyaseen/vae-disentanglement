import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize
from common import constants as c
from models.vae import VAE


class SiameseVAE(VAE):
    
    """
    Siamese VAE
    """

    def __init__(self, args):
        super().__init__(args)

        self.aux_decoder = None

    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z, **kwargs))

    def forward(self, x_true, x_aux, **kwargs):
        
        mu, logvar = self.encode(x=x_true)
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z=z)

        # aux recon
        mu_aux, logvar_aux = self.aux_encoder()
        z_aux = reparametrize(mu, logvar)
        x_aux = self.decode(z=z)

        return x_recon, mu, z, logvar

    def sample(self, num_samples, current_device):