import torch
from torch.autograd import Variable
import torch.nn.init as init

from base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

"""
 Some util funcs used in both methods
"""

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_Vanilla(BaseVAE):

    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 loss_type="H",
                 latent_dim=10,
                 in_channels=3,
                 beta=4,
                 latent_dist_type="bernoulli"):

        super(BetaVAE_Vanilla, self).__init__()
        self.loss_type = loss_type
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.beta = beta
        self.latent_dist_type = latent_dist_type

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32,
                      kernel_size=4, stride=2, padding=1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            # I think we have latent_dim * 2 because we use it for \mu and \sigma at
            # the same time
            nn.Linear(256, latent_dim * 2),             # B, latent_dim * 2
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


    def forward(self, x_input):

        dist_params = self.encode(x_input)
        mu = dist_params[:, :self.latent_dim]
        logvar = dist_params[:, self.latent_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, x_input, mu, logvar

    def loss_function(self, *args, **kwargs) -> dict:

        self.num_iter += 1
        recons, input, mu, log_var = args[0], args[1], args[2], args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        if self.latent_dist_type == "bernoulli":
            recons_loss = F.binary_cross_entropy_with_logits(recons, input, reduction='mean')
        elif self.latent_dist_type == "gaussian":
            recons_loss = F.mse_loss(recons, input, reduction='mean')
        else:
            recons_loss = None
            Exception("Unknown latent dist type")

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == "H":
            # Loss type H (first beta mult. loss)
            loss = recons_loss + self.beta * kld_weight * kld_loss

        elif self.loss_type == "B":
            # Loss type B (capacitated loss)
            self.c_max = self.c_max.to(input.device)
            C = torch.clamp(self.c_max / self.c_stop_iter * self.num_iter, 0, self.c_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()

        else:
            loss = None
            Exception("Unknown latent dist type")

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss.detach(),
                'KLD': kld_loss.detach }

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param batch_size: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(batch_size, self.latent_dim)
        z = z.to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

"""
class BetaVAE_B_Vanilla(BetaVAE_H_Vanilla):
    # Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).

    def __init__(self, latent_dim=10, in_channels=1, c_max=25, c_max_iter=1e5):
        super(BetaVAE_B_Vanilla, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.c_max = c_max
        self.c_max_iter = c_max_iter

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, latent_dim * 2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), # B,  nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x_input):
        dist_params = self._encode(x_input)
        mu = dist_params[:, :self.z_dim]
        logvar = dist_params[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x_input.size())

        return x_recon, x_input, mu, logvar

    def loss_function(self, *args, **kwargs) -> dict:

        self.num_iter += 1
        recons, input, mu, log_var = args[0], args[1], args[2], args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # Loss type B https://arxiv.org/pdf/1804.03599.pdf (capacitated loss)
        self.c_max = self.c_max.to(input.device)
        C = torch.clamp(self.c_max / self.c_stop_iter * self.num_iter, 0, self.c_max.data[0])
        loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss.detach(),
                'KLD': kld_loss.detach }

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
"""