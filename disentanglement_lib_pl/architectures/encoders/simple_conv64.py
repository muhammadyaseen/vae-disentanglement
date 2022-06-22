import torch
import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D
from common.utils import init_layers


class SimpleConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(256, latent_dim, bias=True)
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)

class SimpleGaussianConv64(SimpleConv64):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar

class SimpleConv64CommAss(BaseImageEncoder):
    """
    Encoder as used in 'Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations'
    """
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2), # B, 32, 31 x 31
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2), # B, 32, 14 x 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2), # B, 64, 6 x 6
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2), # B, 64, 2 x 2
            nn.ReLU(True),
            Flatten3D(), # B, 64*2*2
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)

class SimpleGaussianConv64CommAss(SimpleConv64CommAss):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar

class SmallEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels, image_size):
        
        super().__init__()
        
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
                        nn.Conv2d(1, 5, 2, 1, 0), # B,5,2,2
                        nn.ReLU(True),
                        Flatten3D(), # B,5x2x2
                        nn.Linear(20, 10),
                        nn.Tanh(),
                        nn.Linear(10, self.latent_dim * 2)
                    )

    def forward(self, x):

        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        return mu, logvar

class SmallFCEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels, image_size):
        
        super().__init__()
        
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
                        Flatten3D(),
                        nn.Linear(9, 20),
                        nn.Tanh(),
                        nn.Linear(20, 10),
                        nn.Tanh(),
                        nn.Linear(10, self.latent_dim * 2)
        )

    def forward(self, x):

        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        return mu, logvar

class MultiScaleEncoder(nn.Module):
    """
    Extracts features at multiple scales.
    To be used in conjunction wth GNN Encoders in special_modules
    """
    def __init__(self, feature_dim, num_channels, image_size):
        super().__init__()
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.feature_dim = feature_dim

        # in / out feature maps at each scale
        self.scale_3_in, self.scale_3_out = num_channels, 32
        self.scale_2_in, self.scale_2_out = 32, 64
        self.scale_1_in, self.scale_1_out = 64, 128

        # coarsest scale - outputs maps of shape B, 
        self.scale_3 = nn.Sequential(
            nn.Conv2d(self.scale_3_in, 32, 4, 2), # B, 32, 31 x 31
            nn.ReLU(True)
        )
        self.scale_3_feats = nn.Sequential(
            Flatten3D(),
            nn.Linear()
        )
        # mid scale - outputs maps of shape B, 
        self.scale_2 = nn.Sequential(
            nn.Conv2d(self.scale_2_in, 32, 4, 2), # B, 32, 14 x 14
            nn.ReLU(True)
        )
        self.scale_2_feats = nn.Sequential(
            Flatten3D(),
            nn.Linear()
        )
        # finest scale - outs maps of shape B,
        self.scale_1 = nn.Sequential(
            nn.Conv2d(self.scale_1_in, 64, 4, 2), # B, 64, 6 x 6
            nn.ReLU(True)
        )
        self.scale_1_feats = nn.Sequential(
            Flatten3D(),
            nn.Linear()
        )

    def forward(self, x):
        
        scale_3_x = self.scale_3(x)
        scale_3_feats = self.scale_3_feats(scale_3_x)

        scale_2_x = self.scale_2(scale_3_x)
        scale_2_feats = self.scale_2_feats(scale_2_x)

        scale_1_x = self.scale_1(scale_2_x)
        scale_1_feats = self.scale_1_feats(scale_1_x)

        multi_scale_feats = torch.cat([scale_3_feats, scale_2_feats, scale_1_feats])

        # reshape like this so that they can be associated with each latent node
        return multi_scale_feats

