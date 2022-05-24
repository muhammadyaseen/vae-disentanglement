import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Unsqueeze3D, Reshape
from common.utils import init_layers


class SimpleConv64(BaseImageEncoder):
    
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(latent_dim, 256, 1, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 3, 1)
        )
        # output shape = bs x 3 x 64 x 64

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)

class SimpleConv64CommAss(BaseImageEncoder):
    
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(latent_dim, 256, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 64 * 4 * 4, 1, 1),
            nn.ReLU(True),
            Reshape([64, 4, 4]),
            # Contrary to padding=same in TF impl., padding = 1 has been added to all ConvTranspose2d layers because 
            # otherwise output shape isn't correct. This is because TF and PyTorch 
            # handle 'same' padding in different ways
            nn.ConvTranspose2d(64, 64, 4, 2, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1)
        )
        # output shape = bs x 3 x 64 x 64

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)