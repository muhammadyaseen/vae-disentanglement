import torch.nn as nn


class BaseImageEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()

        self._latent_dim = latent_dim
        self._num_channels = num_channels
        self._image_size = image_size

    def forward(self, *input):
        raise NotImplementedError

    def latent_dim(self):
        return self._latent_dim

    def num_channels(self):
        return self._num_channels

    def image_size(self):
        return self._image_size

class BaseEncoder(nn.Module):
    def __init__(self, latent_dim, in_dim):
        super().__init__()

        self._latent_dim = latent_dim
        self._in_dim = in_dim

    def forward(self, *input):
        raise NotImplementedError

    def latent_dim(self):
        return self._latent_dim

    def num_in_dim(self):
        return self._in_dim

