from .base import *
from .vanilla_vae import *
from .beta_vae import *
from .factor_vae import *
from .vq_vae import *
from .betatc_vae import *
from .dip_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {
              'VQVAE':VQVAE,
              'DIPVAE':DIPVAE,
              'BetaVAE':BetaVAE,
              'BetaTCVAE':BetaTCVAE,
              'FactorVAE':FactorVAE,
              'VanillaVAE':VanillaVAE,
              }
