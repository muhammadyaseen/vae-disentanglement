import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common import constants as c


class LatentToImage(nn.Module):
    """
    Given true latent factors z as Input, reconstruct the corresponding image X.
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

        # latent to image decoder
        decoder_name = args.decoder[0]
        decoder = getattr(decoders, decoder_name)

        # model
        self.decoder = decoder(self.z_dim, self.num_channels, self.image_size)

    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        global_step = kwargs['global_step']
        bs = self.batch_size
        current_epoch = kwargs['current_epoch']
        
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0

        if loss_type == 'cross_ent':
            output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        
        if loss_type == 'mse':
            output_losses[c.RECON] = F.mse_loss(x_recon, x_true, reduction='sum') / bs * self.w_recon

        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        return output_losses

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x_true, **kwargs):
        
        fwd_pass_results = dict()

        true_latents = kwargs['labels']
        x_recon = self.decode(z=true_latents, **kwargs)
        
        fwd_pass_results.update({
            'x_recon': x_recon,
            'z': true_latents
        })
        
        return fwd_pass_results
