import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, kl_divergence_diag_mu_var, reparametrize
from common import constants as c

class LadderVAE(nn.Module):

    def __init__(self, args):
        
        super(LadderVAE, self).__init__()
        
        self.alg = args.alg
        
        self.z_dim = args.z_dim
        print(args.z_dim)
        self.z1_dim = args.z_dim[1]
        self.z2_dim = args.z_dim[0]

        self.num_channels = args.in_channels
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.w_recon = args.w_recon
        self.w_kld = args.w_kld
        
        # bottom-up path
        self.nn_d_1 = encoders.SimpleConv64(latent_dim=self.z1_dim * 2, num_channels=self.num_channels, image_size=self.image_size)
        self.nn_d_2 = encoders.SimpleFCNNEncoder(latent_dim=self.z2_dim * 2, in_dim=self.z1_dim * 2, h_dims=[self.z2_dim * 2])       
     
        # top-down path
        self.nn_z_1 = decoders.SimpleFCNNDecoder(latent_dim=self.z1_dim * 2, in_dim=self.z2_dim, h_dims=[self.z1_dim * 2])
        self.nn_x = decoders.SimpleConv64(latent_dim=self.z1_dim, num_channels=self.num_channels, image_size=self.image_size)

    def _ladder_kld_loss_fn(self, mu_q_z1, logvar_q_z1, 
                            mu_q_z2, logvar_q_z2, 
                            mu_p_z1, logvar_p_z1):
        
        # KL(q(z_1 | z_2, x) || p(z_1 | z_2))
        kld_z1 = kl_divergence_diag_mu_var(mu_q_z1, logvar_q_z1, mu_p_z1, logvar_p_z1)
        
        # KL(q(z_2 | z_1) || p(z_2))
        kld_z2 = kl_divergence_mu0_var1(mu_q_z2, logvar_q_z2)

        return kld_z1, kld_z2

    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu_q_z1, logvar_q_z1 = kwargs['mu_q_1'], kwargs['log_var_q_1']
        mu_p_z1, logvar_p_z1 = kwargs['mu_p_1'], kwargs['log_var_p_1']
        mu_q_z2, logvar_q_z2 = kwargs['mu_q_2'], kwargs['log_var_q_2']
        
        bs = self.batch_size
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0
        
        #===== Calculating ELBO components
        # 1. REconstruction loss
        if loss_type == 'cross_ent':
            output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        
        if loss_type == 'mse':
            output_losses[c.RECON] = F.mse_loss(x_recon, x_true, reduction='sum') / bs * self.w_recon
               
        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        # 2. KL-div loss 

        kld_z1, kld_z2 = self._ladder_kld_loss_fn(mu_q_z1, logvar_q_z1, 
                                            mu_q_z2, logvar_q_z2, 
                                            mu_p_z1, logvar_p_z1)

        output_losses["kld_z1"] = kld_z1
        output_losses["kld_z2"] = kld_z2
        output_losses[c.KLD_LOSS] = kld_z1 + kld_z2
        
        output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]
        
        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()
        
        return output_losses
    
    def encode_to_z2(self, x_true, **kwargs):
        
        # step 1
        d_1 = self.nn_d_1(x_true) # use encoder for this nn
        
        #step 2
        mu_q_1_hat, log_var_q_1_hat = torch.chunk(d_1, 2, dim=1)
        #delta_log_var_1 = F.hardtanh(delta_log_var_1, -7., 2.)
        
        # step 3
        d_2  = self.nn_d_2(d_1)
        mu_q_2_hat, log_var_q_2_hat = torch.chunk(d_2, 2, dim=1)
        
        # top-down (uses bottom-up params)
        # step 4
        # use bottom up information and merge it with top down information
        mu_q_2 = mu_q_2_hat
        log_var_q_2 = log_var_q_2_hat
        
        z_2 = reparametrize(mu_q_2, log_var_q_2)
    
        dist_params_z2 = {
            "mu_q_1_hat": mu_q_1_hat,
            "log_var_q_1_hat": log_var_q_1_hat,
            "mu_q_2":       mu_q_2,
            "log_var_q_2":  log_var_q_2,
            "mu_q_2_hat": mu_q_2_hat,
            "log_var_q_2_hat": log_var_q_2_hat
        }

        return z_2, dist_params_z2

    def encode_to_z1(self, z2, dist_params_z2):
        
        # step 5
        h_1 = self.nn_z_1(z2)
        mu_p_1, log_var_p_1 = torch.chunk(h_1, 2, dim=1)
        
        # mu_q_1, log_var_q_1 i.e. var dist params for q(z_1|z_2,x) depend on params 
        # computed at bottom up level so we compute those here
        prec_q_1_hat = dist_params_z2['log_var_q_1_hat'].exp().pow(-1)
        prec_p_1 = log_var_p_1.exp().pow(-1)

        mu_q_1 = (dist_params_z2['mu_q_1_hat'] * prec_q_1_hat + mu_p_1 * prec_p_1) / ( prec_q_1_hat + prec_p_1)
        # have to do this log because of how `reparametrize` is implemented
        log_var_q_1 = (prec_q_1_hat + prec_p_1).pow(-1).log() 

        # step 6
        z_1 = reparametrize(mu_q_1, log_var_q_1)
    
        dist_params_z1 = {
            "mu_q_1":       mu_q_1,
            "log_var_q_1":  log_var_q_1,
            "mu_p_1":       mu_p_1, 
            "log_var_p_1":  log_var_p_1
        }

        return z_1, dist_params_z1

    def encode(self, x_true, **kwargs):
        
        # deterministic bottom-up pass - goes from x to z1 to z2, calculating params of dists
        
        # steps 1-4
        z_2, dist_params_z2 = self.encode_to_z2(x_true)
        # steps 5-6
        z_1, dist_params = self.encode_to_z1(z_2, dist_params_z2)
        
        # merge dist z2 and dist z1 params into one dict
        dist_params.update(dist_params_z2)

        return z_1, z_2, dist_params

    def decode(self, z, latent_layer='z1', **kwargs):
        
        assert latent_layer in ['z1', 'z2']

        if latent_layer == 'z2':
            z1, _ = self.encode_to_z1(z, kwargs)
            return torch.sigmoid(self.nn_x(z1))

        if latent_layer == 'z1':
            return torch.sigmoid(self.nn_x(z))
    
    def forward(self, x_true, **kwargs):
        
        # step 1-6 in the encode function
        z1, z2, fwd_pass_results = self.encode(x_true, **kwargs)
        
        # step 7 (get X given z_1)
        x_recon = self.decode(z1, latent_layer='z1')

        fwd_pass_results.update(
            { "x_recon" : x_recon,
              "x_true" :  x_true,
              'z1': z1,
              'z2': z2
            }
        )

        return fwd_pass_results

    def sample(self, num_samples, current_device):

        #===== Generative part

        # step 1
        z_2 = torch.randn(num_samples, self.z2_dim)
        z_2 = z_2.to(current_device)
        
        # step 2
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        
        # step 3
        z_1 = reparametrize(mu_1, log_var_1)
        
        # step 4
        x_sampled = self.decode(z_1)
        
        return x_sampled

    

