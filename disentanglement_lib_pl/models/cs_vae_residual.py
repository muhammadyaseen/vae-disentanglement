import torch

from models.cs_vae import ConceptStructuredVAE
from common.ops import kl_divergence_mu0_var1, kl_divergence_diag_mu_var, reparametrize

class CSVAE_ResidualDistParameterization(ConceptStructuredVAE):
    """
    Implements Concept Structured VAE but uses a residual parameterization of the 
    Approximate Vriational Posterior distribution as descrined in Sec 3.2 of
    NVAE: A Deep Hierarchical VAE (https://github.com/NVlabs/NVAE) 
    """

    def __init__(self, network_args, **kwargs):
        
        print("init CSVAE RDP")
        super(CSVAE_ResidualDistParameterization, self).__init__(network_args, **kwargs)
        self.residual_dist_param = True
    

    def _cs_vae_kld_loss_fn(self, bu_net_outs, td_net_outs):
        
        dist_params = zip(reversed(bu_net_outs), [] + td_net_outs)

        loss_per_layer = dict()
        kld_loss = 0.0 
        L = len(self.top_down_networks) + 1

        for l, (bu_param, td_param) in enumerate(dist_params):
            
            if l == 0: # for z_L
                layer_loss = kl_divergence_mu0_var1(bu_param['mu_q_hat'], bu_param['sigma_q_hat'])
            else: # for all other z's i.e z_1,...,z_{L-1}
                layer_loss = kl_divergence_diag_mu_var(td_param['mu_q'], td_param['sigma_q'], 
                                          td_param['mu_p'], td_param['sigma_p'])
            
            loss_per_layer[f'KLD_z_{L - l}'] = layer_loss.detach()
            
            kld_loss += layer_loss
        
        return kld_loss * self.w_kld, loss_per_layer
    
    def _top_down_pass(self, bu_net_outs, mode='inference', **kwargs):
        """
        mode: 'sample' OR 'inference'
        When in 'inference' mode, should pass 'current_device' and 'num_sampples' as kwargs
        """
        assert mode in ['sample', 'inference']
        current_device = kwargs.get('current_device')
        #-----------------------------------------------------
        # TOP DOWN pass, goes from z_L, ..., z_1, X
        #-----------------------------------------------------
        # We reverse the outputs because BU pass gives us outputs in the order X, z_1, z_2, ... z_L
        bu_net_outs = list(reversed(bu_net_outs))
        #print(bu_net_outs)
        td_net_outs = []
        # First z i.e z_L is sampled like this because mu_q_hat_L = mu_q_hat and sigma_q_hat_L = sigma_q_hat
        if mode == 'inference':
            z = reparametrize(bu_net_outs[0]['mu_q_hat'], bu_net_outs[0]['sigma_q_hat'])
            interm_output = {
                    'mu_p':     torch.zeros(z.shape, device=current_device),
                    'sigma_p':  torch.zeros(z.shape, device=current_device), # this is log_var, hence zero (e^0 = 1)
                    'mu_q':     bu_net_outs[0]['mu_q_hat'],
                    'sigma_q':  bu_net_outs[0]['sigma_q_hat'],
                    'z': z 
                }
            td_net_outs.append(interm_output)
        
        if mode == 'sample':
            top_layer_dim = self.root_dim if len(self.dag_layer_nodes[0]) == 1 else len(self.dag_layer_nodes[0]) 
            z = torch.randn(kwargs['num_samples'], top_layer_dim, device=current_device)
            interm_output = {'mu_p': None, 'sigma_p': None, 'z': z }
            td_net_outs.append(interm_output)

        # Remaining z's i.e z_{L-1},..., z_2, z_1 are sampled like this
        for L, td_net in enumerate(self.top_down_networks):
            
            mu_p_L, sigma_p_L = td_net(z, current_device=current_device)
            
            if mode == 'inference':
            
                # Now we have to calc {mu|sigma}_q_L given {mu|sigma}_q_L_hat and {mu|sigma}_p_L
                
                mu_q_L = bu_net_outs[L+1]['mu_q_hat'] + mu_p_L
                sigma_q_L = bu_net_outs[L+1]['sigma_q_hat'] + sigma_p_L
                
                # sample for current layer
                z = reparametrize(mu_q_L, sigma_q_L)

                interm_output = {'mu_p': mu_p_L, 'sigma_p': sigma_p_L, 
                                 'mu_q': mu_q_L, 'sigma_q': sigma_q_L, 'z': z }
                td_net_outs.append(interm_output)

            if mode == 'sample':
                z = reparametrize(mu_p_L, sigma_p_L) 
                interm_output = {'mu_p': mu_p_L, 'sigma_p': sigma_p_L, 'z': z}
                td_net_outs.append(interm_output)  

        return td_net_outs
