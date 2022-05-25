from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, kl_divergence_diag_mu_var, reparametrize
from common import constants as c
from common import dag_utils
from common.special_modules import DAGInteractionLayer

class ConceptStructuredVAE(nn.Module):
    """
    Concept Structured VAEs
    """

    def __init__(self, adjacency_matrix, network_args, **kwargs):
        """
        adjacency_matrix: str or list . If str, it is interpreted as path to pickled list

        """
        super(ConceptStructuredVAE, self).__init__()

        self.adjacency_matrix = adjacency_matrix
        # Fixed - For debugging
        self.adjacency_matrix = [(), (27,), (4,), (0,), (13,), (1,), (31,), (4,), (7,), (1,), 
           (17,), (27,), (27,), (27,), (24,), (17,), (14,), (13,), (12,), 
           (12,), (11,), (31,), (11,), (17,), (12,), (13,), (31,), (0,), 
           (13,), (13,), (1,), (13,), (27,)]

        self.node_labels = kwargs['node_labels'] if 'node_labels' in kwargs.keys() else None
        self.interm_unit_dim = network_args.interm_unit_dim
        self.z_dim = network_args.z_dim
        self.num_channels = network_args.num_channels
        self.image_size = network_args.image_size
        self.batch_size = network_args.batch_size
        self.root_dim = network_args.root_dim
        
        # encoder and decoder
        # Encoder is composed of BU nets, TD nets (aka DAG layers)
        #encoder_name = network_args.encoder[0]
        decoder_name = network_args.decoder[0]
        #encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # bottom up networks
        # The number of BU networks depends on the depth of DAG
        self.bottom_up_networks = self._init_bottom_up_networks()

        # DAG
        self.dag_layer_nodes = dag_utils.get_dag_layers(self.adjacency_matrix)
        
        # model
        #self.encoder = encoder(self.z_dim, self.num_channels, self.image_size)
        self.dag_network = self._init_dag_network_modules()
        nodes_in_last_dag_layer = len(self.dag_layer_nodes[-1])
        self.decoder = decoder(nodes_in_last_dag_layer, self.num_channels, self.image_size)
    
    def _init_bottom_up_networks(self):
        
        dag_levels = len(self.dag_layer_nodes)
        bu_nets = []

        for d in reversed(range(dag_levels)):
            
            # first BU net takes image X as input so it needs to have different architecture
            if d == dag_levels - 1:
                out_dim = len(self.dag_layer_nodes[d]) * 2
                bu_nets.append(
                    encoders.SimpleConv64(out_dim, self.num_channels, self.image_size)
                )
            
            # last BU layer corresponds to root, we don't represent root with 
            # a single unit so have to do special here
            elif d == 0:
                inp_dim = len(self.dag_layer_nodes[d+1]) * 2
                bu_nets.append(
                    encoders.SimpleFCNNEncoder(self.root_dim * 2, inp_dim, [inp_dim * 2])
                )
            
            # all the other layers in DAG
            else:
                inp_dim = len(self.dag_layer_nodes[d+1])
                out_dim = len(self.dag_layer_nodes[d]) * 2
                bu_nets.append(
                    encoders.SimpleFCNNEncoder(out_dim, inp_dim, [inp_dim * 2])
                )

        BU_nets = nn.ModuleList(bu_nets)
        
        return BU_nets

    def _init_dag_network_modules(self):

        dag_layers = []
        
        for L in range(len(self.dag_layer_nodes) - 1):
            dag_layers.append(
                DAGInteractionLayer(parents_list = self.dag_layer_nodes[L],
                                    children_list = self.dag_layer_nodes[L+1], 
                                    adjacency_matrix = self.adjacency_matrix, 
                                    interm_unit_dim = self.interm_unit_dim, 
                                    bias=True,
                                    parent_is_root = L == 0,
                                    root_dim=self.root_dim
                )
            )

        return nn.ModuleList(modules=dag_layers)    

    def _top_down_pass(self, bu_net_outs, **kwargs):
        
        #-----------------------------------------------------
        # TOP DOWN pass, goes from z_L, ..., z_1, X
        #-----------------------------------------------------
        # First z i.e z_L is sampled like this because mu_q_hat_L = mu_q_hat and sigma_q_hat_L = sigma_q_hat
        z = reparametrize(bu_net_outs[-1]['mu_q_hat'], bu_net_outs[-1]['sigma_q_hat'])
        td_net_outs = []
        for L, td_net in enumerate(self.dag_network):
            
            # Hope it doesn't mess up the compute graph
            mu_p_L, sigma_p_L = td_net(z)
            
            # Now we have to calc {mu|sigma}_q_L given {mu|sigma}_q_L_hat and {mu|sigma}_p_L
            prec_q_L_hat = sigma_p_L.exp().pow(-1)
            prec_p_L = sigma_p_L.exp().pow(-1)
            mu_q_L = (bu_net_outs[L]['mu_q_hat'] * prec_q_L_hat + mu_p_L * prec_p_L) / ( prec_q_L_hat + prec_p_L)
            # have to do this log because of how `reparametrize` is implemented
            sigma_q_L = (prec_q_L_hat + prec_p_L).pow(-1).log() 
            
            # sample for current layer
            z = reparametrize(mu_q_L, sigma_q_L)

            interm_output = {
                'mu_p': mu_p_L,
                'sigma_p': sigma_p_L,
                'mu_q': mu_q_L,
                'sigma_q': sigma_q_L,
                'z': z 
            }

            td_net_outs.append(interm_output)

        return td_net_outs
    
    def _bottom_up_pass(self, x_true, **kwargs):
        
        #-----------------------------------------------------
        # BOTTOM UP pass, goes from X to z_1 to z_2 to ... z_L
        #-----------------------------------------------------
        d_L = x_true
        bu_net_outs = []
        for L, bu_net in enumerate(self.bottom_up_networks):
            # Hope it doesn't mess up the compute graph
            d_L = bu_net(d_L)
            mu_L, sigma_L = torch.chunk(d_L, 2, dim=1)
            bu_net_outs.append({
                'mu_q_hat': mu_L,
                'sigma_q_hat': sigma_L
            })
        
        return bu_net_outs

    def forward(self, x_true, **kwargs):
        
        fwd_pass_results = dict()
        
        # Encode
        bu_net_outs, td_net_outs = self.encode(x_true, **kwargs)

        # Decode
        x_recon = self.decode(td_net_outs[-1]['z'], **kwargs)

        return fwd_pass_results.update({
            "x_recon": x_recon,
            "x_true" :  x_true,
            "td_net_params": td_net_outs,
            "bu_net_params": bu_net_outs
        })

    def _cs_vae_kld_loss_fn(self, bu_net_outs, td_net_outs):
        
        dist_params = zip(reversed(bu_net_outs), td_net_outs)
        print(dist_params)
        loss_per_layer = []
        for L, bu_param, td_param in enumerate(dist_params):
            
            if L == 0:
                loss = kl_divergence_mu0_var1(bu_param[L]['mu_q_hat'], bu_param[L]['sigma_q_hat'])
            else:
                loss = kl_divergence_diag_mu_var(td_param[L]['mu_q'], td_param[L]['sigma_q'], 
                                          td_param[L]['mu_p'], td_param[L]['sigma_p'])

            loss_per_layer.append(loss)

        print(loss_per_layer)
        return loss_per_layer

    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        td_net_outs = kwargs['td_net_params']
        bu_net_outs = kwargs['bu_net_params']
        global_step = kwargs['global_step']
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
        #-------------------------
        # KLD for our dag network 
        #-------------------------
        # Since we can have arbitrary number of layers, it won't take a fixed form      
        output_losses[c.KLD_LOSS] = self._cs_vae_kld_loss_fn(bu_net_outs, td_net_outs)
        
        output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]
        
        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        return output_losses
    
    def encode(self, x_true, **kwargs):
        #-------
        # Encode
        #-------
        #-----------------------------------------------------
        # BOTTOM UP pass, goes from X to z_1 to z_2 to ... z_L
        #-----------------------------------------------------
        bu_net_outs = self._bottom_up_pass(x_true, **kwargs)

        #-----------------------------------------------------
        # TOP DOWN pass, goes from z_L, ..., z_1, X
        #-----------------------------------------------------
        # First z i.e z_L is sampled like this because mu_q_hat_L = mu_q_hat and sigma_q_hat_L = sigma_q_hat
        td_net_outs = self._top_down_pass(bu_net_outs, **kwargs)
        
        # Technically, this is where the encoding process ends as we now have all the latents
        
        return bu_net_outs, td_net_outs

    def decode(self, z, **kwargs):

        # TODO: have to implement skips here?
        # Final recon, using z_1 as input i.e. final latent        

        return torch.sigmoid(self.decoder(z, **kwargs))

