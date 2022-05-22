from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize
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
        
        # encoder and decoder
        encoder_name = network_args.encoder[0]
        decoder_name = network_args.decoder[0]
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # DAG
        self.dag_layer_nodes = dag_utils.get_dag_layers(adjacency_matrix)
        print(self.dag_layer_nodes)
        # model
        self.encoder = encoder(self.z_dim, self.num_channels, self.image_size)
        self.dag_network = self._init_dag_network(adjacency_matrix)
        self.decoder = decoder(len(self.dag_layer_nodes[-1]), self.num_channels, self.image_size)
        
    def _init_dag_network(self, adjacency_matrix):

        dag_layers = []
        
        for L in range(len(self.dag_layer_nodes) - 1):
            dag_layers.append(
                DAGInteractionLayer(parents_list = self.dag_layer_nodes[L],
                                    children_list = self.dag_layer_nodes[L+1], 
                                    adjacency_matrix = adjacency_matrix, 
                                    interm_unit_dim = self.interm_unit_dim, 
                                    bias=True
                )
            )

        return nn.Sequential(*dag_layers)    


    def forward(self, x_true, **kwargs):
        pass
   
    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        mu, logvar = kwargs['mu'], kwargs['logvar']
        global_step = kwargs['global_step']
        bs = self.batch_size
        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0
    
        # detach all losses except for the full loss
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS:
                continue
            else:
                output_losses[loss_type] = output_losses[loss_type].detach()

        return output_losses
    
    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z, **kwargs))

    def _kld_loss_fn(self, mu, logvar, **kwargs):

        kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        return kld_loss
