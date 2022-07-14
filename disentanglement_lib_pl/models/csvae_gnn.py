import pickle
import torch
from torch import nn
import torch.nn.functional as F

from architectures.encoders.simple_conv64 import MultiScaleEncoder
from architectures.decoders.simple_conv64 import SimpleConv64CommAss

from common.ops import kl_divergence_diag_mu_var_per_node, reparametrize, Flatten3D
from common import constants as c
from common import dag_utils
from common.special_modules import SimpleGNNLayer, SupervisedRegulariser

import pdb

class GNNBasedConceptStructuredVAE(nn.Module):
    """
    Concept Structured VAEs where Prior and Posterior dists have been
    implemented using GNNs
    """

    def __init__(self, network_args, **kwargs):
        """
        adjacency_matrix: str or list . If str, it is interpreted as path to pickled list
        """
        super(GNNBasedConceptStructuredVAE, self).__init__()

        if isinstance(network_args.adjacency_matrix, str):
            adj_mat_and_node_names = pickle.load(open(network_args.adjacency_matrix, 'rb'))
            self.adjacency_list = adj_mat_and_node_names['adj_mat']
            self.node_labels = adj_mat_and_node_names['node_labels']
        elif isinstance(network_args.adjacency_matrix, dict):
            self.adjacency_list = network_args.adjacency_matrix['adj_mat']
            self.node_labels = network_args.adjacency_matrix['node_labels']
        else:
            raise ValueError("Unsupported format for adjacency_matrix")

        self.add_classification_loss = c.AUX_CLASSIFICATION in network_args.loss_terms
        self.num_nodes = len(self.adjacency_list)
        self.adjacency_matrix = dag_utils.get_adj_mat_from_adj_list(self.adjacency_list)
        print(self.adjacency_matrix)
        
        self.num_channels = network_args.in_channels
        self.image_size = network_args.image_size
        self.batch_size = network_args.batch_size
        self.z_dim = network_args.z_dim[0]

        self.w_recon = network_args.w_recon
        self.w_kld = network_args.w_kld
        self.w_sup_reg = 0.05
        self.kl_warmup_epochs = network_args.kl_warmup_epochs
        
        # DAG - 0th element is list of first level nodels, last element is list of leaves / terminal nodes
        self.dag_layer_nodes = dag_utils.get_dag_layers(self.adjacency_list)        
        
        # encoder and decoder
        # multiscale encoder
        msenc_feature_dim = self.num_nodes * 3 # 3 feats per node
        # MSEnc outputs features of shape (batch, V, 3 * (out_feature_dim//num_nodes))
        # in current case it will be (b,V,9) i.e. each node gets a 9-dim feature vector

        self.encoder_cnn = MultiScaleEncoder(msenc_feature_dim, self.num_channels, self.num_nodes)
        
        # uses multi scale features to init node feats
        # Q(Z|X,A)
        self.encoder_gnn = self._init_gnn()

        # converts exogenous vars to prior latents 
        # P(Z|epsilon, A)
        self.prior_gnn = self._init_gnn()

        in_node_feat_dim, out_node_feat_dim = self.z_dim * 2, self.z_dim * 2
        # takes in encoded features and spits out recons
        # we do // 2 because we split the output features into mu and logvar 
        # but we only need mu-dim components for recon
        decoder_input_dim = self.num_nodes * (out_node_feat_dim // 2)
        self.decoder_dcnn = SimpleConv64CommAss(decoder_input_dim, self.num_channels, self.image_size)
        
        # Supervised reg
        self.latents_classifier = self._init_classification_network() if self.add_classification_loss else None
        self.flatten_node_features = Flatten3D()
        
        print("GNNBasedConceptStructuredVAE Model Initialized")

    def forward(self, x_true, **kwargs):

        #pdb.set_trace()

        fwd_pass_results = dict()

        # Encode - extract multiscale feats and then pass thru posterior GNN 
        posterior_mu, posterior_logvar, posterior_z = self.encode(x_true, **kwargs)
        
        if self.add_classification_loss:
            latents_predicted = self.latents_classifier(posterior_z)
            fwd_pass_results.update({
                "latents_predicted": latents_predicted
            })
        
        # Decode
        # reshape posterior_z into right format for decoder dcnn
        # posterior_z is (Batches, V, node_feat_dim) and we flatten it to (Batches, V * node_feat_dim)
        posterior_z = self.flatten_node_features(posterior_z)
        x_recon = self.decode(posterior_z)

        # Enforcing prior structure - sample from prior GNN and use it to predict latents
        prior_mu, prior_logvar = self.prior_to_latents_prediction(x_true.device)

        fwd_pass_results.update({
            "x_recon": x_recon,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "posterior_mu": posterior_mu, 
            "posterior_logvar": posterior_logvar
        })
        
        return fwd_pass_results

    def _init_gnn(self):
        
        in_node_feat_dim, out_node_feat_dim = self.z_dim * 2, self.z_dim * 2
        dag_levels = len(self.dag_layer_nodes)

        gnn_layers = [SimpleGNNLayer(self.encoder_cnn.out_feature_dim, out_node_feat_dim, self.adjacency_matrix, is_final_layer=False)]
        for d in range(1, dag_levels):    
            gnn_layers.append(SimpleGNNLayer(in_node_feat_dim, out_node_feat_dim, self.adjacency_matrix, is_final_layer= d == dag_levels - 1))
            
        return nn.Sequential(*gnn_layers)

    def _init_classification_network(self):

        return SupervisedRegulariser(self.num_nodes, self.z_dim, self.w_sup_reg, self.node_labels)

    def _classification_loss(self, predicted_latents, true_latents):
        
        total_clf_loss, per_node_loss = self.latents_classifier.loss(predicted_latents, true_latents)
        
        return total_clf_loss, per_node_loss

    def _gnn_cs_vae_kld_loss_fn(self, prior_mu, prior_logvar, posterior_mu, posterior_logvar):
        """
        All the arguments have shape (batch, num_nodes, node_feat_dim)
        """
        loss_per_node = dict()
        
        kld_per_node = kl_divergence_diag_mu_var_per_node(posterior_mu, posterior_logvar, 
                                                          prior_mu, prior_logvar)

        for node_idx, node_kld_loss in enumerate(kld_per_node):
            loss_per_node[f'KLD_z_{node_idx}'] = node_kld_loss.detach()
        
        return kld_per_node.sum() * self.w_kld, loss_per_node

    def loss_function(self, loss_type='cross_ent', **kwargs):
        
        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        
        prior_mu = kwargs['prior_mu']
        prior_logvar = kwargs['prior_logvar']

        posterior_mu = kwargs['posterior_mu']
        posterior_logvar = kwargs['posterior_logvar']

        global_step = kwargs['global_step']
        current_epoch = kwargs['current_epoch']
        
        num_is_nan = torch.isnan(x_recon).sum().item()
        if num_is_nan > 0:
            print("NaN detected during training...")
            print(kwargs)
            exit(1)

        output_losses = dict()
        
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0
        
        #===== Calculating ELBO components
        # 1. REconstruction loss
        
        if loss_type == 'cross_ent':
            output_losses[c.RECON] = (F.binary_cross_entropy(x_recon, x_true, reduction='sum') / self.batch_size) * self.w_recon
        
        if loss_type == 'mse':
            output_losses[c.RECON] = (F.mse_loss(x_recon, x_true, reduction='sum') / self.batch_size) * self.w_recon
               
        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        # 2. KL-div loss 
        #-------------------------
        # KLD for our dag network 
        #-------------------------
        # Since we can have arbitrary number of layers, it won't take a fixed form      
        if current_epoch > self.kl_warmup_epochs:
            output_losses[c.KLD_LOSS], kld_loss_per_layer = self._gnn_cs_vae_kld_loss_fn(prior_mu, prior_logvar, posterior_mu, posterior_logvar)
            output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]
            output_losses.update(kld_loss_per_layer)
        else:
            output_losses[c.KLD_LOSS] = torch.Tensor([0.0]).to(device=x_recon.device)
        
        # 3. Auxiliary classification loss
        if self.add_classification_loss:
            
            predicted_latents = kwargs['latents_predicted']
            true_latents = kwargs['true_latents']
            
            output_losses[c.AUX_CLASSIFICATION], clf_loss_per_layer = self._classification_loss(predicted_latents, true_latents)
            output_losses[c.TOTAL_LOSS] += output_losses[c.AUX_CLASSIFICATION]
            output_losses.update(clf_loss_per_layer)

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
        multi_scale_features = self.encoder_cnn(x_true)
        #print(multi_scale_features.shape)
        posterior_mu, posterior_logvar = self.encoder_gnn(multi_scale_features)
        posterior_z = reparametrize(posterior_mu, posterior_logvar)

        return posterior_mu, posterior_logvar, posterior_z

    def decode(self, z, **kwargs):

        return torch.sigmoid(self.decoder_dcnn(z))

    def sample(self, num_samples, current_device):

        #===== Generative part
        # Top down until X

        exogen_vars_sample = torch.randn(size=(num_samples, self.num_nodes, self.encoder_cnn.out_feature_dim),
                                         device=current_device)
        prior_mu, prior_logvar = self.prior_gnn(exogen_vars_sample)
        prior_z = reparametrize(prior_mu, prior_logvar)
        
        # Need to reshape most likely before passing
        prior_z = self.flatten_node_features(prior_z)
        x_sampled = self.decode(prior_z)
        return x_sampled

    def prior_to_latents_prediction(self, current_device):
         
        exogen_vars_sample = self._get_exogen_samples(current_device)
        prior_mu, prior_logvar = self.prior_gnn(exogen_vars_sample)

        return prior_mu, prior_logvar

    def _get_exogen_samples(self, current_device):

        exogen_samples_node = []
        upper, lower = 2, -2
        mus = torch.arange(lower, upper, (upper - lower) / self.num_nodes).to(current_device)
        for i in range(self.num_nodes):
            exogen_samples_node.append(
                mus[i] + torch.randn(size=(self.batch_size, 1, self.encoder_cnn.out_feature_dim), device=current_device)
            )

        return torch.cat(exogen_samples_node, dim=1)


