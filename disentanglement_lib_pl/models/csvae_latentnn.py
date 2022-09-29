import pickle
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from architectures import encoders, decoders
from common.ops import kl_divergence_diag_mu_var_per_node, reparametrize, Flatten3D
from common import constants as c
from common import dag_utils
from common import utils
from common.special_modules import SimpleLatentNN, SupervisedRegulariser

class LatentNN_CSVAE(nn.Module):
    """
    Concept Structured VAEs where Prior and Posterior dists have been
    implemented using GNNs
    """

    def __init__(self, network_args, **kwargs):
        """
        adjacency_matrix: str or list . If str, it is interpreted as path to pickled list
        """
        super(LatentNN_CSVAE, self).__init__()

        self.dataset = network_args.dset_name
        self.loss_type = utils.get_loss_type_for_dataset(self.dataset)

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
        self.add_cov_loss = c.COVARIANCE_LOSS in network_args.loss_terms

        self.dept_adjacency_matrix = dag_utils.get_adj_mat_from_adj_list(self.adjacency_list)
        self.adjacency_matrix = dag_utils.get_adj_mat_from_adj_list(self.adjacency_list)
        self.np_A = dag_utils.adjust_adj_mat_for_prior(self.dept_adjacency_matrix)
        
        print("Posterior mat: ", self.dept_adjacency_matrix)
        print("Prior mat: ", self.np_A)
        
        # Model latents for which we do have labels / DAG connections 
        self.num_dept_nodes = len(self.dept_adjacency_matrix)
        # Model latents for which we don't have labels / DAG connections 
        self.num_indept_nodes = network_args.num_indept_nodes
                
        if self.num_indept_nodes > 0:
            self.adjacency_matrix = dag_utils.extend_adj_mat_with_indept_nodes(self.dept_adjacency_matrix, self.num_dept_nodes, self.num_indept_nodes)
            print(self.adjacency_matrix)

        # total nodes in GNN (z + v latents)
        self.num_nodes = self.num_dept_nodes + self.num_indept_nodes

        self.num_channels = network_args.in_channels
        self.image_size = network_args.image_size
        self.batch_size = network_args.batch_size
        self.z_dim = network_args.z_dim[0]
        self.l_dim = network_args.l_dim
        # 'dependent' feature dims.. not using this idea anymore
        self.d_dim = 0
        
        self.w_recon = network_args.w_recon
        self.w_kld = network_args.w_kld
        self.w_sup_reg = network_args.w_sup_reg
        self.w_cov_loss = network_args.w_cov_loss
        self.kl_warmup_epochs = network_args.kl_warmup_epochs
        self.use_loss_weights = network_args.use_loss_weights

        # KL capacity annealing stuff
        self.controlled_capacity_increase = network_args.controlled_capacity_increase
        self.max_capacity_iters = torch.tensor(network_args.iterations_c, dtype=torch.float)
        self.max_capacity = torch.tensor(network_args.max_capacity, dtype=torch.float)
        self.current_capacity = torch.tensor(0.0)

        # DAG - 0th element is list of first level nodels, last element is list of leaves / terminal nodes
        self.dag_layer_nodes = dag_utils.get_dag_layers(self.adjacency_list)        
        
        # encoder and decoder
        self.init_node_feat_dim = 10
        # this is the size of final output layer of EncoderCNN
        msenc_feature_dim = self.num_nodes * self.init_node_feat_dim
        self.encoder_cnn = encoders.simple_conv64.SimpleConv64CommAss(msenc_feature_dim, self.num_channels, self.image_size,
                                                                        for_gnn=False, num_nodes=self.num_nodes, node_feat_dim=self.init_node_feat_dim)

        # uses multi scale features to init node feats
        # Q(Z|X,A) and Q(V|X)
        self.latent_nns = self.build_latent_networks(interm_dim=10)

        # converts exogenous vars to prior latents 
        # P(Z|epsilon, A)
        # TODO: revisit after introduction of indept nodes
        self.prior_type = "gt_based_fixed"
        self.prior_gnn = None
        
        #self.gt_based_prior = type(self.prior_gnn) == GroundTruthBasedLearnablePrior
        print("prior: ", self.prior_type)

        in_node_feat_dim, out_node_feat_dim = (self.z_dim + self.d_dim) * 2, (self.z_dim + self.d_dim) * 2
        # takes in encoded features and spits out recons
        # we do // 2 because we split the output features into mu and logvar 
        # but we only need mu-dim components for recon
        decoder_input_dim = self.num_nodes * (out_node_feat_dim // 2)
        self.decoder_dcnn = decoders.simple_conv64.SimpleConv64CommAss(decoder_input_dim, self.num_channels, self.image_size)
        
        # Supervised reg
        self.latents_classifier = self._init_classification_network() if self.add_classification_loss else None
        self.flatten_node_features = Flatten3D()
        
        print("GNNBasedConceptStructuredVAE Model Initialized")

    def forward(self, x_true, **kwargs):

        fwd_pass_results = dict()

        # Encode - extract multiscale feats and then pass thru posterior GNN 
        posterior_mu, posterior_logvar, posterior_z = self.encode(x_true, **kwargs)
        
        if self.add_classification_loss:
            
            if self.num_indept_nodes > 0:
                latents_predicted = self.latents_classifier(posterior_z[:,:self.num_dept_nodes, :])
            else:
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
        prior_mu, prior_logvar = self.prior_to_latents_prediction(x_true.device, gt_labels=kwargs['labels'])
        fwd_pass_results.update({
            "x_recon": x_recon,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "posterior_mu": posterior_mu, 
            "posterior_logvar": posterior_logvar
        })
        
        return fwd_pass_results

    def build_latent_networks(self, interm_dim):
        """
        iterm_dim: dim for internal hidden layer
        """
        num_neighbours = self.dept_adjacency_matrix.sum(dim=-1, keepdims=True)
        
        return nn.ModuleList([
            # we do -1 here because in the adj mat we have self connections
            # but we only want to have 'true' parents as input so nodes with 1 
            # parent (self conn) won't have any parent inputs
            SimpleLatentNN(int(num_neighbours[i]) - 1, self.init_node_feat_dim, interm_dim, self.z_dim) for i in range(self.num_nodes)
        ])

    def _init_classification_network(self):

        return SupervisedRegulariser(self.num_dept_nodes, self.z_dim, self.w_sup_reg, self.node_labels)

    def _classification_loss(self, predicted_latents, true_latents):
        
        total_clf_loss, per_node_loss = self.latents_classifier.loss(predicted_latents, true_latents)
        
        return total_clf_loss, per_node_loss

    def _gnn_cs_vae_kld_loss_fn(self, prior_mu, prior_logvar, posterior_mu, posterior_logvar, **kwargs):
        """
        All the arguments have shape (batch, num_nodes, node_feat_dim)
        """
        loss_per_node = dict()
        
        # KL(Q(Z|X,A)||P(Z|A))
        kld_per_node = kl_divergence_diag_mu_var_per_node(posterior_mu, posterior_logvar, 
                                                          prior_mu, prior_logvar)

        for node_idx, node_kld_loss in enumerate(kld_per_node):
            loss_per_node[f'KLD_z_{node_idx}'] = node_kld_loss.detach()
        
        if self.controlled_capacity_increase:
            global_iter = kwargs['global_step']
            self.current_capacity = torch.min(self.max_capacity, self.max_capacity * torch.tensor(global_iter) / self.max_capacity_iters)
            kld_loss = (kld_per_node.sum() - self.current_capacity).abs()
        else:
            kld_loss = kld_per_node.sum()
        
        return  kld_loss * self.w_kld, loss_per_node

    def _gnn_cs_vae_kld_loss_fn_dep_and_indept(self, prior_mu, prior_logvar, posterior_mu, posterior_logvar, **kwargs):
        """
        All the arguments have shape (batch, num_nodes, node_feat_dim)
        prior_mu, prior_logvar: Contain learnable prior dist params for p(Z|A) only
        posterior_mu, posterior_logvar: Contain posterior params for q(Z|A,X) and q(V|X) both
        so they have to be appropriately sliced and passed into correct KLD calc function 
        """
        loss_per_node = dict()
        
        if self.num_indept_nodes > 0:
            
            # KL(Q(Z|X,A)||P(Z|A))
            kld_per_node_dep = kl_divergence_diag_mu_var_per_node(posterior_mu[:, :self.num_dept_nodes, :], 
                                                          posterior_logvar[:, :self.num_dept_nodes, :], 
                                                          prior_mu, prior_logvar)
            
            # KL(Q(V|X)||P(V))
            kld_per_node_indep = kl_divergence_diag_mu_var_per_node(posterior_mu[:, self.num_dept_nodes:, :], 
                                                          posterior_logvar[:, self.num_dept_nodes:, :],
                                                           torch.Tensor([0.]).to(prior_mu.device),
                                                           torch.Tensor([0.]).to(prior_mu.device))

            # concat along node dim
            kld_per_node = torch.cat([kld_per_node_dep, kld_per_node_indep], dim=0)
        
        else:
            # we don't have indept nodes
            # KL(Q(Z|X,A)||P(Z|A))
            kld_per_node = kl_divergence_diag_mu_var_per_node(posterior_mu, posterior_logvar, 
                                                          prior_mu, prior_logvar)
        
        for node_idx, node_kld_loss in enumerate(kld_per_node):
            loss_per_node[f'KLD_z_{node_idx}'] = node_kld_loss.detach()
        
        if self.controlled_capacity_increase:
            global_iter = kwargs['global_step']
            self.current_capacity = torch.min(self.max_capacity, self.max_capacity * torch.tensor(global_iter) / self.max_capacity_iters)
            kld_loss = (kld_per_node.sum() - self.current_capacity).abs()
        else:
            kld_loss = kld_per_node.sum()
        
        return  kld_loss * self.w_kld, loss_per_node

    def covariance_loss(self, true_latents, node_activations):
        """
        true_latents: (B, V)
        node_activations: (B, V, 1)
        """
        
        # TODO: should it be computed over \mus or z's ??
        # if we use gt label as \mu for prior, then it makes sense to use mu here, right?

        learned_mus = node_activations.squeeze(2)
        batch_mean = learned_mus.T.mean(1, keepdims=True)      
        learned_mus_centered = learned_mus.T - batch_mean
        learned_cov = torch.matmul(learned_mus_centered, learned_mus_centered.T) / self.batch_size
        
        gt_mus = true_latents.T.mean(1, keepdims=True)
        gt_mus_centered = true_latents.T - gt_mus
        gt_cov = torch.matmul(gt_mus_centered, gt_mus_centered.T) / self.batch_size

        cov_loss = (learned_cov - gt_cov).pow(2).sum()
        
        return cov_loss

    def loss_function(self, **kwargs):
        
        output_losses = dict()

        x_recon, x_true = kwargs['x_recon'], kwargs['x_true']
        
        prior_mu = kwargs['prior_mu']
        prior_logvar = kwargs['prior_logvar']

        posterior_mu = kwargs['posterior_mu']
        posterior_logvar = kwargs['posterior_logvar']

        global_step, current_epoch, max_epochs = kwargs['global_step'], kwargs['current_epoch'], kwargs['max_epochs']
        
        if self.use_loss_weights:
            self.w_recon, self.w_kld, self.w_sup_reg = self._get_loss_term_weights(global_step, current_epoch, max_epochs)
        
        output_losses['output_aux'] = (self.w_recon, self.w_kld, self.w_sup_reg, self.w_cov_loss)
       
        # initialize the loss of this batch with zero.
        output_losses[c.TOTAL_LOSS] = 0
        
        #===== Calculating ELBO components
        # 1. REconstruction loss
        
        if self.loss_type == 'cross_ent':
            output_losses[c.RECON] = (F.binary_cross_entropy(x_recon, x_true, reduction='sum') / self.batch_size) * self.w_recon
        
        if self.loss_type == 'mse':
            output_losses[c.RECON] = (F.mse_loss(x_recon, x_true, reduction='sum') / self.batch_size) * self.w_recon
               
        output_losses[c.TOTAL_LOSS] += output_losses[c.RECON]

        # 2. KL-div loss 
        #-------------------------
        # KLD for our dag network 
        #-------------------------
        # Since we can have arbitrary number of layers, it won't take a fixed form      
        if current_epoch > self.kl_warmup_epochs:
            output_losses[c.KLD_LOSS], kld_loss_per_layer = self._gnn_cs_vae_kld_loss_fn_dep_and_indept(prior_mu, prior_logvar, posterior_mu, posterior_logvar, global_step=global_step)
            output_losses[c.TOTAL_LOSS] += output_losses[c.KLD_LOSS]
            output_losses.update(kld_loss_per_layer)
        else:
            output_losses[c.KLD_LOSS] = torch.Tensor([0.0]).to(device=x_recon.device)
        
        # 3. Auxiliary classification loss
        true_latents = kwargs['true_latents']
        
        if self.add_classification_loss:
            
            predicted_latents = kwargs['latents_predicted']    
            output_losses[c.AUX_CLASSIFICATION], clf_loss_per_layer = self._classification_loss(predicted_latents, true_latents)
            output_losses[c.TOTAL_LOSS] += output_losses[c.AUX_CLASSIFICATION]
            output_losses.update(clf_loss_per_layer)

        if self.add_cov_loss:
            output_losses[c.COVARIANCE_LOSS] = self.covariance_loss(true_latents, posterior_mu) * self.w_cov_loss
            output_losses[c.TOTAL_LOSS] += output_losses[c.COVARIANCE_LOSS]

        # detach all losses except for the full loss
        
        for loss_type in output_losses.keys():
            if loss_type == c.LOSS or loss_type == 'output_aux':
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
        posterior_mu, posterior_logvar, posterior_z = self.encoder_gnn(multi_scale_features)
        #posterior_z = reparametrize(posterior_mu, posterior_logvar)

        return posterior_mu, posterior_logvar, posterior_z

    # NOTE: This isn't really a GNN -- we keep this name to not break code that 
    # relies on the member name being encoder_gnn e.g. traversal and intervention code
    def encoder_gnn(self, image_features):
        """
        image_features: has shape (B, dim_init_node_feats)
        For successful chunk `dim_init_node_feats` should be a multiple of num_nodes
        """
        # chunk image features into num_node parts -- 1 for each node
        # so after chunk we should get V chunks of shape (B, latent_dim / V)
        #print("image_features.size ", image_features.size())
        node_init_feats = image_features.chunk(self.num_nodes, dim=1)
        num_neighbours = self.dept_adjacency_matrix.sum(dim=-1, keepdims=True)
        zs, mus, logvars = [], [], []

        for node_idx in range(self.num_nodes):
            
            # (A - I) because we have self connections that we don't want to count as parents
            if num_neighbours[node_idx] - 1 != 0:
                # if we have parents for this node we need to pass those as inputs
                parent_indices = (self.dept_adjacency_matrix.numpy() - np.eye(self.num_nodes))[node_idx].nonzero()[0]
                #print("node ", node_idx, " parent_indices ", parent_indices)
                parents_feats = torch.cat([zs[parent_idx] for parent_idx in parent_indices], dim=1) 
                image_and_parents_feats = torch.cat([node_init_feats[node_idx], parents_feats], dim=1)
                mu, logvar, z = self.latent_nns[node_idx](image_and_parents_feats)
            else:
                # if the node is a top-level node we'll only have image features
                mu, logvar, z = self.latent_nns[node_idx](node_init_feats[node_idx])
            
            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        # concat all z's -- this will be input of our decoder cnn
        # we had (B,1), after concat we get (B,V), after unsqueeze we get (B,V,1) for all
        zs = torch.cat(zs, dim=1).unsqueeze(2)
        mus = torch.cat(mus, dim=1).unsqueeze(2)
        logvars = torch.cat(logvars, dim=1).unsqueeze(2)
        
        #print(zs.shape, mus.shape, logvars.shape)

        # all should now have shape (B, V, 1)
        return mus, logvars, zs

    def decode(self, z, **kwargs):

        if self.loss_type == 'cross_ent':
            return torch.sigmoid(self.decoder_dcnn(z))
        else:
            return self.decoder_dcnn(z)

    def sample(self, num_samples, current_device):

        #===== Generative part
        # Top down until X

        exogen_vars_sample = torch.randn(size=(num_samples, self.num_dept_nodes, self.encoder_cnn.out_feature_dim),
                                         device=current_device)
        # TODO: update for gt based prior
        prior_mu, prior_logvar = self.prior_gnn(exogen_vars_sample)
        prior_z = reparametrize(prior_mu, prior_logvar)
        #print(prior_z.shape)
        if self.num_indept_nodes > 0:
            indept_sample = torch.normal(0, 1, size=(num_samples, self.num_indept_nodes, self.z_dim), device=current_device)
            #print(indept_sample.shape)
            # this gives the shape (b, Z+V, feat_dim)
            prior_z = torch.cat([prior_z, indept_sample], dim=1)

        # Need to reshape most likely before passing
        prior_z = self.flatten_node_features(prior_z)
        x_sampled = self.decode(prior_z)
        return x_sampled

    def prior_to_latents_prediction(self, current_device, gt_labels=None):
        
        prior_mu, prior_logvar = self.gt_based_fixed_prior(gt_labels, current_device)
        
        return prior_mu, prior_logvar

    def gt_based_fixed_prior(self, gt_labels, current_device):

        # gt_labels are of size (B, V) and we need (B,V,1) --
        # assuming 1-dim node feats

        mus = gt_labels.unsqueeze(2).to(current_device)
        logvar = torch.zeros(size=mus.size(), device=current_device) # fixed at var = 1 for now

        return mus, logvar
