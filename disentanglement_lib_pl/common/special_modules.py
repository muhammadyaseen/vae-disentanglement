import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from common import dag_utils

LIMIT_A, LIMIT_B, EPSILON = -.1, 1.1, 1e-6

"""
These modules are used in DagInteractionLayer
"""

class InputToIntermediate(nn.Module):
    def __init__(self, input_to_intermediate_mask, in_features, intermediate_group_dim, out_groups):

        super().__init__()

        self.input_to_intermediate_mask = input_to_intermediate_mask
        self.in_features = in_features
        self.out_groups = out_groups
        self.intermediate_group_dim = intermediate_group_dim

        self.W_input_to_interm = nn.Parameter(torch.Tensor(self.in_features, self.intermediate_group_dim * self.out_groups))
        self.B_input_to_interm = nn.Parameter(torch.Tensor(self.intermediate_group_dim * self.out_groups))

        self._init_params()
    
    def _init_params(self):
        
        init.kaiming_normal_(self.W_input_to_interm, mode='fan_in')
        self.B_input_to_interm.data.fill_(0)
        
    def forward(self, layer_input, **kwargs):
        
        self.input_to_intermediate_mask = self.input_to_intermediate_mask.to(kwargs['current_device'])
        
        # input to interm
        masked_input_to_interm = self.W_input_to_interm.mul(self.input_to_intermediate_mask)
        interm_out = layer_input.matmul(masked_input_to_interm)
        interm_out = interm_out + self.B_input_to_interm
        interm_out = torch.tanh(interm_out)
        
        return interm_out

    def __repr__(self):
        return f"InputToIntermediate(in_features={self.in_features}, intermediate_group_dim={self.intermediate_group_dim}, " \
               f"out_groups={self.out_groups})"

class Intermediate(nn.Module):
    def __init__(self, in_out_groups, intermediate_group_dim, in_group_dim):

        super().__init__()

        self.intermediate_mask = torch.from_numpy(dag_utils.get_mask_intermediate_to_intermediate(intermediate_group_dim, in_out_groups, in_group_dim))
        self.in_out_groups = in_out_groups
        self.in_group_dim = in_group_dim
        self.out_features = intermediate_group_dim * in_out_groups
        self.intermediate_group_dim = intermediate_group_dim
        
        self.W_intermediate = nn.Parameter(torch.Tensor(self.in_out_groups * in_group_dim, 
                                                        self.in_out_groups * self.intermediate_group_dim))

        self.b_intermediate = nn.Parameter(torch.Tensor(self.in_out_groups * self.intermediate_group_dim))

        self._init_params()

    def _init_params(self):
        
        init.kaiming_normal_(self.W_intermediate, mode='fan_in')
        self.b_intermediate.data.fill_(0)
    
    def forward(self, layer_input, **kwargs):
        
        self.intermediate_mask = self.intermediate_mask.to(kwargs['current_device'])

        # input to interm
        masked_interm_to_interm = self.W_intermediate.mul(self.intermediate_mask)
        interm_out = layer_input.matmul(masked_interm_to_interm)
        interm_out = interm_out + self.b_intermediate
        interm_out = torch.tanh(interm_out)
        
        return interm_out

    def __repr__(self):
        return f"Intermediate(in_out_groups={self.in_out_groups}, intermediate_group_dim={self.intermediate_group_dim}, " \
               f"in_group_dim={self.in_group_dim})"

class IntermediateToOutput(nn.Module):
    
    def __init__(self, in_groups, in_group_dim):

        super().__init__()

        self.in_groups = in_groups
        self.in_group_dim = in_group_dim
        self.out_group_dim = 1
        self.output_mask = torch.from_numpy(dag_utils.get_mask_intermediate_to_intermediate(self.out_group_dim, in_groups, in_group_dim))
        
        self.W_out_mu = nn.Parameter(torch.Tensor(self.in_groups * self.in_group_dim, self.in_groups))
        self.b_out_mu = nn.Parameter(torch.Tensor(self.in_groups))
        
        self.W_out_logvar = nn.Parameter(torch.Tensor(self.in_groups * self.in_group_dim, self.in_groups))
        self.b_out_logvar = nn.Parameter(torch.Tensor(self.in_groups))

        self._init_params()
    
    def _init_params(self):
        
        init.kaiming_normal_(self.W_out_mu, mode='fan_in')
        init.kaiming_normal_(self.W_out_logvar, mode='fan_in')
        self.b_out_mu.data.fill_(0)
        self.b_out_logvar.data.fill_(0)
    
    def forward(self, layer_input, **kwargs):
        
        self.output_mask = self.output_mask.to(kwargs['current_device'])

        # \mu head
        masked_interm_to_output_mu = self.W_out_mu.mul(self.output_mask)
        mu_out = layer_input.matmul(masked_interm_to_output_mu)
        mu_out = mu_out + self.b_out_mu        
        #mu_out = F.relu(mu_out)
        
        # \sigma head
        masked_interm_to_output_sigma = self.W_out_logvar.mul(self.output_mask)
        logvar_out = layer_input.matmul(masked_interm_to_output_sigma)
        logvar_out = logvar_out + self.b_out_logvar
        #logvar_out = F.relu(logvar_out)
        
        return mu_out, logvar_out

    def __repr__(self):
        return f"IntermediateToOutput(in_groups={self.in_groups}, in_group_dim={self.in_group_dim}, " \
               f"out_group_dim={self.out_group_dim})"

class L0_Dense(nn.Module):
    
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, in_features, out_features, bias=True, l_two_weight=1., 
                 droprate_init=0.5, temperature=2./3.,
                 l_zero_weight=1., local_rep=False, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param l_zero_weight: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0_Dense, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        self.l_two_weight = l_two_weight
        self.l_zero_weight = l_zero_weight
        
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.use_bias = bias
        self.local_rep = local_rep

        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.qz_loga = nn.Parameter(torch.Tensor(in_features))
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.float_tensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.init_params()
        
    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(input.size(0), sample=self.training)
            xin = input.mul(z)
            output = xin.mm(self.weights)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)
        if self.use_bias:
            output.add_(self.bias)
        return output

    def init_params(self):

        init.kaiming_normal(self.weights, mode='fan_out')
        self.qz_loga.data.normal_(np.log(1 - self.droprate_init) - np.log(self.droprate_init), 1e-2)
        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=np.log(1e-2), max=np.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - LIMIT_A) / (LIMIT_B - LIMIT_A)
        logits = np.log(xn) - np.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=EPSILON, max=1 - EPSILON)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (LIMIT_B - LIMIT_A) + LIMIT_A

    def _l0_l2_reg(self):
        
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        
        logpw_col = 0.5 * self.l_two_weight * torch.sum(self.weights.pow(2), 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col) * self.l_zero_weight
        logpb = 0 if not self.use_bias else torch.sum(.5 * self.l_two_weight * self.bias.pow(2))
        return logpw + logpb

    def regularization(self):
        return self._l0_l2_reg()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.float_tensor(size).uniform_(EPSILON, 1 - EPSILON)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.float_tensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (LIMIT_B - LIMIT_A) + LIMIT_A, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.float_tensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def __repr__(self):
                
        s = ('{name}({in_features} -> {out_features}, droprate_init={droprate_init}, '
             'l_zero_weight={l_zero_weight}, temperature={temperature}, l_two_weight={l_two_weight}, '
             'local_rep={local_rep}')
        
        if not self.use_bias:
            s += ', bias=False'
        
        s += ')'
        
        return s.format(name=self.__class__.__name__, **self.__dict__)


class DAGInteractionLayer(nn.Module):

    def __init__(self, parents_list, children_list, adjacency_matrix, interm_unit_dim=1, 
                bias=True, parent_is_root=False, root_dim=10, **kwargs):

        super(DAGInteractionLayer, self).__init__()

        self.parent_is_single_root_node = parent_is_root
        self.root_dim = root_dim if self.parent_is_single_root_node else None

        self.in_features = len(parents_list) if not self.parent_is_single_root_node else self.root_dim
        self.out_features = len(children_list)

        self._parents, self._children = parents_list, children_list
        

        assert self.in_features > 0, f"Number of parents should be > 0, Given: {self.in_features}"
        assert self.out_features > 0, f"Number of children should be > 0, Given: {self.out_features}"
        assert interm_unit_dim > 0, f"Intermediate layer should have at least 1 unit, Given: {interm_unit_dim}"      
        
        self.interm_unit_dim = interm_unit_dim
        self.use_bias = bias
        self.adjacency_matrix = adjacency_matrix

        self.inp_to_interm = InputToIntermediate(self._get_mask_input_to_interm(), self.in_features, self.interm_unit_dim, self.out_features)
        self.interm1 = Intermediate(self.out_features, in_group_dim=self.interm_unit_dim, intermediate_group_dim=3)
        self.out = IntermediateToOutput(self.out_features, in_group_dim=3)
        

    def forward(self, layer_input, **kwargs):
        
        x = self.inp_to_interm(layer_input, **kwargs)
        #print(x)
        x = self.interm1(x, **kwargs)
        #print(x)
        mu, logvar = self.out(x, **kwargs)

        return mu, logvar
        
    def _get_mask_input_to_interm(self):
        
        # TODO: maybe we should allow this flexibility for every layer?
        # i.e. allow multiple units to represent a 'concept' / DAG node

        # If we get a DAG from chow-lin algo it has 1 top-level / root node. This will result in low capacity / bottle neck at 
        # the start of latent network, so instead of representing that root node with a single unit we can
        # instead use multiple units 
        if self.parent_is_single_root_node:
            C = len(self._children)
            return torch.from_numpy(np.ones(shape=(self.root_dim, C * self.interm_unit_dim), dtype=np.float32))
        
        np_mask = dag_utils.get_layer_mask(self._parents, self._children, self.interm_unit_dim, self.adjacency_matrix)
        return torch.from_numpy(np_mask)

class SimpleGNNLayer(nn.Module):
    """
    Can be used to implement GNNs for P(Z|epsilon, A) or Q(Z|X,A)
    """
    def __init__(self, in_node_feat_dim, out_node_feat_dim, adj_mat, is_final_layer=False):
        super().__init__()

        self.in_node_feat_dim = in_node_feat_dim
        self.out_node_feat_dim = out_node_feat_dim
        self.is_final_layer = is_final_layer
        # TODO: check if T is reqd
        self.A = adj_mat.T # this is reqd because the stored mat is in from-to form but the impl needs to-from

        self.num_neighbours = self.A.sum(dim=-1, keepdims=True)
        self.projection = nn.Linear(self.in_node_feat_dim, self.out_node_feat_dim)
    
    def forward(self, node_feats):
        
        self.A = self.A.to(node_feats.device)
        self.num_neighbours = self.num_neighbours.to(node_feats.device)
        
        #print("Input feats: ", node_feats)
        node_feats = self.projection(node_feats)
        #print("Projected feats: ", node_feats)
        node_feats = torch.matmul(self.A, node_feats)
        node_feats = node_feats / self.num_neighbours
        #print("Exchanged feats: ", node_feats)
        
        if self.is_final_layer:
            # split into mu and sigma
            node_feats_mu, node_feats_logvar = node_feats.chunk(2, dim=2)
            return node_feats_mu, node_feats_logvar
        else:
            node_feats = torch.tanh(node_feats)
            #print("Tanh() feats: ", node_feats)
            return node_feats

    def __repr__(self):
        
        return self.projection.__repr__() + f" is_final_layer={self.is_final_layer}"

class PriorGNNLayer(nn.Module):
    """
    For now in this impl. all prior nodes have 1-D features 
    """
    def __init__(self, num_nodes, adj_mat, is_final_layer=False):
        super().__init__()

        self.num_nodes = num_nodes
        self.is_final_layer = is_final_layer
        self.A = adj_mat.T # this is reqd because the stored mat is in from-to form but the impl needs to-from

        self.num_neighbours = self.A.sum(dim=-1, keepdims=True)

        # Represents a diagonal weight matrix W so that all dims can be 
        # multiplied by a unique scalar i.e. z = Wg + b
        # Features are 1-D
        self.projection_W = nn.Parameter(torch.Tensor(self.num_nodes))
        self.projection_b = nn.Parameter(torch.Tensor(self.num_nodes))

        self._init_params()
    
    def _init_params(self):
        
        init.kaiming_normal_(self.projection_W, mode='fan_in')
        self.projection_b.data.fill_(0)
    
    def forward(self, node_feats):
        
        self.A = self.A.to(node_feats.device)
        self.num_neighbours = self.num_neighbours.to(node_feats.device)
        self.projection_W = self.projection_W.to(node_feats.device)
        self.projection_b = self.projection_b.to(node_feats.device)

        node_feats = self.projection_W.mul(node_feats) + self.projection_b
        node_feats = torch.matmul(self.A, node_feats)
        node_feats = node_feats / self.num_neighbours

        return node_feats if self.is_final_layer else torch.tanh(node_feats)

class BifurcatedGNNLayer(nn.Module):
    """
    Can be used to implement GNNs for P(Z|epsilon, A) or Q(Z|X,A)
    """
    def __init__(self, adj_mat, in_ind_dim, out_ind_dim, in_dep_dim, out_dep_dim, layer_type=None):
        super().__init__()

        self.layer_type = layer_type
        self.A = adj_mat.T # this is reqd because the store mat is in from-to form but the impl needs to-from
        
        self.in_ind_dim = in_ind_dim
        self.out_ind_dim = out_ind_dim
        self.in_dep_dim = in_dep_dim
        self.out_dep_dim = out_dep_dim

        self.num_neighbours = self.A.sum(dim=-1, keepdims=True)
        self.dependent_nodes_mask = self.num_neighbours - 1
        
        if self.layer_type == 'first':
            self.projection_ind = nn.Linear(in_ind_dim, out_ind_dim * 2)
            self.projection_dep = nn.Linear(in_dep_dim, out_dep_dim * 2)
        else:
            self.projection_ind = nn.Linear(in_ind_dim * 2, out_ind_dim * 2)
            self.projection_dep = nn.Linear(in_dep_dim * 2, out_dep_dim * 2)
    
    def forward(self, node_feats):
        
        node_feats_dep, node_feats_ind = node_feats
        
        self.A = self.A.to(node_feats_dep.device)
        self.num_neighbours = self.num_neighbours.to(node_feats_dep.device)
        self.dependent_nodes_mask = self.dependent_nodes_mask.to(node_feats_dep.device)
        
        if self.layer_type == 'first':
            
            # in the first layer we get combined features i.e. there's no bifurcation 
            # b/w dependent and independent features. Since they're coming from 
            # encoder CNN net, they're all kinda dependent
            
            # project to ind feat dim - prep indept features first
            indep_node_feats = self.projection_ind(node_feats_dep)
            indep_node_feats = torch.tanh(indep_node_feats)   
        
            # prep dep features
            dep_node_feats = self.projection_dep(node_feats_dep)
            dep_node_feats = torch.matmul(self.A, dep_node_feats)
            dep_node_feats = dep_node_feats / self.num_neighbours
            dep_node_feats = torch.tanh(dep_node_feats)
            dep_node_feats = dep_node_feats * self.dependent_nodes_mask
            
            return dep_node_feats, indep_node_feats
        
        else: # if not first layer...
            
            # take in (dep, ind) return (dep, ind)
            # project to ind feat dim - prep indept features first
            indep_node_feats = self.projection_ind(node_feats_ind)
            indep_node_feats = torch.tanh(indep_node_feats)

            # prep dep features
            dep_node_feats = self.projection_dep(node_feats_dep)
            dep_node_feats = torch.matmul(self.A, dep_node_feats)
            dep_node_feats = dep_node_feats / self.num_neighbours   
            
            if self.layer_type == 'last':

                # split into mu and sigma
                dep_node_feats = dep_node_feats * self.dependent_nodes_mask
                dep_mu, dep_logvar = dep_node_feats.chunk(2, dim=2)
                indep_mu, indep_logvar = indep_node_feats.chunk(2, dim=2)

                # combine params
                mu = torch.cat([indep_mu, dep_mu], dim=2)
                logvar = torch.cat([indep_logvar, dep_logvar], dim=2)

                return mu, logvar

            else:
                dep_node_feats = torch.tanh(dep_node_feats)
                dep_node_feats = dep_node_feats * self.dependent_nodes_mask
                return dep_node_feats, indep_node_feats

    def __repr__(self):
        
        return self.projection_ind.__repr__() + \
               self.projection_dep.__repr__() + \
            f" layer_type={self.layer_type} " + \
            f"in_ind_dim={self.in_ind_dim}, out_ind_dim={self.out_ind_dim} " + \
            f"in_dep_dim={self.in_dep_dim}, out_dep_dim={self.out_dep_dim} "

class SupervisedRegulariser(nn.Module):

    def __init__(self, num_nodes, node_features_dim, w_sup_reg, node_labels = None, labels_type="regression"):
        
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.labels_type = labels_type
        self.w_sup_reg = w_sup_reg
        self.supervised_regularisers = self._get_sup_reg_models()
        self.node_labels = node_labels

    def forward(self, node_features):
        """"
        node_features has shape (batch, V, feature_dim)
        """

        # We need to break it into V different vectors
        # i.e. a list L which has elements of shape (batch, feature_dim) and len(L) = V
        node_features_separated = node_features.chunk(self.num_nodes, dim=1)

        # now we have a list where node_features_separate[i] gives features associated 
        # with i-th node. But the current output shape would be (batch, 1, feature_dim)
        # so we have to squeeze out the singleton dim

        predictions_all_nodes = []
        for node_idx in range(self.num_nodes):
            
            # convert (batch, 1, feature_dim) to (batch, feature_dim)
            features_of_this_node = node_features_separated[node_idx].squeeze(dim=1)

            predictions_all_nodes.append(self.supervised_regularisers[node_idx](features_of_this_node))

        return predictions_all_nodes
        

    def loss(self, predictions_all_nodes, targets_all_nodes):
        """
        predictions_all_nodes: List of length `self.num_nodes`, where each element i of the list corresponds to the batch of predictions associated with i-th node 
        targets_all_nodes: Similar as above
        """
        loss_per_node = dict()
        total_loss = 0.0
        targets_all_nodes = targets_all_nodes.chunk(self.num_nodes, dim=1)
        for node_idx in range(self.num_nodes):

            if self.labels_type == "regression":
                loss_this_node = F.mse_loss(predictions_all_nodes[node_idx], targets_all_nodes[node_idx], reduction='mean')
            elif self.labels_type == "binary":
                loss_this_node = F.binary_cross_entropy(predictions_all_nodes[node_idx], targets_all_nodes[node_idx], reduction='mean')
            else:
                raise NotImplemented()
            
            total_loss += loss_this_node
            loss_per_node[f'clf_{self.node_labels[node_idx]}'] = loss_this_node.detach()

        return total_loss * self.w_sup_reg, loss_per_node

    def get_loss(self, node_idx):
        raise NotImplemented()
    
    def _get_sup_reg_models(self):

        if self.labels_type == "regression":
            return nn.ModuleList([nn.Linear(self.node_features_dim, 1) for n in range(self.num_nodes)])
        
        elif self.labels_type == "binary":
            return nn.ModuleList(
            [ 
                nn.Sequential(
                    nn.Linear(self.node_features_dim, 2), 
                    nn.Sigmoid()
                ) for _ in range(self.num_nodes)
            ]
        )
        else:
            raise NotImplemented()


class GroundTruthBasedPriorNetwork(nn.Module):

    def __init__(self, num_nodes, adj_mat):
        
        super().__init__()

        # transpose is reqd because the stored mat is in from-to form but the impl needs to-from.
        # unlike other modules, here we need a numpy mat because we have to use nonzero()
        self.A = adj_mat
        self.num_nodes = num_nodes
        self.num_neighbours = self.A.sum(axis=-1, keepdims=True)
        print(self.A)
        print(self.num_neighbours)
        self.dist_param_nets = self._get_dist_param_nets()
        

    def forward(self, gt_labels):
        
        """"
        gt_labels has shape (batch, l_dim)
        """
        all_mus = []
        
        for node_idx, prior_module in enumerate(self.dist_param_nets):
            
            # get non-zero index
            parents_idx = self.A[node_idx].nonzero()[0]
            #print("parents_idx")
            #print(parents_idx)
            parents_values = gt_labels[:, parents_idx]
            mu_given_pa = prior_module(parents_values)

            all_mus.append(mu_given_pa)

        # mu_given_pa has shape (batch, 1) so we concat all of them together
        # Return shape is again (batch, l_dim)
        # I'm using fixed Std=1 here, hence zeros in 2nd return val
        # change (B, l_dim) into (B, l_dim, 1) which represents l_dim nodes with 1d feature
        node_aligned_shape = (*gt_labels.size(), 1)
        mus = torch.cat(all_mus, dim=1).reshape(node_aligned_shape)
        logvars = torch.zeros(size=node_aligned_shape).to(mus.device)
        
        return mus, logvars
    
    def _get_dist_param_nets(self):

        modules_list = []

        for node_idx in range(self.num_nodes):
            
            # for each node we figure out how many parents does it have.
            # This helps us determine the intermediate layer value
            num_parents = int(self.num_neighbours[node_idx])
            #print(num_parents)
            # this module produces prior gt based \mu for this node given gt values of the parents 
            modules_list.append(
                    nn.Sequential(
                        nn.Linear(num_parents, num_parents * 2), 
                        nn.Tanh(),
                        nn.Linear(num_parents * 2, 1)
                    )
            )
        
        return nn.ModuleList(modules_list)
