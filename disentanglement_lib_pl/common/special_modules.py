import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from common import dag_utils

LIMIT_A, LIMIT_B, EPSILON = -.1, 1.1, 1e-6


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

    def __init__(self, parents_list, children_list, adjacency_matrix, interm_unit_dim=1, bias=True, **kwargs):

        super(DAGInteractionLayer, self).__init__()

        self.in_features, self.out_features = len(parents_list), len(children_list)
        self._parents, self._children = parents_list, children_list
        
        assert self.in_features > 0, f"Number of parents should be > 0, Given: {self.in_features}"
        assert self.out_features > 0, f"Number of children should be > 0, Given: {self.out_features}"
        assert interm_unit_dim > 0, f"Intermediate layer should have at least 1 unit, Given: {interm_unit_dim}"      
        
        self.interm_unit_dim = interm_unit_dim
        self.use_bias = bias
        self.adjacency_matrix = adjacency_matrix

        # ---------------------#
        # Learnable Parameters #
        # -------------------- #
        self.W_input_to_interm = nn.Parameter(torch.Tensor(self.in_features, self.interm_unit_dim * self.out_features))
        self.W_interm_to_output = nn.Parameter(torch.Tensor(self.interm_unit_dim * self.out_features, self.out_features))
        
        if self.use_bias:
            # we will need mask on bias as well 
            self.B_input_to_interm = nn.Parameter(torch.Tensor(interm_unit_dim * self.out_features))
            self.B_interm_to_output = nn.Parameter(torch.Tensor(self.out_features))
        # -------------------------#
        # End Learnable Parameters #
        # ------------------------ #

        self.mask_input_to_interm = self._get_mask_input_to_interm()
        self.mask_interm_to_output = self._get_mask_interm_to_output()
        
        self._init_params()

        # TODO: here we can probably associate names to units for better
        # readability etc.
        if 'parent_names' in kwargs.keys():
            pass

        if 'children_names' in kwargs.keys():
            pass

    def forward(self, layer_input):
        
        # input to interm
        masked_input_to_interm = self.W_input_to_interm.mul(self.mask_input_to_interm)
        interm_out = layer_input.matmul(masked_input_to_interm)
        interm_out = interm_out + self.B_input_to_interm
        interm_out = F.relu(interm_out)
        
        # interm to output
        masked_interm_to_output = self.W_interm_to_output.mul(self.mask_interm_to_output)
        out = interm_out.matmul(masked_interm_to_output)
        out = out + self.B_interm_to_output
        out = F.relu(out)

        return out

    def diagnostic_forward(self, layer_input):
        
        # input to interm
        print("Input: ", layer_input)
        print("mask_input_to_interm: ", self.mask_input_to_interm)

        masked_input_to_interm = self.W_input_to_interm.mul(self.mask_input_to_interm)
        print("masked_input_to_interm:", masked_input_to_interm)
        print("masked_input_to_interm.Size():", masked_input_to_interm.size())

        interm_out = layer_input.matmul(masked_input_to_interm)
        print("X * W_1", interm_out)
        print("X * W_1 size()", interm_out.size())

        interm_out = interm_out + self.B_input_to_interm

        print("X * W_1 + b_1", interm_out)

        interm_out = F.relu(interm_out)
        print("U = ReLU(X * W_1 + b_1)", interm_out)
        print("U.size()", interm_out.size())
        
        # interm to output
        print("mask_interm_to_output ", self.mask_interm_to_output)

        masked_interm_to_output = self.W_interm_to_output.mul(self.mask_interm_to_output)
        print("masked_interm_to_output: ", masked_interm_to_output)

        out = interm_out.matmul(masked_interm_to_output)
        print("U * W_2", out)
        print("U * W_2 size", out.size())

        out = out + self.B_interm_to_output
        print("U * W_2 + b_2", out)

        out = F.relu(out)
        print("Out = ReLU(U * W_2 + b_2)", out)
        print("Out.size()", out.size())

        return out

    def _init_params(self):
        
        # Say Bismillah
        init.kaiming_normal_(self.W_input_to_interm, mode='fan_out')
        init.kaiming_normal_(self.W_interm_to_output, mode='fan_out')
        
        if self.use_bias:
            self.B_input_to_interm.data.fill_(0)
            self.B_interm_to_output.data.fill_(0)

    def _get_mask_input_to_interm(self):
        
        np_mask = dag_utils.get_layer_mask(self._parents, self._children, self.interm_unit_dim, self.adjacency_matrix)
        return torch.from_numpy(np_mask)

    def _get_mask_interm_to_output(self):
        
        np_mask = dag_utils.get_mask_for_intermediate_to_output(self.interm_unit_dim, self.out_features)
        return torch.from_numpy(np_mask)
   
    def __repr__(self):

        s = ('{name} ({in_features} -> {interm_dim} -> {out_features}, '
        'mask = {interaction_mask}')

        if not self.use_bias:
            s += ', bias=False'
        
        s += ')'
        
        return s.format(name=self.__class__.__name__, **self.__dict__)
