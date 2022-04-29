import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

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