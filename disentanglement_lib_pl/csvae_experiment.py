
import torch
import torchvision
from base_vae_experiment import BaseVAEExperiment
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_colormaps

from models.cs_vae import ConceptStructuredVAE
from common.utils import CenteredNorm

class ConceptStructuredVAEExperiment(BaseVAEExperiment):

    def __init__(self,
                 vae_model: ConceptStructuredVAE,
                 params: dict,
                 dataset_params: dict) -> None:
        
        super(ConceptStructuredVAEExperiment, self).__init__(vae_model, params, dataset_params)
        

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        super(ConceptStructuredVAEExperiment, self).training_step(batch, batch_idx, optimizer_idx)
        
        x_true, label = batch
        self.current_device = x_true.device
        fwd_pass_results = self.forward(x_true, label=label, current_device=self.current_device)

        fwd_pass_results.update({
            'x_true': x_true,
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch
        })
        
        train_step_outputs = self.model.loss_function(loss_type='cross_ent', **fwd_pass_results)

        # We need it for visualizing per layer mean / sigma components
        train_step_outputs.update({
            'td_net_outs': fwd_pass_results['td_net_outs']
        })

        return train_step_outputs        
        
    def training_epoch_end(self, train_step_outputs):

        super(ConceptStructuredVAEExperiment, self).training_epoch_end(train_step_outputs)

        torch.set_grad_enabled(False)
        self.model.eval()

        # Add KLD Loss for every layer
        self._log_kld_loss_per_layer(train_step_outputs)

        # Visualize Components of mean and sigma vector for every layer
        self._log_mu_per_layer(train_step_outputs)
        self._log_logvar_per_layer(train_step_outputs)

        # Visualize per layer weights
        self._log_per_layer_weights(train_step_outputs)

        if self.model.add_classification_loss:
            self._log_classification_losses(train_step_outputs)
        
        torch.set_grad_enabled(True)
        self.model.train()

    def _log_kld_loss_per_layer(self, train_step_outputs):
        
        all_loss_keys = train_step_outputs[0].keys()

        per_layer_kld_keys = [key for key in all_loss_keys if 'KLD_z_' in key]
        
        for kld_loss_key in per_layer_kld_keys:
            kld_loss = torch.stack([tso[kld_loss_key] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"KLD_Per_Layer/{kld_loss_key}", kld_loss, self.current_epoch)

    def _log_mu_per_layer(self, train_step_outputs):
        """
        only logging mu for now
        """
        all_td_net_outs = [tso['td_net_outs'] for tso in train_step_outputs]
        # We do '+1' here because if we have K latents we will have K-1 td_nets,
        # but even then in function cs_vae._top_down_pass() we append an extra 
        # output that comes from the last BU net and serves as input to 1st TD net
        # That last BU net outputs the variational dist params that we want to visualize
        td_net_count = len(self.model.top_down_networks) + 1
        
        # reverse because tdnet[0] actually corresponds to params for z_L
        for net_idx, latent_idx in zip(range(td_net_count), range(td_net_count)[::-1]):
            
            # Histograms
            mus = torch.cat([tdno[net_idx]['mu_q'] for tdno in all_td_net_outs], dim=0)
            # Loop over every dim and add its histogram
            for k in range(mus.shape[1]):
                self.logger.experiment.add_histogram(f"Mu_q{latent_idx + 1}/Dim_{k}", mus[:, k], self.current_epoch)
            
            # Scalars
            mus = mus.mean(0).tolist()
            # we do '+1' because latent indexing is 1-based, there is no Z_0
            mu_dict = {f"Mu_q{latent_idx + 1}/component_{i}": component_val for i, component_val in enumerate(mus)}            
            for k , v in mu_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)
    
    def _log_logvar_per_layer(self, train_step_outputs):

        all_td_net_outs = [tso['td_net_outs'] for tso in train_step_outputs]
        # We do '+1' here because if we have K latents we will have K-1 td_nets,
        # but even then in function cs_vae._top_down_pass() we append an extra 
        # output that comes from the last BU net and serves as input to 1st TD net
        # That last BU net outputs the variational dist params that we want to visualize
        td_net_count = len(self.model.top_down_networks) + 1
        
        # reverse because tdnet[0] actually corresponds to params for z_L
        for net_idx, latent_idx in zip(range(td_net_count), range(td_net_count)[::-1]):
            
            # Histograms
            logvars = torch.cat([tdno[net_idx]['sigma_q'] for tdno in all_td_net_outs], dim=0)
            # Loop over every dim and add its histogram
            for k in range(logvars.shape[1]):
                self.logger.experiment.add_histogram(f"LogVar_q{latent_idx + 1}/Dim_{k}", logvars[:, k], self.current_epoch)
            
            # Scalars
            logvars = logvars.mean(0).tolist()
            # we do '+1' because latent indexing is 1-based, there is no Z_0
            mu_dict = {f"LogVar_q{latent_idx + 1}/component_{i}": component_val for i, component_val in enumerate(logvars)}            
            for k , v in mu_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)

    def _log_mu_histograms(self, train_step_outputs):
        
        all_td_net_outs = [tso['td_net_outs'] for tso in train_step_outputs]
        td_net_count = len(self.model.top_down_networks) + 1
        
        # Every td_net gives 1 (multidim) mu
        for net_idx, latent_idx in zip(range(td_net_count), range(td_net_count)[::-1]):

            mus = torch.cat([tdno[net_idx]['mu_q'] for tdno in all_td_net_outs], dim=0) #.mean(0).tolist()
        
            # Loop over every dim and add its histogram
            for k in range(mus.shape[1]):
                self.logger.experiment.add_histogram(f"Mu_{latent_idx + 1}/Dim_{k}", mus[:, k], self.current_epoch)

    def _log_per_layer_weights(self, train_step_outputs):
        
        T = len(self.model.top_down_networks)

        for t, td_net in enumerate(self.model.top_down_networks):

            full_and_masked_side_by_side = torch.cat([td_net.inp_to_interm.W_input_to_interm, 
                                                      td_net.inp_to_interm.W_input_to_interm.mul(td_net.inp_to_interm.input_to_intermediate_mask)], 
                                                dim = 0).cpu().numpy()
            plt.gcf().tight_layout(pad=0)
            plt.gca().margins(0)
            plt.axis('off')
            plt.imshow(full_and_masked_side_by_side, cmap=mpl_colormaps.coolwarm, norm=CenteredNorm())
        
            self.logger.experiment.add_figure(f"Weights/Z{1+T-t}-to-Z{T-t}", plt.gcf(), self.current_epoch)
            
    def _log_classification_losses(self, train_step_outputs):
        
        all_loss_keys = train_step_outputs[0].keys()

        per_layer_keys = [key for key in all_loss_keys if 'clf_loss_' in key]
        
        for loss_key in per_layer_keys:
            loss = torch.stack([tso[loss_key] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"Clf_Loss_Per_Layer/{loss_key}", loss, self.current_epoch)