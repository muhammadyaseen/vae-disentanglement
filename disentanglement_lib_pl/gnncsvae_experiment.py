
import torch
from base_vae_experiment import BaseVAEExperiment
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_colormaps

from models.csvae_gnn import GNNBasedConceptStructuredVAE
from common.utils import CenteredNorm
from common import constants as c

class GNNCSVAEExperiment(BaseVAEExperiment):

    def __init__(self,
                 vae_model: GNNBasedConceptStructuredVAE,
                 params: dict,
                 dataset_params: dict) -> None:
        
        super(GNNCSVAEExperiment, self).__init__(vae_model, params, dataset_params)
        

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        super(GNNCSVAEExperiment, self).training_step(batch, batch_idx, optimizer_idx)
        
        x_true, labels = batch
        self.current_device = x_true.device
        fwd_pass_results = self.forward(x_true, labels=labels, current_device=self.current_device)

        fwd_pass_results.update({
            'x_true': x_true,
            'true_latents': labels,
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch
        })
        
        train_step_outputs = self.model.loss_function(loss_type='cross_ent', **fwd_pass_results)

        # We need it for visualizing per layer mean / sigma components
        train_step_outputs.update({
            "prior_mu": fwd_pass_results['prior_mu'].detach(),
            "prior_logvar": fwd_pass_results['prior_logvar'].detach(),
            "posterior_mu": fwd_pass_results['posterior_mu'].detach(), 
            "posterior_logvar": fwd_pass_results['posterior_logvar'].detach(),
        })

        return train_step_outputs        
        
    def training_epoch_end(self, train_step_outputs):

        super(GNNCSVAEExperiment, self).training_epoch_end(train_step_outputs)

        torch.set_grad_enabled(False)
        self.model.eval()

        # Add KLD Loss for every layer
        self._log_kld_loss_per_node(train_step_outputs)

        # Visualize Components of mean and sigma vector for every layer
        self._log_mu_per_node(train_step_outputs)
        self._log_logvar_per_node(train_step_outputs)

        if self.model.add_classification_loss:
            self._log_classification_losses(train_step_outputs)
        
        torch.set_grad_enabled(True)
        self.model.train()

    def _log_kld_loss_per_node(self, train_step_outputs):
        
        all_loss_keys = train_step_outputs[0].keys()

        per_node_kld_keys = [key for key in all_loss_keys if 'KLD_z_' in key]
        
        for kld_loss_key in per_node_kld_keys:
            kld_loss = torch.stack([tso[kld_loss_key] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"KLD_Per_Node/{kld_loss_key}", kld_loss, self.current_epoch)

    def _log_mu_per_node(self, train_step_outputs):
       
        post_mus = torch.cat([tso['posterior_mu'] for tso in train_step_outputs], dim=0)
        prior_mus = torch.cat([tso['prior_mu'] for tso in train_step_outputs], dim=0)
        #print(post_mus.shape)

        post_mus_avgs = post_mus.mean(0).tolist()
        prior_mus_avgs = prior_mus.mean(0).tolist()

        #print(post_mus)
        for node_idx in range(self.model.num_nodes):
            
            # Histograms
            # Loop over every dim of mu associated with this node and add its histogram
            for k in range(post_mus.shape[1]):
                self.logger.experiment.add_histogram(f"Mu_q{node_idx + 1}/Dim_{k}", post_mus[:, node_idx, k], self.current_epoch)
                self.logger.experiment.add_histogram(f"Mu_p{node_idx + 1}/Dim_{k}", prior_mus[:, node_idx, k], self.current_epoch)
            
            # Scalars           
            # we do '+1' because latent indexing is 1-based, there is no Z_0
            post_mu_dict = {f"Mu_q{node_idx + 1}/component_{i}": component_val for i, component_val in enumerate(post_mus_avgs[node_idx])}
            #print(post_mu_dict)
            for k , v in post_mu_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)
            
            prior_mu_dict = {f"Mu_p{node_idx + 1}/component_{i}": component_val for i, component_val in enumerate(prior_mus_avgs[node_idx])}            
            for k , v in prior_mu_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)
    
    def _log_logvar_per_node(self, train_step_outputs):

        post_logvars = torch.cat([tso['posterior_logvar'] for tso in train_step_outputs], dim=0)
        prior_logvars = torch.cat([tso['prior_logvar'] for tso in train_step_outputs], dim=0)
        #print(post_logvars.shape)

        post_logvars_avgs = post_logvars.mean(0).tolist()
        prior_logvars_avgs = prior_logvars.mean(0).tolist()

        for node_idx in range(self.model.num_nodes):
            
            # Histograms
            # Loop over every dim and add its histogram
            for k in range(post_logvars.shape[1]):
                self.logger.experiment.add_histogram(f"LogVar_q{node_idx + 1}/Dim_{k}", post_logvars[:, node_idx, k], self.current_epoch)
                self.logger.experiment.add_histogram(f"LogVar_p{node_idx + 1}/Dim_{k}", prior_logvars[:, node_idx, k], self.current_epoch)
            
            # Scalars
            # we do '+1' because latent indexing is 1-based, there is no Z_0
            post_logvar_dict = {f"LogVar_q{node_idx + 1}/component_{i}": component_val for i, component_val in enumerate(post_logvars_avgs[node_idx])}            
            for k , v in post_logvar_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)

            prior_logvar_dict = {f"LogVar_p{node_idx + 1}/component_{i}": component_val for i, component_val in enumerate(prior_logvars_avgs[node_idx])}            
            for k , v in prior_logvar_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)
            
    def _log_classification_losses(self, train_step_outputs):
        
        clf_loss = torch.stack([tso[c.AUX_CLASSIFICATION] for tso in train_step_outputs]).mean()
        self.logger.experiment.add_scalar(f"Clf_Loss", clf_loss, self.current_epoch)