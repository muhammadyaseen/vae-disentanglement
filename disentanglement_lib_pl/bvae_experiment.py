import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim

from models.vae import VAE
from common import constants as c
from common import data_loader
from evaluation import evaluation_utils
from base_vae_experiment import BaseVAEExperiment

class BVAEExperiment(BaseVAEExperiment):

    def __init__(self,
                 vae_model: VAE,
                 params: dict,
                 dataset_params: dict) -> None:
        
        super(BVAEExperiment, self).__init__(vae_model, params, dataset_params)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        super(BVAEExperiment, self).training_step(batch, batch_idx, optimizer_idx)

        x_true, label = batch
        self.current_device = x_true.device
        fwd_pass_results = self.forward(x_true, label=label, current_device=self.current_device)

        fwd_pass_results.update({
            'x_true': x_true,
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx,
            'global_step': self.global_step
        })

        losses = self.model.loss_function(loss_type='cross_ent', **fwd_pass_results)

        return losses
    
    def training_epoch_end(self, train_step_outputs):
               
        super(BVAEExperiment, self).training_epoch_end(train_step_outputs)

        torch.set_grad_enabled(False)
        self.model.eval()
        
        # Visualize Components of mean and sigma vector for every layer
        self._log_mu_per_layer(train_step_outputs)
        self._log_mu_histograms(train_step_outputs)

        if isinstance(self.model, VAE) and self.model.controlled_capacity_increase:
            self.logger.experiment.add_scalar("C", self.model.c_current, self.global_step)

        if 'BetaTCVAE' in self.model.loss_terms:
            avg_tc_loss = torch.stack([tso['vae_betatc'] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar("TC Loss (Train)", avg_tc_loss, self.current_epoch)
        
        
        torch.set_grad_enabled(True)
        self.model.train()

    def _log_mu_per_layer(self, train_step_outputs):
        """
        only logging mu for now
        """
        mus = torch.cat([tso['mu'] for tso in train_step_outputs], dim=0).mean(0).tolist()
        
        mu_dict = {f"mu_q/component_{i}": component_val for i, component_val in enumerate(mus)}            
        for k , v in mu_dict.items():
            self.logger.experiment.add_scalar(k, v, self.current_epoch)

    def _log_mu_histograms(self, train_step_outputs):
        
        mus = torch.cat([tso['mu'] for tso in train_step_outputs], dim=0)
        
        for k in range(mus.shape[1]):
            self.logger.experiment.add_histogram(f"Mu/Dim_{k}", mus[:, k], self.current_epoch)