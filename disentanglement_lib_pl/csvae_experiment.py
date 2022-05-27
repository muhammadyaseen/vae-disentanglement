
from numpy.lib.function_base import select
import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim
from torchvision import transforms
from base_vae_experiment import BaseVAEExperiment

from models.cs_vae import ConceptStructuredVAE
from common import constants as c
from common import data_loader
from evaluation import evaluation_utils

class ConceptStructuredVAEExperiment(BaseVAEExperiment):

    def __init__(self,
                 vae_model: ConceptStructuredVAE,
                 params: dict) -> None:
        
        super(ConceptStructuredVAEExperiment, self).__init__(vae_model, params)
        

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        super(ConceptStructuredVAEExperiment, self).training_step(batch, batch_idx, optimizer_idx)
        
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
        
        super(BaseVAEExperiment, self).training_epoch_end(train_step_outputs)

        torch.set_grad_enabled(False)
        self.model.eval()
        
        # 1. Show algo specific scalars
        
        # TODO: find out a way to cleanly visualize mu and logvar of multiple layers
        # TODO: need to visualize CS - VAE specific stuff here

        torch.set_grad_enabled(True)
        self.model.train()





 