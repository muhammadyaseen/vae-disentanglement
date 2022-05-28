
from numpy.lib.function_base import select
import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from torch import optim
from torchvision import transforms

from models.ladder_vae import LadderVAE
from common import constants as c
from common import data_loader
from evaluation import evaluation_utils

from base_vae_experiment import BaseVAEExperiment

class LadderVAEExperiment(BaseVAEExperiment):

    def __init__(self,
                 vae_model: LadderVAE,
                 params: dict,
                 dataset_params: dict) -> None:
        
        super(LadderVAEExperiment, self).__init__(vae_model, params, dataset_params)
      
        self.l_zero_reg = params['l_zero_reg']

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        super(LadderVAEExperiment, self).training_step(self, batch, batch_idx, optimizer_idx)

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

        # TODO: Doing .cpu() here will cause problems in multiple gpu mode
        # Move it to *_epoch_end functions ?
        #losses['mu_batch'] = mu.detach().cpu()
        #losses['logvar_batch'] = logvar.detach().cpu()

        return losses
    
    def training_epoch_end(self, train_step_outputs):
        
        super(LadderVAEExperiment, self).training_epoch_end(train_step_outputs)

        torch.set_grad_enabled(False)
        self.model.eval()
        
        # 1. Show algo specific scalars
        
        avg_kld_z1_loss = torch.stack([tso["kld_z1"] for tso in train_step_outputs]).mean()
        avg_kld_z2_loss = torch.stack([tso["kld_z2"] for tso in train_step_outputs]).mean()
        
        self.logger.experiment.add_scalar("KLD Loss z1 (Train)", avg_kld_z1_loss, self.current_epoch)
        self.logger.experiment.add_scalar("KLD Loss z2 (Train)", avg_kld_z2_loss, self.current_epoch)

        if self.l_zero_reg:
            reg_loss = torch.stack([tso["l_zero_reg"] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar("Reg L-0 Loss", reg_loss, self.current_epoch)
        
        # TODO: find out a way to cleanly visualize mu and logvar of multiple layers

        torch.set_grad_enabled(True)
        self.model.train()




