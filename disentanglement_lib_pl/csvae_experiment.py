
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

        # TODO: need to visualize CS - VAE specific stuff here
        all_loss_keys = train_step_outputs[0].keys()

        # Add KLD Loss for every layer
        per_layer_kld_keys = [key for key in all_loss_keys if 'KLD_z_' in key]
        for kld_loss_key in per_layer_kld_keys:
            kld_loss = torch.stack([tso[kld_loss_key] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"KLD_Per_Layer/{kld_loss_key}", kld_loss, self.current_epoch)

        # TODO: find out a way to cleanly visualize mu and logvar of multiple layers        
        # Visualize Components of mean and sigma vector for every layer
        all_td_net_outs = [tso['td_net_outs'] for tso in train_step_outputs]
        td_net_count = len(all_td_net_outs[0])
        
        for t in range(td_net_count):
            # get mu and sigma of t-th layer
            #mus
            mus = torch.cat([tdno[t]['mu_q'] for tdno in all_td_net_outs], dim=0).mean(0).tolist()
            mu_dict = {f"mu_q_layer_{t}/component_{i}": component_val for i, component_val in enumerate(mus)}            
            for k , v in mu_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)

            #sigmas = torch.cat([tdno[t]['sigma_q'] for tdno in all_td_net_outs], dim=0).mean(0).tolist()
            #sigma_dict = {f"sigma_q_{t}_[{i}]": component_val for i, component_val in enumerate(sigmas)}
            #self.logger.experiment.add_scalars(f"logvar_{t} components", sigma_dict, self.current_epoch)

        #if self.current_epoch == 0:
        #    print(tags)
            # layout = {
            #     'Layer0': {'mu_q_0': ['Multiline', tags[0]]}, 
            #     'Layer1': {'mu_q_1': ['Multiline', tags[1]]}
            # }
            # self.logger.experiment.add_custom_scalars(layout)
        
        torch.set_grad_enabled(True)
        self.model.train()

