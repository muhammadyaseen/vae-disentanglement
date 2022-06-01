
import torch
import torchvision
from base_vae_experiment import BaseVAEExperiment

from models.cs_vae import ConceptStructuredVAE

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

        # Add KLD Loss for every layer
        self._log_kld_loss_per_layer(train_step_outputs)

        # Visualize Components of mean and sigma vector for every layer
        self._log_mu_sigma_per_layer(train_step_outputs)

        # Visualize per layer weights
        self._log_per_layer_weights(train_step_outputs)

        torch.set_grad_enabled(True)
        self.model.train()

    def _log_kld_loss_per_layer(self, train_step_outputs):
        
        all_loss_keys = train_step_outputs[0].keys()

        per_layer_kld_keys = [key for key in all_loss_keys if 'KLD_z_' in key]
        
        for kld_loss_key in per_layer_kld_keys:
            kld_loss = torch.stack([tso[kld_loss_key] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"KLD_Per_Layer/{kld_loss_key}", kld_loss, self.current_epoch)

    def _log_mu_sigma_per_layer(self, train_step_outputs):
        """
        only logging mu for now
        """
        all_td_net_outs = [tso['td_net_outs'] for tso in train_step_outputs]
        td_net_count = len(all_td_net_outs[0])
        
        for t in range(td_net_count):
            mus = torch.cat([tdno[t]['mu_q'] for tdno in all_td_net_outs], dim=0).mean(0).tolist()
            mu_dict = {f"mu_q_layer_{t}/component_{i}": component_val for i, component_val in enumerate(mus)}            
            for k , v in mu_dict.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)

    def _log_per_layer_weights(self, train_step_outputs):
        
        T = len(self.model.top_down_networks)

        for t, td_net in enumerate(self.model.top_down_networks):
            print(f"Z{1+T-t}-to-Z{T-t}")

            full_mat = torchvision.utils.make_grid(td_net.W_input_to_interm)
            print(td_net.W_input_to_interm.shape, full_mat.shape)
            masked_mat = torchvision.utils.make_grid(td_net.W_input_to_interm.mul(td_net.mask_input_to_interm))
            print(td_net.mask_input_to_interm.shape, masked_mat.shape)
            
            full_and_masked_side_by_side = torch.cat([full_mat, masked_mat], dim = 2) 
            self.logger.experiment.add_image(f"Weights/Z{1+T-t}-to-Z{T-t}", full_and_masked_side_by_side, self.current_epoch)

