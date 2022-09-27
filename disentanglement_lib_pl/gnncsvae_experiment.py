import os
import torch

from base_vae_experiment import BaseVAEExperiment
from models.csvae_gnn import GNNBasedConceptStructuredVAE
from common import utils
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
            'current_epoch': self.current_epoch,
            'max_epochs': self.max_epochs
        })
        
        train_step_outputs = self.model.loss_function(**fwd_pass_results)

        # We need it for visualizing per layer mean / sigma components
        train_step_outputs.update({
            "prior_mu": fwd_pass_results['prior_mu'].detach(),
            "prior_logvar": fwd_pass_results['prior_logvar'].detach(),
            "posterior_mu": fwd_pass_results['posterior_mu'].detach(), 
            "posterior_logvar": fwd_pass_results['posterior_logvar'].detach(),
        })

        return train_step_outputs        
        
    def training_step_end(self, train_step_output):
        
        super(GNNCSVAEExperiment, self).training_step_end(train_step_output)

        # plot this to debug the dimensions tracking behaviour
        
        # Visualize Components of mean and sigma vector for every layer
        #self._log_mu_per_node([train_step_output], step_type='global')
        #self._log_logvar_per_node([train_step_output], step_type='global')
        
        # Add KLD Loss for every layer
        #self._log_kld_loss_per_node([train_step_output], step_type='global')


    def training_epoch_end(self, train_step_outputs):

        super(GNNCSVAEExperiment, self).training_epoch_end(train_step_outputs)

        torch.set_grad_enabled(False)
        self.model.eval()

        # Add KLD Loss for every layer
        self._log_kld_loss_per_node(train_step_outputs)

        # Visualize Components of mean and sigma vector for every layer
        self._log_mu_per_node(train_step_outputs)
        self._log_std_per_node(train_step_outputs)

        self._log_loss_func_weights(train_step_outputs)
        self._save_latent_space_plot()

        if self.model.add_classification_loss:
            self._log_classification_losses(train_step_outputs)
        
        if self.model.add_cov_loss:
            self._log_batch_covariance_losses(train_step_outputs)

        if self.model.controlled_capacity_increase:
            self.logger.experiment.add_scalar("Controlled_Capacity/Current_Capacity", self.model.current_capacity, self.global_step)
            self.logger.experiment.add_scalar("Controlled_Capacity/Max_Capacity", self.model.max_capacity, self.global_step)
        
        torch.set_grad_enabled(True)
        self.model.train()

    def _log_loss_func_weights(self, train_step_outputs):
        
        self.logger.experiment.add_scalar("LossTermWeights/w_recon", train_step_outputs[0]['output_aux'][0], self.current_epoch)
        self.logger.experiment.add_scalar("LossTermWeights/w_kld", train_step_outputs[0]['output_aux'][1], self.current_epoch)
        self.logger.experiment.add_scalar("LossTermWeights/w_sup_reg", train_step_outputs[0]['output_aux'][2], self.current_epoch)
        self.logger.experiment.add_scalar("LossTermWeights/w_cov_loss", train_step_outputs[0]['output_aux'][3], self.current_epoch)

    def _log_kld_loss_per_node(self, train_step_outputs, step_type='epoch'):
        
        step = self.current_epoch if step_type == 'epoch' else self.global_step

        all_loss_keys = train_step_outputs[0].keys()

        per_node_kld_keys = [key for key in all_loss_keys if 'KLD_z_' in key]
        
        for kld_loss_key in per_node_kld_keys:
            kld_loss = torch.stack([tso[kld_loss_key] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"KLD_Per_Node/{kld_loss_key}", kld_loss, step)

    def _log_mu_per_node(self, train_step_outputs, step_type='epoch'):
       
        step = self.current_epoch if step_type == 'epoch' else self.global_step

        post_mus = torch.cat([tso['posterior_mu'] for tso in train_step_outputs], dim=0)
        prior_mus = torch.cat([tso['prior_mu'] for tso in train_step_outputs], dim=0)

        post_mus_avgs = post_mus.mean(0).tolist()
        prior_mus_avgs = prior_mus.mean(0).tolist()

        # log posterior mus
        for node_idx in range(self.model.num_nodes):
            
            # Histograms
            # Loop over every dim of mu associated with this node and add its histogram
            for k in range(post_mus.shape[2]):
                self.logger.experiment.add_histogram(f"Node_{node_idx + 1}/Mu_q_Dim_{k}", post_mus[:, node_idx, k], step)
                #self.logger.experiment.add_histogram(f"Node_{node_idx + 1}/Mu_p_Dim_{k}", prior_mus[:, node_idx, k], step)
            
            # Scalars           
            # we do '+1' because latent indexing is 1-based, there is no Z_0
            post_mu_dict = {f"Node_{node_idx + 1}/Mu_q_comp_{i}": component_val for i, component_val in enumerate(post_mus_avgs[node_idx])}
            #print(post_mu_dict)
            for k , v in post_mu_dict.items():
                self.logger.experiment.add_scalar(k, v, step)
            
            #prior_mu_dict = {f"Node_{node_idx + 1}/Mu_p_comp_{i}": component_val for i, component_val in enumerate(prior_mus_avgs[node_idx])}            
            #for k , v in prior_mu_dict.items():
            #    self.logger.experiment.add_scalar(k, v, step)
            
        #TODO: log learnable prior of dept nodes
        for node_idx in range(self.model.num_dept_nodes):
            # Histograms
            # Loop over every dim and add its histogram
            for k in range(prior_mus.shape[2]):
                self.logger.experiment.add_histogram(f"Node_{node_idx + 1}/Mu_p_Dim_{k}", prior_mus[:, node_idx, k], step)
            
            prior_mu_dict = {f"Node_{node_idx + 1}/Mu_p_comp_{i}": component_val for i, component_val in enumerate(prior_mus_avgs[node_idx])}            
            for k , v in prior_mu_dict.items():
                self.logger.experiment.add_scalar(k, v, step)

    def _log_std_per_node(self, train_step_outputs, step_type='epoch'):

        step = self.current_epoch if step_type == 'epoch' else self.global_step

        # These should have the shape (batch, num_nodes, num_feat_dim)
        # std = exp( ln(Variance) / 2) = exp( log_var / 2)
        post_stds = torch.exp(torch.cat([tso['posterior_logvar'] for tso in train_step_outputs], dim=0) / 2.0)
        prior_stds = torch.exp(torch.cat([tso['prior_logvar'] for tso in train_step_outputs], dim=0) / 2.0)

        post_std_avgs = post_stds.mean(0).tolist()
        prior_std_avgs = prior_stds.mean(0).tolist()

        # log posterior stds
        for node_idx in range(self.model.num_nodes):
            
            # Histograms
            # Loop over every dim and add its histogram
            for k in range(post_stds.shape[2]):
                self.logger.experiment.add_histogram(f"Node_{node_idx + 1}/Std_q_Dim_{k}", post_stds[:, node_idx, k], step)
                #self.logger.experiment.add_histogram(f"Node_{node_idx + 1}/Std_p_Dim_{k}", prior_stds[:, node_idx, k], step)
            
            # Scalars
            # we do '+1' because latent indexing is 1-based, there is no Z_0
            post_std_dict = {f"Node_{node_idx + 1}/Std_q_comp_{i}": component_val for i, component_val in enumerate(post_std_avgs[node_idx])}            
            for k , v in post_std_dict.items():
                self.logger.experiment.add_scalar(k, v, step)

            #prior_std_dict = {f"Node_{node_idx + 1}/Std_p_comp_{i}": component_val for i, component_val in enumerate(prior_std_avgs[node_idx])}            
            #for k , v in prior_std_dict.items():
            #    self.logger.experiment.add_scalar(k, v, step)
        
        #TODO: log learnable prior of dept nodes
        for node_idx in range(self.model.num_dept_nodes):
            # Histograms
            # Loop over every dim and add its histogram
            for k in range(prior_stds.shape[2]):
                self.logger.experiment.add_histogram(f"Node_{node_idx + 1}/Std_p_Dim_{k}", prior_stds[:, node_idx, k], step)
            
            prior_std_dict = {f"Node_{node_idx + 1}/Std_p_comp_{i}": component_val for i, component_val in enumerate(prior_std_avgs[node_idx])}            
            for k , v in prior_std_dict.items():
                self.logger.experiment.add_scalar(k, v, step)
       
    def _log_classification_losses(self, train_step_outputs):
        
        print("adding clf plots")
        # Log total sup. reg. loss        
        clf_loss = torch.stack([tso[c.AUX_CLASSIFICATION] for tso in train_step_outputs]).mean()
        self.logger.experiment.add_scalar(f"SupReg/Total_Loss", clf_loss, self.current_epoch)

        # Log per node sup. reg. loss
        for node_idx in range(self.model.num_dept_nodes):
            node_name = self.model.node_labels[node_idx]
            clf_loss_node = torch.stack([tso[f'clf_{node_name}'] for tso in train_step_outputs]).mean()
            self.logger.experiment.add_scalar(f"SupReg/clf_{node_name}", clf_loss_node, self.current_epoch)

    def _log_batch_covariance_losses(self, train_step_outputs):
        print("adding cov plots")
        cov_loss = torch.stack([tso['covariance_loss'] for tso in train_step_outputs]).mean()
        self.logger.experiment.add_scalar(f"SupReg/Covariance_Loss", cov_loss, self.current_epoch)

    def _save_latent_space_plot(self, num_batches = 200):

        #assert self.model.z_dim == 2, f"_save_2D_latent_space_plot() expects 2D latent space and you have {self.model.z_dim}d"
        
        # TODO: make this function more general s.t. it works for all datasets

        # get latent activations for the given number of batches
        current_device = next(self.model.parameters()).device
        num_batches = len(self.sample_loader) if self.params['dataset'] in ["pendulum", "flow"] else num_batches
        hue_factors = ['theta', 'phi', 'shade','mid']
        padded_epoch = str(self.current_epoch).zfill(len(str(self.max_epochs)))
        
        from common import notebook_utils

        mus, labels = notebook_utils.csvaegnn_get_latent_activations_with_labels_for_scatter(
            self.model,
            self.sample_loader,
            current_device,
            num_batches
        )
        
        # for each node and hue combination, plot and save
        for node_idx in range(self.model.num_dept_nodes):
            for hue_idx, hue_factor in enumerate(hue_factors):
                
                # need to do this to make sure gifs are properly sequenced other wise 1.jpg is followed by 10.jpg, and 100.jpg
                
                image_file_name = f"latentspace.node.{node_idx}.hue.{hue_idx}.epoch.{padded_epoch}.jpg"
                
                latentspace_images_path = os.path.join(
                    self.logger.log_dir, 
                    "latent_space_plots", 
                    image_file_name
                )

                utils.plot_1d_latent_space(mus, labels, hue_factors, node_idx, 
                                        hue_idx, hue_idx, self.current_epoch, latentspace_images_path)

        # pairwise activation plot of each node 
        # this can become incomprehensible if there are too many nodes
        image_file_name = f"posterior_plot.epoch.{padded_epoch}.jpg"
        pairplot_images_path = os.path.join(
                    self.logger.log_dir, 
                    "latent_space_plots", 
                    image_file_name
                )
    
        utils.pairwise_node_activation_plots(mus, padded_epoch, pairplot_images_path)
        
        if self.model.prior_type == "gt_based_learnable":
            prior_mus, _ = notebook_utils.get_prior_mus_given_gt_labels(
                self.model,
                self.sample_loader,
                current_device,
                num_batches
            )
            image_file_name = f"prior_plot.epoch.{padded_epoch}.jpg"
            pairplot_images_path = os.path.join(
                        self.logger.log_dir, 
                        "latent_space_plots", 
                        image_file_name
                    )
            
            utils.pairwise_node_activation_plots(prior_mus, padded_epoch, pairplot_images_path)
