import visdom
import torch
import torchvision
import matplotlib.pyplot as plt

class VisdomVisualiser:

    def __init__(self, params):
        
        visual_args = params['visual_args'] 
        visdom_args = params['visdom_args']

        self.port = visdom_args['port'] if "port" in visdom_args.keys() else 8097
        self.logfile = visdom_args['logfile'] if "logfile" in visdom_args.keys() else None
        self.name = "Visdom on {}".format(visual_args['dataset'])
        
        self.visdom_instance = visdom.Visdom(port=self.port, log_to_filename=self.logfile)
        
        # ['recon_loss', 'total_loss', 'kld_loss', 'mu', 'var' etc.] 
        self.scalar_window_names = visual_args['scalar_metrics']
        self.scalar_windows = self._initialize_scalar_windows()
        self.visualize_experiment_metadata(params)

        # ['beta-vae', 'factorvae' etc.] 
        self.disent_metrics_names = visual_args['disent_metrics']
        self.disent_windows = self._initialize_disent_metrics_windows()
        self.multidim_windows = dict(
            mu_batch=None,
            logvar_batch=None
        )

    def _initialize_scalar_windows(self):
        
        if self.scalar_window_names is None:
            return None
        
        else:
            _scalar_windows = dict()
            for win in self.scalar_window_names:
                _scalar_windows[win] = None
            
            return _scalar_windows

    def _initialize_disent_metrics_windows(self):
        
        if self.disent_metrics_names is None:
            return None
        
        else:
            _disent_windows = dict()
            
            for win in self.disent_metrics_names:
                _disent_windows[win] = None

            return _disent_windows

    def visualize_reconstruction(self, x_inputs, x_recons, global_step):
        
        # input is BCHW
        # we want to see image and its reconstruction side-by-side, and not the 
        # images and recons grid side-by-side
        inputs_and_reconds_side_by_side = torch.cat([x_inputs, x_recons], dim = 3)
        img_input_vs_recon = torchvision.utils.make_grid(inputs_and_reconds_side_by_side, normalize=True)

        # x_inputs = torchvision.utils.make_grid(x_inputs, normalize=True)
        # x_recons = torchvision.utils.make_grid(x_recons, normalize=True)
        # img_input_vs_recon = torch.cat([x_inputs, x_recons], dim=2).cpu()

        self.visdom_instance.images(img_input_vs_recon,
                                    env=self.name + '_reconstruction',
                                    opts=dict(title="Recon at step {}".format(global_step)),
                                    nrow=10)

    def visualize_scalar_metrics(self, new_scalar_metric_values, global_step):
        """
        All passed in tensors should have been .cpu()'d beforehand
        Can be used to plot things like :
        recon_losses, mus, vars, dim_wise_klds, mean_klds, total_loss, kld_loss 
        """
        
        iters = torch.Tensor([global_step])#.unsqueeze(0)

        # legend = []
        # for z_j in range(new_scalar_metric_values['mus'].size()[0]):
        #    legend.append('z_{}'.format(z_j))
        
        #legend.append('mean')
        #legend.append('total')

        window_titles_and_values = {
            'recon': {'title': 'Reconsturction Loss', 'legend': None},
            'kld_loss': {'title': 'KL Divergence (mean)', 'legend': None},
            'loss': {'title': 'Total Loss', 'legend': None},
            'mu': {'title': 'Posterior Mean', 'legend': None},
            'var': {'title': 'Posterior Variance', 'legend': None}
        }

        # Update (or create, if non-existent) the scalar windows
        for win in self.scalar_window_names:

            if self.scalar_windows[win] is None:
                self.scalar_windows[win] = self.visdom_instance.line(
                    X=iters,
                    Y=new_scalar_metric_values[win].cpu().unsqueeze(0),
                    env=self.name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title=window_titles_and_values[win]['title']))
            else:
                self.scalar_windows[win] = self.visdom_instance.line(
                    X=iters,
                    Y=new_scalar_metric_values[win].cpu().unsqueeze(0),
                    env=self.name + '_lines',
                    win=self.scalar_windows[win],
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title=window_titles_and_values[win]['title']))

    def visualize_disentanglement_metrics(self, new_disent_metrics, global_step):

        window_titles_and_values = dict()
        iters = torch.Tensor([global_step])
        
        for eval_metric_name, eval_metric_value in new_disent_metrics.items():
            print(eval_metric_name, eval_metric_value)
            metric_name = eval_metric_name.replace("eval_", "")
            window_titles_and_values[metric_name] = {
                'title': metric_name,
                'value': torch.Tensor([eval_metric_value]).unsqueeze(0),
                'legend': None
            }

        for win in self.disent_metrics_names:

            if self.disent_windows[win] is None:
                self.disent_windows[win] = self.visdom_instance.line(
                    X=iters,
                    Y=window_titles_and_values[win]['value'],
                    env=self.name + '_disent_eval',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title=window_titles_and_values[win]['title']))
            else:
                self.disent_windows[win] = self.visdom_instance.line(
                    X=iters,
                    Y=window_titles_and_values[win]['value'],
                    env=self.name + '_disent_eval',
                    win=self.disent_windows[win],
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title=window_titles_and_values[win]['title']))

    def visualize_experiment_metadata(self, expr_config):
        
        metadata = ""
        for param, value in expr_config.items():
            if not type(value) in [dict, list]:
                metadata += f"<b> {param} </b> = {str(value)} <br /> "

        self.visdom_instance.text(metadata, win='metadata', env=self.name + '_lines')
    
    def visualize_multidim_metrics(self, multidim_metrics, global_step):
        """
        Mainly used to plot \mu and \Sigma for each dimension on one plot
        """
        assert "mu_batch" in multidim_metrics.keys()
        assert "logvar_batch" in multidim_metrics.keys()

        iters = torch.Tensor([global_step])

        print("in mdm()")
        print(multidim_metrics['mu_batch'].shape)
        print(multidim_metrics['logvar_batch'].shape)
        
        for mdm in ['mu_batch', 'logvar_batch']:
            
            if self.multidim_windows[mdm] is None:
                self.multidim_windows[mdm] = self.visdom_instance.line(
                    X=iters,
                    Y=multidim_metrics[mdm].mean(0).mean(0, keepdim=True),
                    env=self.name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title=mdm))
            else:
                self.multidim_windows[mdm] = self.visdom_instance.line(
                    X=iters,
                    Y=multidim_metrics[mdm].mean(0).mean(0, keepdim=True),
                    env=self.name + '_lines',
                    win=self.multidim_windows[mdm],
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title=mdm))
