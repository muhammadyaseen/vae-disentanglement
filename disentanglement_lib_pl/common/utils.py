import logging
import argparse
import subprocess
import scipy.linalg as linalg
import numpy as np
import random
from importlib import reload
import os

import torch.nn
import torch.nn.init as init
from torch.autograd import Variable

from common import constants as c


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


def str2bool(v):
    """
    Thank to stackoverflow user: Maxim
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    :param v: A command line argument with values [yes, true, t, y, 1, True, no, false, f, n, 0, False]
    :return: Boolean version of the command line argument
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def grid2gif(image_str, output_gif, delay=100):
    """
    Makes an animated GIF from input images.
    Thanks to the stackoverflow user: pkjain (https://stackoverflow.com/users/5241303/pkjain)

    :param image_str: The wildcard path to all images whose names are numerically sorted
    :param output_gif: Path to the generated GIF
    :param delay: Delay to introduce in-between showing consecutive images of a GIF
    """

    str1 = 'convert -delay {} -loop 0 {} {}'.format(delay, image_str, output_gif)
    subprocess.call(str1, shell=True)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : mu of dist 1
    -- mu2   : mu of dist 2
    -- sigma1: The covariance matrix of dist 1
    -- sigma2: The covariance matrix of dist 2
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class LinearScheduler:
    def __init__(self, start_value, target_value=None, epochs=None):
        self.start_value = start_value
        self.target_value = target_value
        assert start_value != target_value, 'start_value and target_value should be different'
        self.mode = min if target_value > start_value else max
        self.per_step = (target_value - start_value) / epochs

    def step(self, step_num):
        return self.mode(self.start_value + step_num * self.per_step, self.target_value)


def get_data_for_visualization(dataset, device):
    def random_idx():
        return random.randint(0, dataset.__len__() - 1)

    sample_idx = {}
    dset_name = dataset.name
    if dset_name == c.DATASETS[1]:  # dsprites_full
        fixed_idx = [87040, 332800, 578560]  # square ellipse heart
        sample_idx = {'{}_{}'.format(c.FIXED, c.SQUARE): fixed_idx[0],
                      '{}_{}'.format(c.FIXED, c.ELLIPSE): fixed_idx[1],
                      '{}_{}'.format(c.FIXED, c.HEART): fixed_idx[2]
                      }

    elif dset_name == c.DATASETS[0]:  # celebA
        fixed_idx = [11281, 114307, 10535, 59434]
        sample_idx = {}
        for i in range(len(fixed_idx)):
            sample_idx.update({c.FIXED + '_' + str(i): fixed_idx[i]})

    else:
        for i in range(3):
            sample_idx.update({'rand' + str(i): random_idx()})

    # add a random sample to all
    sample_idx.update({c.RANDOM: random_idx()})

    images = {}
    labels = {}
    for key, idx in sample_idx.items():
        try:
            data = dataset.__getitem__(idx)
        except IndexError:
            data = dataset.__getitem__(random_idx())

        images[key] = data[0].to(device).unsqueeze(0)

        labels[key] = None
        if dataset.has_labels():
            labels[key] = data[1].to(device, dtype=torch.long).unsqueeze(0)

    return images, labels


def prepare_data_for_visualization(data):
    sample_images, sample_labels = data
    sample_images_dict = {}
    sample_labels_dict = {}
    for i, img in enumerate(sample_images):
        sample_images_dict.update({str(i): img})
        if sample_labels is not None:
            sample_labels_dict.update({str(i): sample_labels[i].to(dtype=torch.long).unsqueeze(0)})
        else:
            sample_labels_dict.update({str(i): None})

    return sample_images_dict, sample_labels_dict


class StyleFormatter(logging.Formatter):
    CSI = "\x1B["
    YELLOW = '33;40m'
    RED = '31;40m'

    # Add %(asctime)s after [ to include the time-date of the log
    high_style = '{}{}(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, RED, CSI)
    medium_style = '{}{}(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, YELLOW, CSI)
    low_style = '(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s'

    def __init__(self, fmt=None, datefmt='%b-%d %H:%M', style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        if record.levelno <= logging.INFO:
            self._style = logging.PercentStyle(StyleFormatter.low_style)
        elif record.levelno <= logging.WARNING:
            self._style = logging.PercentStyle(StyleFormatter.medium_style)
        else:
            self._style = logging.PercentStyle(StyleFormatter.high_style)

        return logging.Formatter.format(self, record)


def _init_layer(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight.data)


def init_layers(modules):
    for block in modules:
        from collections.abc import Iterable
        if isinstance(modules[block], Iterable):
            for m in modules[block]:
                _init_layer(m)
        else:
            _init_layer(modules[block])


@static_var("Ys", dict())
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N, NumLabels].
      num_classes: list of int for number of classes for each label, or single label (int)

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    if isinstance(num_classes, int):
        num_classes = [num_classes]

    one_hots = []
    for i in range(len(num_classes)):
        num_class = num_classes[i]

        if num_class not in one_hot_embedding.Ys:
            one_hot_embedding.Ys[num_class] = cuda(torch.eye(num_class))

        y = one_hot_embedding.Ys[num_class]
        one_hots.append(y[labels[:, i]])

    return torch.cat(one_hots, dim=1)


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = float(v)
            except ValueError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def get_scheduler(value, scheduler_type, scheduler_args):
    if scheduler_type is None:
        return
    if isinstance(value, torch.optim.Optimizer):
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_class(value, **scheduler_args)
    else:
        if scheduler_type == 'LinearScheduler':
            scheduler = LinearScheduler(value, **scheduler_args)
        else:
            raise NotImplementedError

    return scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def setup_logging(verbose):
    # verbosity
    reload(logging)  # to turn off any changes to logging done by other imported libraries
    h = logging.StreamHandler()
    h.setFormatter(StyleFormatter())
    h.setLevel(0)
    logging.root.addHandler(h)
    logging.root.setLevel(verbose)


def initialize_seeds(seed):
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def set_environment_variables(dset_dir, dset_name):
    """
    If the argument dset_dir is set, overwrite DISENTANGLEMENT_LIB_DATA.
    else if only $DATASETS is set, use the same for $DISENTANGLEMENT_LIB_DATA
    else if only $DISENTANGLEMENT_LIB_DATA is set, use the same for $DATASETS
    else print warning that the environment variables are not set or inconsistent.

    If the argument dset_name is set, overwrite $DATASET_NAME.
    else if only $DATASET_NAME is set, use the same for $AICROWD_DATASET_NAME
    else if only $AICROWD_DATASET_NAME is set, use the same for $DATASET_NAME
    else print warning that the environment variables are not set or inconsistent.

    :param dset_dir: directory where all the datasets are saved
    :param dset_name: name of the dataset to be loaded by the dataloader
    """
    if dset_dir:
        os.environ['DISENTANGLEMENT_LIB_DATA'] = dset_dir
    if not os.environ.get('DISENTANGLEMENT_LIB_DATA'):
        logging.warning(f"Environment variables are not correctly set:\n"
                        f"$DISENTANGLEMENT_LIB_DATA={os.environ.get('DISENTANGLEMENT_LIB_DATA')}\n")

    if dset_name:
        os.environ['DATASET_NAME'] = dset_name
    if os.environ.get('DATASET_NAME') and not os.environ.get('AICROWD_DATASET_NAME'):
        os.environ['AICROWD_DATASET_NAME'] = os.getenv('DATASET_NAME')
    elif os.environ.get('AICROWD_DATASET_NAME') and not os.environ.get('DATASET_NAME'):
        os.environ['DATASET_NAME'] = os.getenv('AICROWD_DATASET_NAME')
    elif os.environ.get('AICROWD_DATASET_NAME') != os.environ.get('DATASET_NAME'):
        logging.warning(f"Environment variables are not correctly set:\n"
                        f"$AICROWD_DATASET_NAME={os.environ.get('AICROWD_DATASET_NAME')}\n"
                        f"$DATASET_NAME={os.environ.get('DATASET_NAME')}")

    logging.info(f"$AICROWD_DATASET_NAME={os.environ.get('AICROWD_DATASET_NAME')}")
    logging.info(f"$DATASET_NAME={os.environ.get('DATASET_NAME')}")
    logging.info(f"$DISENTANGLEMENT_LIB_DATA={os.environ.get('DISENTANGLEMENT_LIB_DATA')}")


def make_dirs(args):
    # makedirs
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)

class VisdomDataGatherer(object):

    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):

        return dict(iter=[],
                    recon_loss=[],
                    kld_loss=[],
                    total_loss=[],
                    mu=[],
                    var=[],
                    input_images=[],
                    recon_images=[])

    def insert(self, **kwargs):

        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

def visdom_visualize_reconstruction(self):

    self.net_mode(train=False)

    to_show = min(self.batch_size, 64)
    x_inputs = torchvision.utils.make_grid(self.visdom_gatherer.data['input_images'][:to_show],
                                    normalize=True)
    x_recons = torchvision.utils.make_grid(self.visdom_gatherer.data['recon_images'][:to_show],
                                            normalize=True)

    #white_line = torch.ones((3, x_inputs.size(1), 10)).to(self.device)
    img_input_vs_recon = torch.cat([x_inputs, x_recons], dim=3).cpu()
    #img_input_vs_recon = torch.stack([x_inputs, x_recons], dim=0).cpu()

    self.visdom_instance.images(img_input_vs_recon,
                                env=self.viz_name + '_reconstruction',
                                opts=dict(title="Recon at {} iter".format(self.iter)),
                                nrow=10)

    self.net_mode(train=True)

def visdom_visualize_traverse(self, limit: tuple, spacing, data=None, test=False):

    self.net_mode(train=False)
    interp_values = torch.arange(limit[0], limit[1], spacing)
    num_cols = interp_values.size(0)

    if data is None:
        sample_images_dict, sample_labels_dict = get_data_for_visualization(self.data_loader.dataset, self.device)
    else:
        sample_images_dict, sample_labels_dict = prepare_data_for_visualization(data)

    encodings = dict()
    for key in sample_images_dict.keys():
        encodings[key] = self.encode_deterministic(images=sample_images_dict[key], labels=sample_labels_dict[key])

    gifs = []
    for key in encodings:
        latent_orig = encodings[key]
        label_orig = sample_labels_dict[key]
        logging.debug('latent_orig: {}, label_orig: {}'.format(latent_orig, label_orig))
        samples = []

        # encode original on the first row
        sample = self.decode(latent=latent_orig.detach(), labels=label_orig)
        for _ in interp_values:
            samples.append(sample)

        # traverse latents (deleted traverse c and l for now)
        if self.traverse_z:
            for zid in range(self.z_dim):
                for val in interp_values:
                    latent = latent_orig.clone()
                    latent[:, zid] = val
                    self.set_z(latent, zid, val)
                    sample = self.decode(latent=latent, labels=label_orig)

                    samples.append(sample)
                    gifs.append(sample)

        samples = torch.cat(samples, dim=0).cpu()
        samples = torchvision.utils.make_grid(samples, nrow=num_cols)
        title = '{}_latent_traversal(iter:{})'.format(key, self.iter)
        self.visdom_instance.images(samples,
                                    env=self.viz_name + '_traverse',
                                    opts=dict(title="Traverse at {} iter".format(self.iter)),
                                    nrow=len(interp_values))

def visdom_visualize_scalar_metrics(self):

    self.net_mode(train=False)

    recon_losses = torch.stack(self.visdom_gatherer.data['recon_loss']).cpu()
    mus = torch.stack(self.visdom_gatherer.data['mu']).cpu()
    vars = torch.stack(self.visdom_gatherer.data['var']).cpu()

    #dim_wise_klds = torch.stack(self.visdom_gatherer.data['dim_wise_kld'])
    #mean_klds = torch.stack(self.visdom_gatherer.data['mean_kld'])
    total_loss = torch.stack(self.visdom_gatherer.data['total_loss'])
    kld_loss = torch.stack(self.visdom_gatherer.data['kld_loss'])
    #klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
    iters = torch.Tensor(self.visdom_gatherer.data['iter'])
    evaluation_metrics_results = self.evaluate_results

    legend = []
    for z_j in range(self.z_dim):
        legend.append('z_{}'.format(z_j))
    #legend.append('mean')
    #legend.append('total')

    window_titles_and_values = {
        'recon_loss': {'title': 'Reconsturction Loss', 'value': recon_losses, 'legend': None},
        'kld_loss': {'title': 'KL Divergence (mean)', 'value' : kld_loss, 'legend': None},
        'total_loss': {'title': 'Total Loss', 'value': total_loss, 'legend': None},
        'mu': {'title': 'Posterior Mean', 'value': mus, 'legend': legend},
        'var': {'title': 'Posterior Variance', 'value': vars, 'legend': legend}
    }

    for eval_metric_name, eval_metric_value in evaluation_metrics_results.items():
        metric_name = eval_metric_name.replace("eval_", "")
        window_titles_and_values[metric_name] = {
            'title': metric_name,
            'value': torch.Tensor([eval_metric_value]),
            'legend': None
        }

    # Update (or create, if non-existent) the scalar windows
    for win in self.scalar_windows:
        print(window_titles_and_values[win]['value'])
        if self.visdom_scalar_windows[win] is None:
            self.visdom_scalar_windows[win] = self.visdom_instance.line(
                X=iters,
                Y=window_titles_and_values[win]['value'],
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    #legend=window_titles_and_values[win]['legend'],
                    xlabel='iteration',
                    title=window_titles_and_values[win]['title'], ))
        else:
            self.visdom_scalar_windows[win] = self.visdom_instance.line(
                X=iters,
                Y=window_titles_and_values[win]['value'],
                env=self.viz_name + '_lines',
                win=self.visdom_scalar_windows[win],
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    #legend=window_titles_and_values[win]['legend'],
                    xlabel='iteration',
                    title=window_titles_and_values[win]['title'],))

def update_visdom_visualisations(self, x_inputs, x_recons,
                                    mu_batch, logvar_batch,
                                    losses):

    self.visdom_gatherer.insert(iter=self.iter,
                                mu=mu_batch.mean(0).data,
                                var=logvar_batch.exp().mean(0).data,
                                recon_loss=losses[c.RECON],
                                kld_loss=losses['kld'],
                                total_loss=losses[c.TOTAL_VAE])

    self.visdom_gatherer.insert(input_images=x_inputs.data)
    self.visdom_gatherer.insert(recon_images=x_recons.data)

    self.visdom_visualize_reconstruction()
    self.visdom_visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing)
    self.visdom_visualize_scalar_metrics()

    # clean up data for next iter
    self.visdom_gatherer.flush()

    self.net_mode(train=True)

def _log_sampled_images(sampled_images, global_step): # should be .cpu().data'ed

        # samples = self.model.sample(36, self.curr_device)
        grid_of_samples = vutils.make_grid(sampled_images, normalize=True, nrow=12, value_range=(0.0,1.0))
        self.logger.experiment.add_image("Sampled Images", grid_of_samples, global_step=global_step)

def _log_reconstructed_images(global_step):

    # Get sample reconstruction image
    test_input, test_label = next(iter(self.sample_dataloader))
    test_input = test_input.to(self.curr_device)
    print("curr dev:", self.curr_device)
    recons = self.model.generate(test_input, labels = test_label)
    recons_grid = vutils.make_grid(recons.cpu().data, normalize=True, nrow=12, value_range=(0.0,1.0))
    self.logger.experiment.add_image("Reconstructed Images", recons_grid,
                                        global_step=global_step)

    del test_input, test_label, recons

def _log_latent_layer_activations(global_step):

    # TODO: probably we should save hist over WHOLE val dataset and
    # not just a single batch

    test_input, _ = next(iter(self.sample_dataloader))
    test_input = test_input.to(self.curr_device)
    activations_mu, activations_logvar = self.model.encode(test_input)
    self.logger.experiment.add_histogram("Latent Activations", activations_mu, global_step)