import os
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as T
from torchvision import transforms
from torch.utils.data import DataLoader

from bvae_experiment import BVAEExperiment
from common.ops import reparametrize
from laddervae_experiment import LadderVAEExperiment

from common.data_loader import CustomImageFolder
from common.known_datasets import DSpritesDataset, CorrelatedDSpritesDataset, ThreeShapesDataset, ContinumDataset 
from common import utils

from experiment_runner import get_dataset_specific_params

ModelParams = namedtuple('ModelParams', ["z_dim", "l_dim", "num_labels" , "in_channels", 
                                        "image_size", "batch_size", "w_recon", "w_kld", 
                                         "controlled_capacity_increase", "max_c", "iterations_c",
                                        "w_tc", "w_infovae", "w_dipvae", "lambda_od", "lambda_d_factor",
                                        "encoder", "decoder", "loss_terms", "alg", "l_zero_reg"])

CmdLineArgsProxy = namedtuple('DatasetParams', ["dset_name", "correlation_strength"])

"""
Start: dsprites notebook functions
These functions were orignally part of dsprites loading notebook file but 
have been adapted to our setup
"""

def load_dsprites(npz_path):
    """
    npz_path: string like "../datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    """

    dataset_zip = np.load(npz_path, allow_pickle=True, encoding='latin1')
    metadata = dataset_zip['metadata'][()]

    return {
        'images': dataset_zip['imgs'],
        'latent_values': dataset_zip['latents_values'],
        'latent_classes': dataset_zip['latents_classes'],
        'metadata': metadata,
        'latents_sizes': metadata['latents_sizes'],
        'latents_bases': np.concatenate((metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1,])))
    }

def latent_to_index(latents, latents_bases):
    
    # when indexing we have to fix color=0 because of how they generated the dataset
    latents[:,0] = np.zeros(len(latents)) 
    return np.dot(latents, latents_bases).astype(int)

def sample_latent(how_many, latents_sizes, correlated=False, 
                    in_idx=None, out_idx=None, map_fn=None):
    
    samples = np.zeros((how_many, latents_sizes.size))
    
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=how_many)

    if correlated:
        samples[:, out_idx] = map_fn(samples[:,in_idx])
    
    return samples

def show_images_grid(imgs_, num_images=25):
    
    from matplotlib import cm

    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    norm = cm.colors.Normalize(vmax=1.0, vmin=0.0)
    
    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='gray',  norm=norm)#, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.axis('off')

"""
End: dsprites notebook functions
"""
def show_image_grid_pt(imgs, **kwargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, **kwargs)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return fig

def __handle_celeba(dset_dir):

    # TODO: this is quick and dirty fix. need to better handle it
    labels = None
    label_weights = None
    label_idx = None
    label_names = None
    class_values = None

    root = os.path.join(dset_dir, 'celeba')
    labels_file = os.path.join(root, 'list_attr_celeba.csv')

    # celebA images are properly numbered, so the order should remain intact in loading
    labels = None
    if label_names is not None:
        labels = []
        labels_all = np.genfromtxt(labels_file, delimiter=',', names=True)
        for label_name in label_names:
            labels.append(labels_all[label_name])
        labels = np.array(labels).transpose()
    elif label_idx is not None:
        labels_all = np.genfromtxt(labels_file, delimiter=',', skip_header=True)
        labels = labels_all[:, label_idx]

    if labels is not None:
        # celebA labels are all binary with values -1 and +1
        labels[labels == -1] = 0
        from pathlib import Path
        num_l = labels.shape[0]
        num_i = len(list(Path(root).glob('**/*.jpg')))
        assert num_i == num_l, 'num_images ({}) != num_labels ({})'.format(num_i, num_l)

        # calculate weight adversely proportional to each class's population
        num_labels = labels.shape[1]
        label_weights = []
        for i in range(num_labels):
            ones = labels[:, i].sum()
            prob_one = ones / labels.shape[0]
            label_weights.append([prob_one, 1 - prob_one])
        label_weights = np.array(label_weights)

        # all labels in celebA are binary
        class_values = [[0, 1]] * num_labels

    data_kwargs = {'root': root,
                    'labels': labels,
                    'label_weights': label_weights,
                    'class_values': class_values,
                    'num_channels': 3,
                    'name': 'celeba',
                    'seed': 123,
                    'transforms': transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor()])}

    return CustomImageFolder(**data_kwargs)

def __handle_pendulum(dset_dir, dset_name):
    
    root = os.path.join(dset_dir, dset_name)
    labels_file = os.path.join(root, 'pendulum_labels.csv')
    labels_all = np.genfromtxt(labels_file, delimiter=',', names=True)
    data_kwargs = {'root': root,
                'labels': labels_all,
                'label_weights': [],
                'class_values': [],
                'num_channels': 3,
                'name': dset_name,
                'seed': 123,
                'transforms': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()]),
                'dtype': torch.float}

    return CustomImageFolder(**data_kwargs)

def get_configured_dataset(dset_name):
    
    dataset = None

    if dset_name == 'dsprites_full':
        dataset = DSpritesDataset(root="../datasets/dsprites/", split="train", transforms=transforms.ToTensor())
    elif dset_name == 'dsprites_correlated':
        dataset = CorrelatedDSpritesDataset(correlation_strength=0.2, split="train", seed=123)
    elif dset_name == 'dsprites_colored':
        dataset = DSpritesDataset(root="../datasets/dsprites/", split="train", transforms=transforms.ToTensor(), 
        correlated=True, colored=True)
    elif dset_name == 'dsprites_cond':
        dataset = DSpritesDataset(root="../datasets/dsprites/", split="train", transforms=transforms.ToTensor(), 
        correlated=False, colored=False, conditioned=True)

    elif dset_name == 'threeshapesnoisy':
        dataset = ThreeShapesDataset(root="../datasets/threeshapesnoisy/", split="train", transforms=transforms.ToTensor())
    elif dset_name == 'threeshapes':
        dataset = ThreeShapesDataset(root="../datasets/threeshapes/", split="train", transforms=transforms.ToTensor())
    elif dset_name == 'celeba':
        dataset = __handle_celeba("../datasets/")
    elif dset_name in ['pendulum','pendulum_switch']:
        dataset = __handle_pendulum("../datasets/", dset_name)
    else:
        raise NotImplementedError

    return dataset

def load_vae_model(algo_name, algo_type, checkpoint_path, curr_dev, 
                   model_params, exp_params):
    
    assert algo_type in ['bvae', 'laddervae']

    import models
    
    cmdline_args = CmdLineArgsProxy("dsprites_correlated", 0.2)
    dataset_params = get_dataset_specific_params(cmdline_args)
    vae_model_class = getattr(models, algo_name)
    vae_model = vae_model_class(model_params)
        
    vae_experiment = None

    if algo_type == 'bvae':
        vae_experiment = BVAEExperiment.load_from_checkpoint(
            checkpoint_path,
            map_location=curr_dev,
            vae_model=vae_model, 
            params=exp_params,
            dataset_params=dataset_params)
    
    if algo_type == 'laddervae':
        vae_experiment = LadderVAEExperiment.load_from_checkpoint(
            checkpoint_path,
            map_location=curr_dev,
            vae_model=vae_model, 
            params=exp_params,
            dataset_params=dataset_params)

    vae_experiment = vae_experiment.to(curr_dev)

    return vae_experiment

def get_latent_activations(dataloader, vae_model, curr_dev, batch_size = 32, batches = None, 
                                    model_type='bvae'):
   
    assert model_type in ['bvae', 'laddervae']
    #assert latent_layer in ['z1', 'z2']

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    
    latent_acts = None
    
    if model_type == 'bvae':
        latent_acts = []
        for _ in range(vae_model.model.z_dim):
            latent_acts.append([])
    
    if model_type == 'laddervae':
        latent_acts = {'z1': [], 'z2': []}
    
        for _ in range(vae_model.model.z1_dim):
            latent_acts['z1'].append([])
        for _ in range(vae_model.model.z2_dim):
            latent_acts['z2'].append([])
    
    batches_processed = 0

    with torch.no_grad():

        for x_batch, _ in dataloader:
            
            if batches is not None and batches_processed >= batches:
                break
            
            # First we encode this batch
            x_batch = x_batch.to(curr_dev)
            
            # -- For Single Latent layer models
            mu_batch = None
            
            if model_type == 'bvae':
                
                # Then, we get \mu and \sigma for this batch
                mu_batch, _ = vae_model.model.encode(x_batch)
                mu_batch = mu_batch.detach().cpu().numpy()

                for b in range(batch_size):
                    mu_of_this_x = mu_batch[b]              
                    # for reach dimension
                    for m, m_dim in enumerate(mu_of_this_x): 
                        latent_acts[m].append(m_dim.item())

            # -- For Multiple Latent layer models
            z1_batch, z2_batch = None, None

            if model_type == 'laddervae':
            
                z1_batch, z2_batch, _ = vae_model.model.encode(x_batch)
                
                z1_batch = z1_batch.detach().cpu().numpy()
                z2_batch = z2_batch.detach().cpu().numpy()
            
                for b in range(batch_size):
                    
                    z1_of_this_x = z1_batch[b]
                    z2_of_this_x = z2_batch[b]
                    
                    # for reach dimension
                    for m, m_dim in enumerate(z1_of_this_x): 
                        latent_acts['z1'][m].append(m_dim.item())
                    
                    for m, m_dim in enumerate(z2_of_this_x): 
                        latent_acts['z2'][m].append(m_dim.item())

            batches_processed += 1
    
    return latent_acts

def plot_latent_dist(activations, label="Factor X"):

    fig, axes = plt.subplots(1)
    axes.hist(activations, label=label, align='mid')
    axes.legend(prop={'size': 10},loc='upper left')

def get_latent_activations_with_labels(dataloader, vae_model, curr_dev, batch_size = 32, 
                                    batches = None, group_by=False, model_type='bvae', latent_layer='z1'):
   
    assert model_type in ['bvae', 'laddervae']
    assert latent_layer in ['z1', 'z2']

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    label_and_latent_act_dict = defaultdict(list)
    
    batches_processed = 0
    with torch.no_grad():
        for x_batch, label_batch in dataloader:
            
            if batches is not None and batches_processed >= batches:
                break
            
            # First we encode this batch and get \mu and \sigma
            x_batch = x_batch.to(curr_dev)
            
            mu_batch = None
            
            if model_type == 'bvae':
                mu_batch, _ = vae_model.model.encode(x_batch)
            
            if model_type == 'laddervae':
                z1, z2, _ = vae_model.model.encode(x_batch)
                mu_batch = z1 if latent_layer == 'z1' else z2
            
            # convert to numpy format so that we can easily use in matplotlib
            mu_batch = mu_batch.detach().cpu().numpy()
            
            label_batch = label_batch.cpu().numpy()
            
            # using labels, we place all \mu's belonging to same class together
            if group_by:
                for b in range(batch_size):
                    mu_of_this_x = mu_batch[b]
                    # this only works if label is 1-dim
                    label_of_this_x = label_batch[b].item()
                    label_and_latent_act_dict[label_of_this_x].append(mu_of_this_x)
            
            batches_processed += 1
    
    # convert to numpy for easier manipulation / indexing
    for k, _ in label_and_latent_act_dict.items():
        label_and_latent_act_dict[k] = np.array(label_and_latent_act_dict[k])
    
    return label_and_latent_act_dict

def get_latent_activations_with_labels_for_scatter(dataloader, vae_model, curr_dev, z_dim, l_dim, 
                                batch_size = 32, batches = None, model_type='bvae', latent_layer='z1'):
   
    assert model_type in ['bvae', 'laddervae']
    assert latent_layer in ['z1', 'z2']

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    
    B = batches if batches is not None else len(dataloader)
    mu_batches = np.zeros(shape=(batch_size * B, z_dim))
    label_batches = np.zeros(shape=(batch_size * B, l_dim))
    
    batches_processed = 0
    with torch.no_grad():
        for x_batch, label_batch in dataloader:
            
            if batches is not None and batches_processed >= batches:
                break
            
            # First we encode this batch and get \mu and \sigma
            x_batch = x_batch.to(curr_dev)
            
            mu_batch = None
            if model_type == 'bvae':
                mu_batch, _ = vae_model.model.encode(x_batch)

            if model_type == 'laddervae':
                # TODO: fix this
                fwd_pass_results = vae_model.model.encode(x_batch)
                mu_batch = None
                # mu_batch = z1 if latent_layer == 'z1' else z2

            # convert to numpy format so that we can easily use in matplotlib
            mu_batch = mu_batch.detach().cpu().numpy()
            label_batch = label_batch.numpy()
            
            for i in range(batch_size):
                mu_batches[batches_processed*batch_size + i] = mu_batch[i]
                label_batches[batches_processed*batch_size + i] = label_batch[i]

            batches_processed += 1
        
    return mu_batches, label_batches

def do_latent_traversal(vae_model, random_img, limit=3, inter=2/3, dim=-1, mode='relative'):

    """
    random_image: image to be used as starting point to traversal (relevant in case of mode='relative')
    mode        : If 'relative', random image's Z is used as starting point for traversal.
                  If 'fixed', the space is explored uniformly in the given interval
    """    
    with torch.no_grad():

        interpolation = None
        random_img_z, _ = vae_model.model.encode(random_img)
        
        samples = []

        for row in range(vae_model.model.z_dim):
            if dim != -1 and row != dim:
                continue

            z = random_img_z.clone()
            
            if mode == 'relative':
                lim = z[:, row]/2
                interpolation = torch.arange(z[:,row] - lim, z[:,row] + lim + 0.1, inter)
            else:
                interpolation = torch.arange(-limit, limit+0.1, inter)
            
            for val in interpolation:
                z[:, row] = val
                sample = vae_model.model.decode(z).data
                samples.append((z,sample))

    return samples

def do_latent_traversal_scatter(vae_model, ref_img, limit=3, inter=2/3, 
    layer_to_explore='z1', dim_to_explore=-1, model_type='bvae', mode='relative', 
    lb=None, ub=None, fix_dim=None, fix_val=None):

    """
    This function also
    dim_to_explore: Dimension that we want to traverse. 
    ref_img       : Image to be used as starting point to traversal (relevant in case of mode='relative')
    mode          : If 'relative', random image's Z is used as starting point for traversal.
                  If 'fixed', the space is explored uniformly in the given interval
    """

    assert model_type in ['bvae', 'laddervae']
    assert layer_to_explore in ['z1', 'z2']

    with torch.no_grad():

        interpolation, ref = None, None

        random_img_z = None
        var_dist_params = None
        dim_size_to_iter = 0

        if model_type == 'bvae':
            random_img_z, _ = vae_model.model.encode(ref_img)
            dim_size_to_iter = vae_model.model.z_dim
        
        if model_type == 'laddervae':
            z1, z2, var_dist_params = vae_model.model.encode(ref_img)
            
            if layer_to_explore == 'z1':
                random_img_z = z1
                dim_size_to_iter = vae_model.model.z1_dim
            if layer_to_explore == 'z2':
                random_img_z = z2
                dim_size_to_iter = vae_model.model.z2_dim
        
        samples = []

        for current_dim in range(dim_size_to_iter):
                     
            if dim_to_explore != -1 and current_dim != dim_to_explore:
                continue

            z = random_img_z.clone()
            
            if fix_dim is not None and fix_val is not None:
                z[:, fix_dim] = fix_val
            
            if mode == 'relative':
                ref, lim = z[:, current_dim].item(), 3
                lower_bound = ref - lim if lb is None else lb
                upper_bound = ref + lim + 0.1 if ub is None else ub
                if lower_bound > upper_bound:
                    inter = -inter
                    lim = -lim
                print(f"Visualizing latent space from {lower_bound:3.2f} to {upper_bound:3.2f}, with center at {(upper_bound - lower_bound)/2:3.2f}")
                interpolation = torch.arange(lower_bound, upper_bound, inter)
            else:
                interpolation = torch.arange(-limit, limit+0.1, inter)
                
            for val in interpolation:
                z[:, current_dim] = val
                
                if model_type == 'laddervae':
                    sample = vae_model.model.decode(z, layer_to_explore, **var_dist_params).data
                
                if model_type == 'bvae':
                    sample = vae_model.model.decode(z).data
                
                samples.append((z[:, current_dim].cpu().item(),sample))

    return samples, ref

def show_traversal_plot(vae_model, anchor_image, limit, interp_step, dim=-1, mode='relative',
                    layer_to_explore='z1', model_type='bvae'):
    
    assert model_type in ['bvae', 'laddervae']
    assert layer_to_explore in ['z1', 'z2']

    traverse_maps, ref = do_latent_traversal_scatter(vae_model, anchor_image, limit=limit, 
                                            inter=interp_step, dim_to_explore=dim, mode=mode,
                                            layer_to_explore=layer_to_explore, model_type=model_type)                                                  

    _ , ax = plt.subplots(figsize=(15,1))
    
    for z, img in traverse_maps:
        ax.scatter(z, 0.2) 
        ab = AnnotationBbox(OffsetImage(img.squeeze(0).cpu().permute(1,2,0), zoom=0.5,cmap='gray'), 
                            (z, 0.2), 
                            frameon=False)
        ax.add_artist(ab)
    ax.vlines(ref,0,0.5)

def show_traversal_images(vae_model, anchor_image, limit, interp_step, dim=-1, mode='relative',
                    layer_to_explore='z1', model_type='bvae',nrow=10,  **kwargs):
    
    assert model_type in ['bvae', 'laddervae']
    assert layer_to_explore in ['z1', 'z2']

    traversed_images, ref = do_latent_traversal_scatter(vae_model, anchor_image, limit=limit, 
                                            inter=interp_step, dim_to_explore=dim, mode=mode,
                                            layer_to_explore=layer_to_explore, model_type=model_type)

    # every image in traversed_images has shape [1,channels,img_size, img_size] 
    # .squeeze() removes the first dim and then stack concats the images along a new first dim
    # to give [num_images,channels,img_size, img_size]
    traversed_images_stacked = torch.stack([t_img.squeeze(0) for _ , t_img in traversed_images], dim=0)
    img_grid = vutils.make_grid(traversed_images_stacked, normalize=True, nrow=nrow, value_range=(0.0,1.0), pad_value=1.0)

    return show_image_grid_pt(img_grid, **kwargs)


def load_model_and_data_and_get_activations(dset_name, dset_path, batch_size, z_dim , beta, 
                                            checkpoint_path, current_device, 
                                            activation_type='without_labels', seed=123,  batches=None,
                                            in_channels=1, l_zero_reg=False, **kwargs
    ):

    bvae_model_params = ModelParams(
        [z_dim], 6, 0, in_channels, 64, batch_size, 1.0, beta,
        False, 0, 0,
        0, 0, 0, 0, 0,
        ['SimpleGaussianConv64CommAss'],['SimpleConv64CommAss'], None, 'BetaVAE', l_zero_reg
    )
    experiment_config = dict(
            in_channels=in_channels,
            image_size=64,
            LR=1e-4,
            weight_decay=0.0,       
            dataset=dset_name,
            datapath=dset_path,
            droplast=True,        
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            seed=seed,
            evaluation_metrics=None,
            visdom_on=False,
            save_dir=None,
            max_epochs=1,
            l_zero_reg=l_zero_reg
    )

    model_for_dset = load_vae_model(
        algo_name='BetaVAE',
        algo_type='bvae', 
        checkpoint_path=checkpoint_path, 
        curr_dev=current_device,
        model_params=bvae_model_params,
        exp_params=experiment_config
    )

    model_for_dset.eval()

    dataset = get_configured_dataset(dset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    
    activations = None
    if activation_type == 'with_labels':
        activations = get_latent_activations_with_labels(loader, model_for_dset, 
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches)
        
    elif activation_type == 'without_labels':
        activations = get_latent_activations(loader, 
                                    model_for_dset,
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches)
    
    elif activation_type == 'for_scatter':
        activations = get_latent_activations_with_labels_for_scatter(loader, 
                                    model_for_dset,
                                    current_device,
                                    z_dim=z_dim,
                                    l_dim=kwargs['l_dim'],
                                    batch_size=batch_size,
                                    batches=batches)
                        
    return activations, dataset, model_for_dset

def do_semantic_manipulation(sampled_images, vae_model, current_device):
    """
    sampled_images: Currently we expect it to contain three images (x0,x1,x2)

    We transfer a property of x1 to x2. Assume:
    x0 = square on right
    x1 = square on left
    x3 = heart on right
    Then, we expect after decoding \mu(x2) + (\mu(x1) - \mu(x0)) to be a heart on the left
    x0 and x1 share all factors except one factor F_t. This F_t is what we want to transfer to x2
    """
    fig, axs = plt.subplots(1,4)
    mus, logvars = [], []

    with torch.no_grad():
        
        for i, img in enumerate(sampled_images):

            # show the three images we'll operate on
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
            
            # conver to tensor and encode
            img = transforms.ToTensor()(img.astype(np.float32)).to(current_device)
            mu, logvar = vae_model.model.encode(img.unsqueeze(0))
            mus.append(mu)
            logvars.append(logvars)

        # create a new conversion vector in direction of change 
        change_vec = mus[1] - mus[0]  # points from x0 to x1
        test_mu = mus[2] + change_vec 
        new_img = vae_model.model.decode(test_mu).squeeze(0)
        axs[3].imshow(new_img.cpu().permute(1,2,0), cmap='gray')
        axs[3].axis('off')    

        return mus, logvars, new_img

def sample_latent_pairs_differing_in_one_factor(diff_factor_idx, npz_dataset, how_many_pairs=1, val1=None, val2=None):
    
    """
    diff_factor_idx: All factors except this one will be shared b/w x_1 and x_2 in each pair
    val1 / val2: If we want to fix a value of F_t in either x_1 or x_2 we can supply it here.
    """

    pairs = []
    
    for _ in range(how_many_pairs):
        
        # sample a value for factors which changes b/w pair
        diff_factor_val1 = np.random.randint(npz_dataset['latents_sizes'][diff_factor_idx], size=1) if val1 is None else val1
        diff_factor_val2 = np.random.randint(npz_dataset['latents_sizes'][diff_factor_idx], size=1) if val2 is None else val2

        # Latent for x_1
        l1 = sample_latent(1, npz_dataset['latents_sizes'])
        l1[:, diff_factor_idx] = diff_factor_val1
        indices_sampled = latent_to_index(l1, npz_dataset['latents_bases'])
        img1 = npz_dataset['images'][indices_sampled]

        # Latent for x_2
        l2 = l1.copy()
        l2[:, diff_factor_idx] = diff_factor_val2
        indices_sampled = latent_to_index(l2, npz_dataset['latents_bases'])
        img2 = npz_dataset['images'][indices_sampled]
        
        pairs.append((l1,img1.squeeze(0),
                      l2,img2.squeeze(0)))
    
    return pairs

def sample_latent_pairs_maximally_differing_in_one_factor(diff_factor_idx, npz_dataset, direction='min-to-max', 
    how_many_pairs=1, latent_min_val=0,latent_max_val=0):
    
    """
    Same as `sample_latent_pairs_differing_in_one_factor` but here we want the lantent values to differ maximally.
    That is we fix one value to the minimum or range and other to maximum of range
    diff_factor_idx: All factors except this one will be shared b/w x_1 and x_2 in each pair
    latent_min_val: If we want to fix a min value of F_t we can supply it here.
    latent_max_val: If we want to fix a max value of F_t we can supply it here.
    """

    assert diff_factor_idx != 0, "Changing Color not supported"
    
    pairs = []
    
    for _ in range(how_many_pairs):
        
        # sample a value for factors which changes b/w pair
        # this gives us the index of minimum value for this latent dim
        diff_factor_val_min = latent_min_val
        # this gives us the index of maximum possible value for this latent dim
        diff_factor_val_max = latent_max_val

        l1 = sample_latent(1, npz_dataset['latents_sizes'])
        l1[:, diff_factor_idx] = diff_factor_val_min
        indices_sampled = latent_to_index(l1, npz_dataset['latents_bases'])
        img1 = npz_dataset['images'][indices_sampled]

        l2 = l1.copy()
        l2[:, diff_factor_idx] = diff_factor_val_max
        indices_sampled = latent_to_index(l2, npz_dataset['latents_bases'])
        img2 = npz_dataset['images'][indices_sampled]
        
        # We have to sample a third image to which we will apply semantic changes
        # if direction='min-to-max', it means we want to change from smallest possible value to largest 
        # so our sampled images should have smallest value. So we will use l1 as base and (randomly) change shape
        # Converse reasoning for direction='max-to-min' case
        
        l3 = l1.copy() if direction == 'min-to-max' else l2.copy()
        # get a random shape, but don't reuse shapes from l1 or l2
        if diff_factor_idx != 1:
            l3[:, 1] = np.random.choice(
                list(set(range(npz_dataset['latents_sizes'][1])).difference(l1[:,1])), size=1
            )
        
        indices_sampled = latent_to_index(l3, npz_dataset['latents_bases']) 
        img3 = npz_dataset['images'][indices_sampled]
        
        pairs.append(
            { 
                'latents': (l1, l2, l3), 
                'images': (img1.squeeze(0), img2.squeeze(0), img3.squeeze(0))
            }
        )
    
    return pairs

def estimate_responsible_dimension(pairs, vae_model, current_device):
    """
    Given pairs of examples which differ in one factor, estimate which latent unit is responsbile for which factor.
    We use maximum variance in latent dimension as a proxy for responsiblity
    """
    
    assert len(pairs) > 1, "Need > 1 pairs to compute variance"

    with torch.no_grad():
        
        diff_vecs = []
        mu_vecs = []
        for (latent1,image1,latent2,image2) in pairs:
            
            image1 = transforms.ToTensor()(image1.astype(np.float32)).to(current_device)
            image2 = transforms.ToTensor()(image2.astype(np.float32)).to(current_device)

            mu1, logvar1 = vae_model.model.encode(image1.unsqueeze(0))
            mu2, logvar2 = vae_model.model.encode(image2.unsqueeze(0))
            
            # TODO: should we ignore the color dimension forcefully when calcultaing variance?
            diff_vec = (mu1 - mu2).squeeze()
            diff_vecs.append(diff_vec)
            mu_vecs.append((mu1, mu2))
        
        diff_vecs = torch.stack(diff_vecs)
        dim_wise_variance = diff_vecs.pow(2).sum(0)/(len(pairs)-1)
        most_varied_dim = torch.argmax(dim_wise_variance)
        return diff_vecs.cpu().numpy(), dim_wise_variance.cpu().numpy(), most_varied_dim.item()

def check_correlated_dimensions(image_batch, vae_model, current_device, perturb_value=None, perturb_mode='fixed'):
    """
    Train a normal Beta-VAE network on the correlated data
    Once it has been trained, pass a batch of B examples and do the following:

    For each example X:
        Pass it thru network to get latent activations then
        For each latent dim l from 1 to L do:
            perturb unit l which gives an image X_l
            pass X_l again thru the network and record the perturbed mean \mu_l

    For each dimension see if passing the image again leads to a change in any dimension other than the perturbed one (correlation)

    Ideally, we expect only the unit associated with the originally perturned dimension to change.

    If changing unit m consistently results in changes in unit n then we conclude that those two dimensions are correlated.

    We can then introduce a layer before that and connect these two dims to a unit in prev layer.
    """

    
    with torch.no_grad():
        
        sq_diff_batch = []
        #fwd_pass_results = vae_model.model.forward(image_batch, current_device=current_device)
        mus_orig, logvars_orig = vae_model.model.encode(image_batch)
        #x_recon_orig, mu_orig = fwd_pass_results['x_recon'], fwd_pass_results['mu'] 

        # for each example X, perturb unit l=1 to L 
        for mu in mus_orig:
            # (dim(mu), mu)
            mus_perturbed = _generate_perturbed_copies(mu, perturb_mode=perturb_mode,fixed_val=perturb_value)
            
            # generate and image from these perturbed means
            # (dim(mu), X.shape)
            x_recons_perturbed = vae_model.model.decode(mus_perturbed, current_device=current_device)
            # pass images again and compare with mu_perturbed
            # (dim(mu), mu)
            mus_perturbed_recon, _ = vae_model.model.encode(x_recons_perturbed, current_device=current_device)

            x_recons_from_mu_perturbed_recon = vae_model.model.decode(mus_perturbed_recon, current_device=current_device)

            sq_diff = (mus_perturbed_recon - mus_perturbed).pow(2)
            sq_diff_batch.append((sq_diff, mus_perturbed_recon, mus_perturbed, x_recons_perturbed, x_recons_from_mu_perturbed_recon))

        return sq_diff_batch

def _generate_perturbed_copies(vector, dims_to_perturb=None, perturb_mode = 'fixed', fixed_val=None):
    """
    Assumes that vector is of shape (vector_dim, )
    Returns (vector_dim, vector_dim) shaped vector where in (i, vector_dim)
    perturb_mode: 'fixed','relative-min','relative-max' 
    """

    perturbed_copies = []

    for d in range(vector.shape[0]):
        if dims_to_perturb is not None and d not in dims_to_perturb:
                continue
        
        vector_d = vector.clone()
        lim = vector_d[d] / 2
        min_val, max_val = vector_d[d] - lim, vector_d[d] + lim + 0.1
        
        if perturb_mode == 'relative-max':    
            vector_d[d] = vector_d[d] + max_val
        
        if perturb_mode == 'relative-min':
            vector_d[d] = vector_d[d] - min_val
        
        if perturb_mode == 'fixed':
            vector_d[d] = vector_d[d] + fixed_val

        perturbed_copies.append(vector_d)

    return torch.stack(perturbed_copies, dim=0)

def visualize_perturbed_dims(diff_entry):
    """
    diff_entry: is a single element from the output of `check_correlated_dimensions`
    """

    # 0-th index stores the squared differences
    diff_np = diff_entry[0].cpu().numpy()
    
    fig, axs = plt.subplots(1,diff_np.shape[0])
    for r in range(diff_np.shape[0]):
        axs[r].imshow(np.expand_dims(diff_np[r,:], axis=1), cmap='Reds')
        axs[r].set_yticklabels([])
        axs[r].set_yticks([])
        axs[r].set_xticks([])
        axs[r].set_xticklabels([])

def sample_latent_pairs_differing_in_k_factors(diff_factor_indices, npz_dataset, how_many_pairs=1):
    
    """
    diff_factor_idx: All factors except this one will be shared b/w x_1 and x_2 in each pair
    val1 / val2: If we want to fix a value of F_t in either x_1 or x_2 we can supply it here.
    """

    pairs = []
    K = len(diff_factor_indices)

    for _ in range(how_many_pairs):
        
        # sample a value for factors which changes b/w pair
        D = []
        for diff_factor_idx in diff_factor_indices:
            diff_factor_val1 = np.random.randint(npz_dataset['latents_sizes'][diff_factor_idx], size=1)
            diff_factor_val2 = np.random.randint(npz_dataset['latents_sizes'][diff_factor_idx], size=1)

            D.append((diff_factor_val1, diff_factor_val2))

        # Latent for x_1
        l1 = sample_latent(1, npz_dataset['latents_sizes'])
        for vals, diff_factor_idx in zip(D, diff_factor_indices):
            l1[:, diff_factor_idx] = vals[0]
        indices_sampled = latent_to_index(l1, npz_dataset['latents_bases'])
        img1 = npz_dataset['images'][indices_sampled]

        # Latent for x_2
        l2 = l1.copy()
        for vals, diff_factor_idx in zip(D, diff_factor_indices):
            l1[:, diff_factor_idx] = vals[1]
        indices_sampled = latent_to_index(l2, npz_dataset['latents_bases'])
        img2 = npz_dataset['images'][indices_sampled]
        
        pairs.append((l1,img1.squeeze(0),
                      l2,img2.squeeze(0)))
    
    return pairs

"""
START: Notebook + Visualization functions required for LadderVAE
These funcs have been adapted from their beta-vae variants above 
and use the prefix 'laddervae_'
"""
def laddervae_load_model_and_data_and_get_activations(dset_name, dset_path, batch_size, z_dim , beta, 
                                            checkpoint_path, current_device, activation_type='without_labels', 
                                            seed=123,  batches=None, in_channels=1, activations_for_latent_layer='z1', **kwargs
    ):

    # TODO: In LadderVAE case z_dim should be a list because latent layers can be of different sizes
    bvae_model_params = ModelParams(
        z_dim, 6, 0, in_channels, 64, batch_size, 1.0, beta,
        False, 0, 0,
        0, 0, 0, 0, 0,
        ['SimpleGaussianConv64'],['SimpleConv64'], None, 'LadderVAE', kwargs['l_zero_reg']
    )
    experiment_config = dict(
            in_channels=in_channels,
            image_size=64,
            LR=1e-4,
            weight_decay=0.0,       
            dataset=dset_name,
            datapath=dset_path,
            droplast=True,        
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            seed=seed,
            evaluation_metrics=None,
            visdom_on=False,
            save_dir=None,
            max_epochs=1,
            l_zero_reg=kwargs['l_zero_reg']
    )

    model_for_dset = load_vae_model(
        algo_name='LadderVAE',
        algo_type='laddervae', 
        checkpoint_path=checkpoint_path, 
        curr_dev=current_device,
        model_params=bvae_model_params,
        exp_params=experiment_config
    )

    model_for_dset.eval()

    dataset = get_configured_dataset(dset_name)   
    
    activations = None

    assert activation_type in ['with_labels', 'without_labels', 'for_scatter']

    if activation_type == 'with_labels':
        activations = get_latent_activations_with_labels(dataset, model_for_dset, 
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches,
                                    model_type='laddervae', 
                                    latent_layer=activations_for_latent_layer)
        
    elif activation_type == 'without_labels':
        activations = get_latent_activations(dataset, 
                                    model_for_dset,
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches,
                                    model_type='laddervae')
    
    elif activation_type == 'for_scatter':
        activations = get_latent_activations_with_labels_for_scatter(dataset, 
                                    model_for_dset,
                                    current_device,
                                    z_dim=z_dim,
                                    l_dim=kwargs['l_dim'],
                                    batch_size=batch_size,
                                    batches=batches,
                                    model_type='laddervae', 
                                    latent_layer=activations_for_latent_layer)
                        
    return activations, dataset, model_for_dset

"""
END: Notebook + Visualization functions required for LadderVAE
"""

"""
START: special functions for visualizing internals / traversals for csvae_gnn model
"""
def csvaegnn_show_traversal_images(vae_model, anchor_image, limit, interp_step, current_device=None,
                                    dim_to_explore=0, mode='relative',
                                    nodes_to_explore=[], datapoints=None, nrow=10, **kwargs):
    
    traversed_images = None
    
    if len(nodes_to_explore) == 1:
        
        traversed_images = csvaegnn_do_latent_traversal_scatter(vae_model, anchor_image, 
                                                                limit=limit, inter=interp_step, 
                                                                dim_to_explore=dim_to_explore, mode=mode,
                                                                node_to_explore=nodes_to_explore[0])

    if len(nodes_to_explore) == 2:
        
        traversed_images = csvaegnn_do_latent_traversal_multiple(vae_model, anchor_image, inter=interp_step, 
                                                                    nodes_to_explore=nodes_to_explore,
                                                                    datapoints=datapoints, 
                                                                    current_device=current_device)

    # every image in traversed_images has shape [1,channels,img_size, img_size] 
    # .squeeze() removes the first dim and then stack concats the images along a new first dim
    # to give [num_images,channels,img_size, img_size]

    traversed_images_stacked = torch.stack([t_img.squeeze(0) for t_img in traversed_images], dim=0)
    img_grid = vutils.make_grid(traversed_images_stacked, normalize=True, nrow=nrow, value_range=(0.0,1.0), pad_value=1.0)

    return show_image_grid_pt(img_grid, **kwargs)

def csvaegnn_do_latent_traversal_scatter(vae_model, ref_img, limit=3, inter=2/3, 
                                        node_to_explore=0, dim_to_explore=0, mode='relative', 
                                        lb=None, ub=None, fix_dim=None, fix_val=None):

    """
    node_to_explore: which node of GNN to explore
    dim_to_explore: Dimension that we want to traverse for this node. This is useful because node_features can be multi-dim
    ref_img       : Image to be used as starting point to traversal (relevant in case of mode='relative')
    mode          : If 'relative', random image's Z is used as starting point for traversal.
                  If 'fixed', the space is explored uniformly in the given interval
    """
    with torch.no_grad():

        interpolation, ref = None, None       
        dim_size_to_iter = vae_model.z_dim
        layer_level_to_explore = 0

        # these will be of shape (Batches, V, node_feat_dim)
        posterior_mu, posterior_logvar, posterior_z = vae_model.encode(ref_img)
        num_nodes = vae_model.num_nodes            
        random_img_z = posterior_z
        samples = []

        for node_idx in range(num_nodes):

            if node_idx != node_to_explore:
                continue

            for current_dim in range(dim_size_to_iter):
                        
                if dim_to_explore != -1 and current_dim != dim_to_explore:
                    continue

                z = random_img_z.clone()
                
                if fix_dim is not None and fix_val is not None:
                    z[:, node_idx, fix_dim] = fix_val
                
                if mode == 'relative':
                    ref, lim = z[:, node_idx, current_dim].item(), 3
                    lower_bound = ref - lim if lb is None else lb
                    upper_bound = ref + lim + 0.1 if ub is None else ub
                    if lower_bound > upper_bound:
                        inter = -inter
                        lim = -lim
                    print(f"Visualizing latent space from {lower_bound:3.2f} to {upper_bound:3.2f}, with center at {(upper_bound - lower_bound)/2:3.2f}")
                    interpolation = torch.arange(lower_bound, upper_bound, inter)
                else:
                    interpolation = torch.arange(-limit, limit+0.1, inter)
                    
                # Traverse and decode
                for val in interpolation:
                    z[:, node_idx, current_dim] = val
                    z_flattened = vae_model.flatten_node_features(z)
                    sample = vae_model.decode(z_flattened).data
                    
                    samples.append(sample)

    return samples

def csvaegnn_do_latent_traversal_multiple(vae_model, ref_img, 
                                    inter=2/3, nodes_to_explore=[], datapoints=None, 
                                    current_device=None):

    """
    node_to_explore: which node of GNN to explore
    dim_to_explore: Dimension that we want to traverse for this node. This is useful because node_features can be multi-dim
    ref_img       : Image to be used as starting point to traversal (relevant in case of mode='relative')
    """
    with torch.no_grad():

        # these will be of shape (Batches, V, node_feat_dim)
        posterior_mu, posterior_logvar, posterior_z = vae_model.encode(ref_img)
        samples = []
        z = posterior_z.clone()

        if z.size(2) > 1:
            print("This func was written with 1d latents in min. Behaviour for >1d latents not guaranteed")
        
        #interpolation = torch.arange(-3., 3.1, inter).to(current_device)

        # Traverse and decode
        for point in datapoints:
            # the the values of all nodes in `nodes_to_explore` to the same value
            # we are exploring along the diagonal
            z[:, nodes_to_explore, 0] = point.reshape(1,2)
            z_flattened = vae_model.flatten_node_features(z)
            sample = vae_model.decode(z_flattened).data
            
            samples.append(sample)

    return samples

def csvaegnn_get_latent_activations_with_labels_for_scatter(vae_model, dataset_loader, curr_dev, batches = None):
   
    num_nodes, batch_size = vae_model.num_nodes, vae_model.batch_size
    z_dim, l_dim = vae_model.z_dim, vae_model.l_dim

    B = batches if batches is not None else len(dataset_loader)
    mu_batches = np.zeros(shape=(batch_size * B, num_nodes, z_dim))
    label_batches = np.zeros(shape=(batch_size * B, l_dim))
    
    batches_processed = 0
    with torch.no_grad():
        
        for x_batch, label_batch in iter(dataset_loader):
            
            if batches is not None and batches_processed >= batches:
                break
            
            # First we encode this batch and get \mu and \sigma
            x_batch = x_batch.to(curr_dev)
            
            # posterior_mu has shape (B, V, feat_dim)
            posterior_mu, posterior_logvar, posterior_z = vae_model.encode(x_batch)
            
            # convert to numpy format so that we can easily use in matplotlib
            posterior_mu = posterior_mu.detach().cpu().numpy()
            label_batch = label_batch.numpy()
            
            for i in range(batch_size):
                mu_batches[batches_processed * batch_size + i] = posterior_mu[i]
                label_batches[batches_processed * batch_size + i] = label_batch[i]

            batches_processed += 1
        
    return mu_batches, label_batches

def csvaegnn_intervene_upper_layers(vae_model, x, intervention_level, intervention_nodes, intervention_values):
    """
    Function assumes 1-d latent space for node features

    """
    
    gnn_levels = len(vae_model.encoder_gnn)
    assert intervention_level < gnn_levels, f"GNN depth has {gnn_levels} layers, can't intervene on {intervention_level} layer"
    mu, logvar = None, None
    samples = []
    
    with torch.no_grad():
        
        mse = vae_model.encoder_cnn(x)

        for intervention_value in intervention_values:
            
            params = mse.clone()
            
            for g, gnn_layer in enumerate(vae_model.encoder_gnn):
            
                # compute activations at this GNN level
                params = gnn_layer(params)

                # if this is the intervention level we have to replace the computed 
                # value for nodes on which we're intervening to the given intervention_values
                if g == intervention_level:
                    # if in final level
                    if g == gnn_levels - 1:
                        mu, logvar = params
                        #z_sample = reparametrize(*params)
                        #z_sample[:, intervention_nodes, 0] = intervention_value
                        mu[:, intervention_nodes, 0] = intervention_value
                        params = mu, logvar
                    else:
                        params[:, intervention_nodes, 0] = intervention_value

            # intervention has propagated, now we can recon
            z_sample = reparametrize(*params)
            z_flattened = vae_model.flatten_node_features(z_sample)
            sample = vae_model.decode(z_flattened).data

            samples.append(sample)

    return samples
    
def get_prior_mus_given_gt_labels(vae_model, dataloader, current_device, batches):
    
    l_dim = vae_model.l_dim
    batch_size = vae_model.batch_size

    B = batches if batches is not None else len(dataloader)
    prior_mu_batches = np.zeros(shape=(batch_size * B, l_dim))
    gt_batches = np.zeros(shape=(batch_size * B, l_dim))
    
    batches_processed = 0
    
    with torch.no_grad():
        for _, label_batch in dataloader:
            
            if batches is not None and batches_processed >= batches:
                break
            
            label_batch = label_batch.to(current_device)
            prior_mu_batch, _ = vae_model.prior_gnn(label_batch)

            # convert to numpy format so that we can easily use in matplotlib
            prior_mu_batch = prior_mu_batch.squeeze().detach().cpu().numpy()
            label_batch = label_batch.cpu().numpy()
            
            for i in range(batch_size):
                prior_mu_batches[batches_processed*batch_size + i] = prior_mu_batch[i]
                gt_batches[batches_processed*batch_size + i] = label_batch[i]

            batches_processed += 1
    
    return prior_mu_batches, gt_batches

def marginal_node_effect_at_final_level(vae_model, latent_act_joint_dist, node_idx, node_value, current_device, samples=1):
    """
    This func can only intervene on the final layer of GNN.
    Thus the cascade of cause to effect wont be visible thru this function.
    In terms of DAG, this func can only meaningfully intervene on the 'leaves'

    latent_act_joint_dist: Can be the output of `csvaegnn_get_latent_activations_with_labels_for_scatter`
    
    """
    samples = []

    for _ in range(samples):
        
        # choose any random image's activation. This can be seen as 
        # sampling from q(z|X).
        random_idx = np.random.choice(latent_act_joint_dist.size[0])
        z_sample = latent_act_joint_dist[random_idx]
        
        # Intervene do(z_i = C)
        z_sample[node_idx] = node_value
        
        # reconstruct
        # Right now we have (V, feat_dim). We have to add the batch dimension
        # so it is of shape (1, V, feat_dim) and then can be passed thru model
        z_sample = z_sample.reshape(1, z_sample.shape[0]) # [np.newaxis,:,:]
        z_sample = torch.from_numpy(z_sample).type(torch.FloatTensor).to(current_device)

        # Sample from p(X|z_js, do(z_i = C))
        samples.append(vae_model.decode(z_sample))

    return samples

def csvaegnn_intervene_final_layer(vae_model, x, intervention_nodes, intervention_values):
    """
    Function assumes 1-d latent space for node features

    """
    
    samples = []
    
    with torch.no_grad():
        
        mu, logvar, z = vae_model.encode(x)
        z_sample = reparametrize(mu, logvar)

        amputated_mat = vae_model.dept_adjacency_matrix
        amputated_mat[intervention_nodes] = torch.zeros(amputated_mat.size()[0])
        amputated_mat[intervention_nodes, intervention_nodes] = 1.0
        print(amputated_mat)

        for intervention_value in intervention_values:
            
            z_sample = z.clone()
            z_sample[:, intervention_nodes, 0] = intervention_value

            # apply amputated matrix to transfer effects
            z_sample = torch.matmul(amputated_mat, z_sample)
            # intervention has propagated, now we can recon
            z_flattened = vae_model.flatten_node_features(z_sample)
            sample = vae_model.decode(z_flattened).data

            samples.append(sample)

    return samples

def latentnn_intervene(vae_model, x,  intervened_node, intervention_values):
    
    with torch.no_grad():
        
        image_features = vae_model.encoder_cnn(x)
        samples = []
        
        for iv in intervention_values:
            
            mus, logvars, zs = vae_model.encoder_gnn_intervention(image_features, intervened_node, iv)
            posterior_z = vae_model.flatten_node_features(zs)
            x_recon = vae_model.decode(posterior_z).data

            samples.append(x_recon)

        return samples

def latentnn_show_intervention_atlas_from_anchor(intervened_node, intervention_values, anchor_image, vae_model, 
                                                nrow=12, figsize=(10,10)):
    
    traversed_images = latentnn_intervene(vae_model, anchor_image, intervened_node, intervention_values)
    traversed_images_stacked = torch.stack([t_img.squeeze(0) for t_img in traversed_images], dim=0)
    img_grid = vutils.make_grid(traversed_images_stacked, normalize=True, nrow=nrow, value_range=(0.0,1.0), pad_value=1.0)
    return show_image_grid_pt(img_grid, figsize=figsize)


def latentnn_show_intervention_comparison(intervened_node, intervention_values, num_samples, 
                                          dataloader, vae_model, current_device, 
                                          pad_value=1.0, figsize=(10,10)):

    figures = []
    for node_value in intervention_values:   

        images, labels = next(dataloader.__iter__())
        bs = images.size()[0]
        intervened_images, orig_images = [], []
        
        # for every fixed value C, we apply it as intervention to num_samples images
        for _ in range(num_samples):

            # select a random image from batch
            x = images[np.random.choice(bs)].unsqueeze(0).to(current_device)
            
            # padd height so we can see a border below the images
            n, c, h, w = x.shape
            padding_tensor = torch.zeros(size=(n,c,2,w), device=current_device) if pad_value == 0. else torch.ones(size=(n,c,2,w), device=current_device) 
            x = torch.cat([x, padding_tensor], dim=2)
            
            orig_images.append(x)

            x_intervened = latentnn_intervene(vae_model, x, intervened_node, intervention_values=[node_value])[0]
            x_intervened = torch.cat([x_intervened, padding_tensor], dim=2)
            intervened_images.append(x_intervened)

        # each row represents a fixed intervention value across different images
        stacked_images = []
        # stack of original images - top row
        stacked_images.append(torch.stack([o_img.squeeze(0) for o_img in orig_images], dim=0))
        # stack of traversed images - bottom row
        stacked_images.append(torch.stack([t_img.squeeze(0) for t_img in intervened_images], dim=0))
        comparison_visual = torch.cat(stacked_images, dim=2)
        img_grid = vutils.make_grid(comparison_visual, normalize=True, nrow=num_samples, value_range=(0.0,1.0), pad_value=pad_value)

        figures.append(show_image_grid_pt(img_grid, figsize=figsize))

    return figures

