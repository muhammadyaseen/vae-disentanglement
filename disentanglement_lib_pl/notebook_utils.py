import os
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple
from tqdm import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as T
from torchvision import transforms
from torch.utils.data import DataLoader

from base_vae_experiment import BaseVAEExperiment
from bvae_experiment import BVAEExperiment
from laddervae_experiment import LadderVAEExperiment

from common.data_loader import CustomImageFolder
from common.known_datasets import DSpritesDataset, CorrelatedDSpritesDataset, ThreeShapesDataset, ContinumDataset 

ModelParams = namedtuple('ModelParams', ["z_dim", "l_dim", "num_labels" , "in_channels", 
                                        "image_size", "batch_size", "w_recon", "w_kld", 
                                         "controlled_capacity_increase", "max_c", "iterations_c",
                                        "w_tc", "w_infovae", "w_dipvae", "lambda_od", "lambda_d_factor",
                                        "encoder", "decoder", "loss_terms", "alg", "l_zero_reg"])

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
def __show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

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
    else:
        raise NotImplementedError

    return dataset

def load_vae_model(algo_name, algo_type, checkpoint_path, curr_dev, 
                   model_params, exp_params):
    
    assert algo_type in ['bvae', 'laddervae']

    import models
    
    vae_model_class = getattr(models, algo_name)
    vae_model = vae_model_class(model_params)
        
    vae_experiment = None

    if algo_type == 'bvae':
        vae_experiment = BVAEExperiment.load_from_checkpoint(
            checkpoint_path,
            map_location=curr_dev,
            vae_model=vae_model, 
            params=exp_params)
    
    if algo_type == 'laddervae':
        vae_experiment = LadderVAEExperiment.load_from_checkpoint(
            checkpoint_path,
            map_location=curr_dev,
            vae_model=vae_model, 
            params=exp_params)

    vae_experiment = vae_experiment.to(curr_dev)

    return vae_experiment

def get_latent_activations(dataset, vae_model, curr_dev, batch_size = 32, batches = None, 
                                    model_type='bvae'):
   
    assert model_type in ['bvae', 'laddervae']
    #assert latent_layer in ['z1', 'z2']

    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    
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

        for x_batch, _ in tqdm(loader):
            
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

def get_latent_activations_with_labels(dataset, vae_model, curr_dev, batch_size = 32, 
                                    batches = None, group_by=False, model_type='bvae', latent_layer='z1'):
   
    assert model_type in ['bvae', 'laddervae']
    assert latent_layer in ['z1', 'z2']

    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    label_and_latent_act_dict = defaultdict(list)
    
    batches_processed = 0
    with torch.no_grad():
        for x_batch, label_batch in tqdm(loader):
            
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

def get_latent_activations_with_labels_for_scatter(dataset, vae_model, curr_dev, z_dim, l_dim, 
                                batch_size = 32, batches = None, model_type='bvae', latent_layer='z1'):
   
    assert model_type in ['bvae', 'laddervae']
    assert latent_layer in ['z1', 'z2']

    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    
    B = batches if batches is not None else len(loader)
    mu_batches = np.zeros(shape=(batch_size * B, z_dim))
    label_batches = np.zeros(shape=(batch_size * B, l_dim))
    
    batches_processed = 0
    with torch.no_grad():
        for x_batch, label_batch in tqdm(loader):
            
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
                    layer_to_explore='z1', model_type='bvae',nrow=10):
    
    assert model_type in ['bvae', 'laddervae']
    assert layer_to_explore in ['z1', 'z2']

    traversed_images, ref = do_latent_traversal_scatter(vae_model, anchor_image, limit=limit, 
                                            inter=interp_step, dim_to_explore=dim, mode=mode,
                                            layer_to_explore=layer_to_explore, model_type=model_type)

    # every image in traversed_images has shape [1,channels,img_size, img_size] 
    # .squeeze() removes the first dim and then stack concats the images along a new first dim
    # to give [num_images,channels,img_size, img_size]
    traversed_images_stacked = torch.stack([t_img.squeeze(0) for _ , t_img in traversed_images], dim=0)
    img_grid = vutils.make_grid(traversed_images_stacked, normalize=True, nrow=nrow, value_range=(0.0,1.0))

    __show(img_grid)


def load_model_and_data_and_get_activations(dset_name, dset_path, batch_size, z_dim , beta, 
                                            checkpoint_path, current_device, 
                                            activation_type='without_labels', seed=123,  batches=None,
                                            in_channels=1, **kwargs
    ):

    bvae_model_params = ModelParams(
        z_dim, 6, 0, in_channels, 64, batch_size, 1.0, beta,
        False, 0, 0,
        0, 0, 0, 0, 0,
        ['SimpleGaussianConv64CommAss'],['SimpleConv64CommAss'], None, 'BetaVAE'
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
            save_dir=None
    )

    model_for_dset = load_vae_model(
        algo_name='BetaVAE', 
        checkpoint_path=checkpoint_path, 
        curr_dev=current_device,
        model_params=bvae_model_params,
        exp_params=experiment_config
    )

    model_for_dset.eval()

    dataset = get_configured_dataset(dset_name)

    activations = None
    if activation_type == 'with_labels':
        activations = get_latent_activations_with_labels(dataset, model_for_dset, 
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches)
        
    elif activation_type == 'without_labels':
        activations = get_latent_activations(dataset, 
                                    model_for_dset,
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches)
    
    elif activation_type == 'for_scatter':
        activations = get_latent_activations_with_labels_for_scatter(dataset, 
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

def check_correlated_dimensions(image_batch, vae_model, current_device):
    """
    Hi, just to make sure I understood the experiment you wanted to do correctly I have written it down. 
    Let me know if it is right.

    Train a normal \beta-VAE network on the correlated data
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

        for img, _ in image_batch:

            fwd_pass_results = vae_model.model.forward(img, current_device=current_device)
            x_recon, mu_orig = fwd_pass_results['x_recon'], fwd_pass_results['mu'] 

            # for each example X
            # perturb unit l=1 to L 
            mu_perturbed = [_perturb_mu_in_given_dim(mu_orig, dim_to_perturb=i) for i in mu_orig.shape[1]]
            # generate and image from these perturbed means
            x_recons_perturbed = [vae_model.model.decode(mu_perturbed_i, current_device=current_device) for mu_perturbed_i in mu_perturbed]
            # pass images again and compare with mu_perturbed
            mus_compare = [vae_model.model.encode(x_recons_perturbed_i, current_device=current_device)[0] for x_recons_perturbed_i in x_recons_perturbed]


            # Let's say we perturbed dim=n in mu_perturbed, now we have to check how many dims in corresponding `mus_compare` are different
            # we can take a difference an see ?

def _perturb_mu_in_given_dim(vector_batch, dim_to_perturb):
    pass


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