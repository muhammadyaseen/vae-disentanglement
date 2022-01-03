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

from vae_experiment import VAEExperiment
from common.data_loader import DSpritesDataset, ThreeShapesDataset, ContinumDataset

ModelParams = namedtuple('ModelParams', ["z_dim", "l_dim", "num_labels" , "in_channels", 
                                        "image_size", "batch_size", "w_recon", "w_kld", 
                                         "controlled_capacity_increase", "max_c", "iterations_c",
                                        "w_tc", "w_infovae", "w_dipvae", "lambda_od", "lambda_d_factor",
                                        "encoder", "decoder", "loss_terms", "alg"])

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

def load_vae_model(algo_name, checkpoint_path, curr_dev, 
                   model_params, exp_params):
    
    import models
    
    vae_model_class = getattr(models, algo_name)
    vae_model = vae_model_class(model_params)
        
    vae_experiment = VAEExperiment.load_from_checkpoint(
        checkpoint_path,
        map_location=curr_dev,
        vae_model=vae_model, 
        params=exp_params)

    vae_experiment = vae_experiment.to(curr_dev)

    return vae_experiment

def get_latent_activations(dataset, vae_model, curr_dev, batch_size = 32, batches = None):
   
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    latent_acts = []
    for _ in range(vae_model.model.z_dim):
        latent_acts.append([])
    
    batches_processed = 0

    with torch.no_grad():

        for x_batch, label_batch in tqdm(loader):
            
            if batches is not None and batches_processed >= batches:
                break
            
            # First we encode this batch
            x_batch = x_batch.to(curr_dev)
            #print(x_batch.shape)
            mu_batch, log_var_batch = vae_model.model.encode(x_batch)

            # Then, we get \mu and \sigma for this batch
            mu_batch = mu_batch.detach().cpu().numpy()
            log_var_batch = log_var_batch.detach().cpu().numpy()

            # and labels
            label_batch = label_batch.cpu().numpy()
            
            # using labels, we place all \mu's belonging to same class together
            for b in range(batch_size):

                mu_of_this_x = mu_batch[b]
                
                # for reach dimension
                for m, m_dim in enumerate(mu_of_this_x): 
                    
                    #latent_act_dict[m].append(m_dim.item())
                    latent_acts[m].append(m_dim.item())
                    
            batches_processed += 1
    
    return latent_acts #latent_act_dict

def plot_latent_dist(activations, label="Factor X"):

    fig, axes = plt.subplots(1)
    axes.hist(activations, label=label, align='mid')
    axes.legend(prop={'size': 10},loc='upper left')

def get_latent_activations_with_labels(dataset, vae_model, curr_dev, batch_size = 32, batches = None):
   
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, drop_last=True)
    label_and_latent_act_dict = defaultdict(list)
    
    batches_processed = 0
    with torch.no_grad():
        for x_batch, label_batch in tqdm(loader):
            
            if batches is not None and batches_processed >= batches:
                break
            
            # First we encode this batch and get \mu and \sigma
            x_batch = x_batch.to(curr_dev)
            mu_batch, log_var_batch = vae_model.model.encode(x_batch)
            
            # convert to numpy format so that we can easily use in matplotlib
            mu_batch = mu_batch.detach().cpu().numpy()
            log_var_batch = log_var_batch.detach().cpu().numpy()
            label_batch = label_batch.cpu().numpy()
            
            # using labels, we place all \mu's belonging to same class together
            for b in range(batch_size):

                mu_of_this_x = mu_batch[b]
                
                # this only words if label is 1-dim
                label_of_this_x = label_batch[b].item()
                label_and_latent_act_dict[label_of_this_x].append(mu_of_this_x)
            
            batches_processed += 1
    
    # convert to numpy for easier manipulation / indexing
    for k, _ in label_and_latent_act_dict.items():
        label_and_latent_act_dict[k] = np.array(label_and_latent_act_dict[k])
    
    return label_and_latent_act_dict

def do_latent_traversal(vae_model, random_img, limit=3, inter=2/3, loc=-1, mode='relative'):

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
            if loc != -1 and row != loc:
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

def do_latent_traversal_scatter(vae_model, random_img, limit=3, inter=2/3, loc=-1, mode='relative'):

    """
    This function also 
    random_image: image to be used as starting point to traversal (relevant in case of mode='relative')
    mode        : If 'relative', random image's Z is used as starting point for traversal.
                  If 'fixed', the space is explored uniformly in the given interval
    """    
    with torch.no_grad():

        interpolation, ref = None, None
        random_img_z, _ = vae_model.model.encode(random_img)
        
        samples = []

        for row in range(vae_model.model.z_dim):
            if loc != -1 and row != loc:
                continue

            z = random_img_z.clone()
            if mode == 'relative':
                ref, lim = z[:, row].item(), 3
                lower_bound, upper_bound = ref - lim, ref + lim + 0.1 
                if lower_bound > upper_bound:
                    inter = -inter
                    lim = -lim
                print(f"Visualizing latent space from {lower_bound:3.2f} to {upper_bound:3.2f}, with center at {ref:3.2f}")
                interpolation = torch.arange(lower_bound, upper_bound, inter)
            else:
                interpolation = torch.arange(-limit, limit+0.1, inter)
                
            for val in interpolation:
                z[:, row] = val
                sample = vae_model.model.decode(z).data
                samples.append((z[:, row].cpu().item(),sample))

    return samples, ref

def show_traversal_plot(vae_model, anchor_image, limit, interp_step, dim=-1, mode='relative'):
    
    traverse_maps, ref = do_latent_traversal_scatter(vae_model, anchor_image, limit=limit, 
                                            inter=interp_step, loc=dim, mode=mode)

    _ , ax = plt.subplots(figsize=(15,1))
    
    for z, img in traverse_maps:
        ax.scatter(z, 0.2) 
        ab = AnnotationBbox(OffsetImage(img.squeeze(0).cpu().permute(1,2,0), zoom=0.5,cmap='gray'), 
                            (z, 0.2), 
                            frameon=False)
        ax.add_artist(ab)
    ax.vlines(ref,0,0.5)

def load_model_and_data_and_get_activations(dset_name, dset_path, batch_size, z_dim , beta, 
                                            checkpoint_path, current_device, 
                                            activation_with_label=False, seed=123,  batches=None,
                                            in_channels=1
    ):

    bvae_model_params = ModelParams(
        z_dim, 6, 0, in_channels, 64, batch_size, 1.0, beta,
        False, 0, 0,
        0, 0, 0, 0, 0,
        ['SimpleGaussianConv64'],['SimpleConv64'], None, 'BetaVAE'
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

    dataset = None
    if dset_name == 'dsprites_full':
        dataset = DSpritesDataset(root="../datasets/dsprites/", split="train", transforms=transforms.ToTensor())
    elif dset_name == 'dsprites_correlated':
        dataset = DSpritesDataset(root="../datasets/dsprites/", split="train", transforms=transforms.ToTensor())
    elif dset_name == 'dsprites_colored':
        dataset = DSpritesDataset(root="../datasets/dsprites/", split="train", transforms=transforms.ToTensor(), 
        correlated=True, colored=True)
    elif dset_name == 'threeshapesnoisy':
        dataset = ThreeShapesDataset(root="../datasets/threeshapesnoisy/", split="train", transforms=transforms.ToTensor())
    elif dset_name == 'threeshapes':
        dataset = ThreeShapesDataset(root="../datasets/threeshapes/", split="train", transforms=transforms.ToTensor())
    else:
        raise NotImplementedError
    
    activations = None
    if activation_with_label:
        activations = get_latent_activations_with_labels(dataset, model_for_dset, 
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches)
        
    else:
        activations = get_latent_activations(dataset, 
                                    model_for_dset,
                                    current_device,
                                    batch_size=batch_size,
                                    batches=batches)
                        
    return activations, dataset, model_for_dset

def do_semantic_manipulation(sampled_images, vae_model, current_device):

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
        change_vec = mus[1] - mus[0]  # from x0 to x1
        test_mu = mus[2] + change_vec 
        new_img = vae_model.model.decode(test_mu).squeeze(0)
        axs[3].imshow(new_img.cpu().permute(1,2,0), cmap='gray')
        axs[3].axis('off')    

        return mus, logvars, new_img

def sample_latent_pairs_differing_in_one_factor(diff_factor_idx, npz_dataset, how_many_pairs=1):
    
    pairs = []
    
    for _ in range(how_many_pairs):
        
        # sample a value for factors which changes b/w pair
        diff_factor_val1 = np.random.randint(npz_dataset['latents_sizes'][diff_factor_idx], size=1)
        diff_factor_val2 = np.random.randint(npz_dataset['latents_sizes'][diff_factor_idx], size=1)

        l1 = sample_latent(1, npz_dataset['latents_sizes'])
        l1[:, diff_factor_idx] = diff_factor_val1
        indices_sampled = latent_to_index(l1, npz_dataset['latents_bases'])
        img1 = npz_dataset['images'][indices_sampled]

        l2 = l1.copy()
        l2[:, diff_factor_idx] = diff_factor_val2
        indices_sampled = latent_to_index(l2, npz_dataset['latents_bases'])
        img2 = npz_dataset['images'][indices_sampled]
        
        pairs.append((l1,img1.squeeze(0),
                      l2,img2.squeeze(0)))
    
    return pairs

def sample_latent_pairs_maximally_differing_in_one_factor(diff_factor_idx, npz_dataset, direction='min-to-max', 
    how_many_pairs=1, latent_min_val=0,latent_max_val=0):
    
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
    
    assert len(pairs) > 1, "Need > 1 pairs to compute variance"

    with torch.no_grad():
        
        diff_vecs = []
        
        for (latent1,image1,latent2,image2) in pairs:
            
            image1 = transforms.ToTensor()(image1.astype(np.float32)).to(current_device)
            image2 = transforms.ToTensor()(image2.astype(np.float32)).to(current_device)

            mu1, logvar1 = vae_model.model.encode(image1.unsqueeze(0))
            mu2, logvar2 = vae_model.model.encode(image2.unsqueeze(0))
            
            # TODO: should we ignore the color dimension forcefully when calcultaing variance?
            diff_vec = (mu1 - mu2).squeeze()
            diff_vecs.append(diff_vec)
            
        diff_vecs = torch.stack(diff_vecs)
        dim_wise_variance = diff_vecs.pow(2).sum(0)/(len(pairs)-1)
        most_varied_dim = torch.argmax(dim_wise_variance)
        return diff_vecs.cpu().numpy(), dim_wise_variance.cpu().numpy(), most_varied_dim.item()

