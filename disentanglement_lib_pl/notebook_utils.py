import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple
from tqdm import tqdm

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