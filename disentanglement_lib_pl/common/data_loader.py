import os
import numpy as np
import logging

import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from common import constants as c
from common.known_datasets import CorrelatedDSpritesDataset, ThreeShapesDataset, OneDimLatentDataset, ContinumDataset, PolynomialDataset, DSpritesDataset

class LabelHandler(object):
    def __init__(self, labels, label_weights, class_values):
        self.labels = labels
        self._label_weights = None
        self._num_classes_torch = torch.tensor((0,))
        self._num_classes_list = [0]
        self._class_values = None
        if labels is not None:
            self._label_weights = [torch.tensor(w) for w in label_weights]
            self._num_classes_torch = torch.tensor([len(cv) for cv in class_values])
            self._num_classes_list = [len(cv) for cv in class_values]
            self._class_values = class_values

    def label_weights(self, i):
        return self._label_weights[i]

    def num_classes(self, as_tensor=True):
        if as_tensor:
            return self._num_classes_torch
        else:
            return self._num_classes_list

    def class_values(self):
        return self._class_values

    def get_label(self, idx):
        if self.labels is not None:
            return torch.tensor(self.labels[idx], dtype=torch.long)
        return None

    def has_labels(self):
        return self.labels is not None


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transforms, labels, label_weights, name, class_values, num_channels, seed):
        super(CustomImageFolder, self).__init__(root, transforms)
        self.indices = range(len(self))
        self._num_channels = num_channels
        self._name = name
        self.seed = seed

        self.label_handler = LabelHandler(labels, label_weights, class_values)

    @property
    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        path1 = self.imgs[index1][0]
        img1 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform(img1)

        label1 = 0
        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
        return img1, label1


class CustomNpzDataset(Dataset):
    def __init__(self, data_images, transform, labels, label_weights, name, class_values, num_channels, seed):
        self.seed = seed
        self.data_npz = data_images
        self._name = name
        self._num_channels = num_channels

        self.label_handler = LabelHandler(labels, label_weights, class_values)

        self.transform = transform
        self.indices = range(len(self))

    @property
    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        img1 = Image.fromarray(self.data_npz[index1] * 255)
        if self.transform is not None:
            img1 = self.transform(img1)

        label1 = 0
        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
        return img1, label1

    def __len__(self):
        return self.data_npz.shape[0]


class DisentanglementLibDataset(Dataset):
    """
    Data-loading from Disentanglement Library

    Note:
        Unlike a traditional Pytorch dataset, indexing with _any_ index fetches a random batch.
        What this means is dataset[0] != dataset[0]. Also, you'll need to specify the size
        of the dataset, which defines the length of one training epoch.

        This is done to ensure compatibility with disentanglement_lib.
    """

    def __init__(self, name, seed=0):
        """
        Parameters
        ----------
        name : str
            Name of the dataset use. You may use `get_dataset_name`.
        seed : int
            Random seed.
        iterator_len : int
            Length of the dataset. This defines the length of one training epoch.
        """
        from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset = get_named_ground_truth_data(self.name)
        self.iterator_len = self.dataset.images.shape[0]

    @staticmethod
    def has_labels():
        return False


    def num_channels(self):
        return self.dataset.observation_shape[2]

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, item):
        assert item < self.iterator_len
        output = self.dataset.sample_observations(1, random_state=self.random_state)[0]
        # Convert output to CHW from HWC
        return torch.from_numpy(np.moveaxis(output, 2, 0), ).type(torch.FloatTensor), 0


def _get_transforms_for_dataset(dataset_name, image_size):

    if dataset_name == "celeba":
        return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])
    
    # for these datasets, we only need to convert numpy to tensors.
    if dataset_name in ["dsprites_full", "dsprites_correlated", "dsprites_colored", "dsprites_cond",  
                        "threeshapes", "threeshapesnoisy", "onedim", "continum"]:
        return transforms.ToTensor()

    return None


def _get_dataloader_with_labels(dataset_name, dset_dir, batch_size, seed, num_workers, image_size, include_labels, pin_memory,
                                shuffle, droplast, split, train_pct):
        
    dataset_name = dataset_name.lower()
        
    labels = None
    label_weights = None
    label_idx = None
    label_names = None
    class_values = None

    # check if labels are provided as indices or names
    if include_labels is not None:
        try:
            int(include_labels[0])
            label_idx = [int(s) for s in include_labels]
        except ValueError:
            label_names = include_labels
    logging.info('include_labels: {}'.format(include_labels))

    # TODO: is there a better way to do this ???
    if dataset_name == 'celeba':
        
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
                       'name': dataset_name,
                       'seed': seed}
    
        dset = CustomImageFolder
    
    elif dataset_name in ['dsprites_full', 'dsprites_colored', 'dsprites_cond']:
        
        root = os.path.join(dset_dir, 'dsprites')

        """
        if label_idx is not None:
            labels = npz['latents_values'][:, label_idx]
            if 1 in label_idx:
                index_shape = label_idx.index(1)
                labels[:, index_shape] -= 1

            # dsprite has uniformly distributed labels
            num_labels = labels.shape[1]
            label_weights = []
            class_values = []
            for i in range(num_labels):
                unique_values, count = np.unique(labels[:, i], axis=0, return_counts=True)
                weight = 1 - count / labels.shape[0]
                if len(weight) == 1:
                    weight = np.array(1)
                else:
                    weight /= sum(weight)
                label_weights.append(np.array(weight))

                # always set label values to integers starting from zero
                unique_values_mock = np.arange(len(unique_values))
                class_values.append(unique_values_mock)
            label_weights = np.array(label_weights)
        """

        data_kwargs = {'root': root,
                       'train_pct': train_pct,
                       'split': split,
                       'correlated': dataset_name == 'dsprites_correlated' or dataset_name == 'dsprites_colored',
                       'colored': dataset_name == 'dsprites_colored',
                       'conditioned': dataset_name == 'dsprites_cond'}
        
        dset = DSpritesDataset
    
    elif dataset_name == 'threeshapes' or dataset_name == "threeshapesnoisy":
        
        root = os.path.join(dset_dir, dataset_name)

        data_kwargs = {'root': root,
                       'train_pct': train_pct,
                       'split': split}
        
        dset = ThreeShapesDataset

    elif dataset_name == 'onedim':
    
        root = os.path.join(dset_dir, dataset_name)

        data_kwargs = {'root': root,
                       'train_pct': train_pct,
                       'split': split}
        
        dset = OneDimLatentDataset
    
    elif dataset_name == 'continum':
    
        root = os.path.join(dset_dir, dataset_name)

        data_kwargs = {'root': root,
                       'train_pct': train_pct,
                       'split': split}
        
        dset = ContinumDataset

    elif dataset_name == 'polynomial':
    
        data_kwargs = {}
        
        dset = PolynomialDataset
    
    else:
        raise NotImplementedError
    
    transforms = _get_transforms_for_dataset(dataset_name, image_size)
    data_kwargs.update({'transforms': transforms})
    #print(data_kwargs)
    dataset = dset(**data_kwargs)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=droplast)

    if include_labels is not None:
        logging.info('num_classes: {}'.format(dataset.num_classes(False)))
        logging.info('class_values: {}'.format(class_values))

    return data_loader


def _get_dataloader(name, batch_size, seed, num_workers, pin_memory, shuffle, droplast):
    """
    Makes a dataset using the disentanglement_lib.data.ground_truth functions, and returns a PyTorch dataloader.
    Image sizes are fixed to 64x64 in the disentanglement_lib.
    :param name: Name of the dataset use. Should match those of disentanglement_lib
    :return: DataLoader
    """
    dataset = DisentanglementLibDataset(name, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast, pin_memory=pin_memory,
                        num_workers=num_workers,)
    return loader

def _get_dataloader_disentlib(name, batch_size, seed, num_workers, pin_memory, shuffle, droplast, **kwargs):
    """
    Special case of `_get_dataloader` for correlated datasets so that we can pass additional params
    Makes a dataset using the disentanglement_lib.data.ground_truth functions, and returns a PyTorch dataloader.
    Image sizes are fixed to 64x64 in the disentanglement_lib.
    :param name: Name of the dataset use. Should match those of disentanglement_lib
    :return: DataLoader
    """


    dataset = CorrelatedDSpritesDataset(correlation_strength=0.3, seed=seed, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast, pin_memory=pin_memory,
                        num_workers=num_workers,)
    return loader

def get_dataloader(dset_name, dset_dir, batch_size, seed, num_workers, image_size, include_labels, pin_memory,
                   shuffle, droplast, split="train", train_pct=0.90):
    
    #locally_supported_datasets = c.DATASETS[0:10]

    logging.info(f'Datasets root: {dset_dir}')
    logging.info(f'Dataset: {dset_name}')
    logging.info("Locally supported: " + ','.join(c.KNOWN_DATASETS))
    logging.info("Locally supported disentlib: " + ','.join(c.KNOWN_DISENTLIB_DATASETS))

    if dset_name in c.KNOWN_DATASETS:
        return _get_dataloader_with_labels(dset_name, dset_dir, batch_size, seed, num_workers, image_size,
                                           include_labels, pin_memory, shuffle, droplast, split, train_pct)
    elif dset_name in c.KNOWN_DISENTLIB_DATASETS:
        return _get_dataloader_disentlib(dset_name, batch_size, seed, num_workers, pin_memory, shuffle, droplast, split=split)
    else:
        # use the dataloader of Google's disentanglement_lib
        return _get_dataloader(dset_name, batch_size, seed, num_workers, pin_memory, shuffle, droplast)

