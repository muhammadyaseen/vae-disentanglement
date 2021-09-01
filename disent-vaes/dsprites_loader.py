import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

FILE_NAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


class DSpritesDataset(Dataset):

    def __init__(self, root, split="train"):
        """
        Args:
            root (string): Directory with the .npz file.
        """
        self.root_dir = root

        dataset_zip = np.load(os.path.join(root, FILE_NAME),
                              allow_pickle=True, encoding='latin1')

        self.images = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']

        metadata = dataset_zip['metadata'][()]
        self.latents_names = metadata['latents_names']

        if split == "train":
            self.images = self.images[: len(self.images) - 100]

        if split == "test":
            self.images = self.images[len(self.images) - 100:]

    def __len__(self):
        return len(self.latents_values)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = transforms.ToTensor()(self.images[idx])
        latent = self.latents_values[idx]

        return image, latent
