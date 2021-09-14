import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

FILE_NAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


class DSpritesDataset(Dataset):

    def __init__(self, root, split="train", transform=None):
        """
        Args:
            root (string): Directory with the .npz file.
        """
        self.root_dir = root
        self.transform = transform

        dataset_zip = np.load(os.path.join(root, FILE_NAME),
                              allow_pickle=True, encoding='latin1')

        self.images = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']

        metadata = dataset_zip['metadata'][()]
        self.latents_names = metadata['latents_names']

        # save about 25% as validation.
        MAX_TRAIN_IDX = int( len(self.images) * 0.75 )
        print(MAX_TRAIN_IDX)

        if split == "train":
            self.images = self.images[: MAX_TRAIN_IDX]
            self.latents_values = self.latents_values[: MAX_TRAIN_IDX]

        if split == "test":
            self.images = self.images[MAX_TRAIN_IDX:]
            self.latents_values = self.latents_values[MAX_TRAIN_IDX:]
            #self.images = self.images[MAX_TRAIN_IDX: MAX_TRAIN_IDX + 100]
            #self.latents_values = self.latents_values[MAX_TRAIN_IDX: MAX_TRAIN_IDX + 100]

    def __len__(self):
        return len(self.latents_values)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #three_channel_image = np.stack((self.images[idx], self.images[idx], self.images[idx]), axis=-1).astype(np.float64)
        image = self.images[idx].astype(np.float64)
        image = transforms.ToTensor()(image)

        if self.transform is not None:
            image = self.transform(image)

        latent = self.latents_values[idx]

        return image, latent
