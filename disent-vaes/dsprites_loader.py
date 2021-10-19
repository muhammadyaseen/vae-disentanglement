import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

FILE_NAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


class DSpritesDataset(Dataset):

    def __init__(self, root, split="train", transform=None, train_pct=0.75):
        """
        Args:
            root (string): Directory with the .npz file.
        """
        self.root_dir = root
        self.transform = transform
        self.split = split

        dataset_zip = np.load(os.path.join(root, FILE_NAME),
                              allow_pickle=True, encoding='latin1')

        self.images = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']

        metadata = dataset_zip['metadata'][()]
        self.latents_names = metadata['latents_names']

        MAX_TRAIN_IDX = int(len(self.images) * train_pct)

        # get shuffled indices of training samples
        self.train_indices = np.random.choice(len(self.images),
                         size=MAX_TRAIN_IDX,
                         replace=False)

        # disjoint (shuffled) validation set indices
        self.test_indices = np.random.permutation(
                                np.setdiff1d(
                                    range(len(self.images)), self.train_indices
                                )
                            )

        #if split == "train":
        #    self.images = self.images[: MAX_TRAIN_IDX]
        #    self.latents_values = self.latents_values[: MAX_TRAIN_IDX]

        #if split == "test":
        #    self.images = self.images[MAX_TRAIN_IDX:]
        #    self.latents_values = self.latents_values[MAX_TRAIN_IDX:]

    def __len__(self):

        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "test":
            return len(self.test_indices)
        else:
            Exception("Unknown split type")


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.split == "train":
            idx = self.train_indices[idx]

        if self.split == "test":
            idx = self.test_indices[idx]

        image = self.images[idx].astype(np.float32)
        image = transforms.ToTensor()(image)

        if self.transform is not None:
            image = self.transform(image)

        latent = self.latents_values[idx]

        return image, latent
