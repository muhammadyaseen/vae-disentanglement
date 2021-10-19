import os
import numpy as np

import glob

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from PIL import Image, ImageDraw
from torchvision.transforms import transforms


class ContinumDataset(Dataset):

    def __init__(self, root, split="train", train_pct=0.90):
        """
        Args:
            root (string): Directory with all the images.
        """
        self.root_dir = root
        self.files_list = glob.glob(os.path.join(self.root_dir, "*.jpg"))
        print("Total images :", len(self.files_list))
        MAX_TRAIN_IDX = int(len(self.files_list) * train_pct)

        if split == "train":
            self.files_list = self.files_list[: MAX_TRAIN_IDX]
            print("Selected for train: ", len(self.files_list))
        if split == "test":
            self.files_list = self.files_list[MAX_TRAIN_IDX:]
            print("Selected for val: ", len(self.files_list))

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files_list[idx]
        image = plt.imread(img_name)
        image = transforms.ToTensor()(image)

        label = int(os.path.basename(self.files_list[idx]).split("_")[2].replace(".jpg", ""))
        label = np.array([label])
        label = label.astype('float')
        return image, label

    @staticmethod
    def generate_data(path, N, x_offset=0, y_offset=0):

        # white, black, black
        fill_color, outline_color, bg_color = 255, 0, 0
        W, H = 64, 64

        for i in range(N):

            # single channel black/white image. Should use cmap='gray' when showing in matplotlib
            im = Image.new('L', (W, H), bg_color)
            draw = ImageDraw.Draw(im)
            rect_bounds = (W / 4 + x_offset,
                           W / 4 + y_offset,
                           W / (4/3) + x_offset,
                           W / (4/3) + y_offset)

            # Note: Radius should be modified if the shape proportions change
            rnd_radius = np.random.choice(range(1, 20, 4))
            draw.rounded_rectangle(rect_bounds,
                                   radius=rnd_radius,
                                   fill=fill_color,
                                   outline=outline_color)

            im.save(os.path.join(path,f'image_{i}_{rnd_radius}.jpg'))
