import os
import numpy as np

import glob

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from PIL import Image, ImageDraw
from torchvision.transforms import transforms


class ThreeShapesDataset(Dataset):

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

        label_to_sides = { 'triangle': 3,
                            'square': 4,
                            'circle': 500
                        }
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files_list[idx]
        image = plt.imread(img_name)
        image = transforms.ToTensor()(image)

        # get filename e.g. image_257_circle.jpg and return 'circle'
        label = int(os.path.basename(self.files_list[idx]).split("_")[2].replace(".jpg", ""))
        label = np.array([label_to_sides[label]])
        label = label.astype('float')
        return image, label

    @staticmethod
    def generate_data(path, N, x_offset=0, y_offset=0):

        # white, black, black
        fill_color, outline_color, bg_color = 255, 0, 0
        W, H = 64, 64

        circle_radius = 24
        circle_center = (W/2,H/2)
        bounding_circle = (circle_center, circle_radius)

        n_sides = { 'triangle': 3,
                    'square': 4,
                    'circle': 500
                }

        for i in range(N):

            # single channel black/white image. Should use cmap='gray' when showing in matplotlib
            im = Image.new('L', (W, H), bg_color)
            draw = ImageDraw.Draw(im)
            
            chosen_shape = np.random.choice(list(n_sides.keys()))

            draw.regular_polygon( bounding_circle, n_sides[chosen_shape], 
                         rotation=0,
                         fill=fill_color, 
                         outline=outline_color)
            
            im.save(os.path.join(path, f'image_{i}_{chosen_shape}.jpg'))
