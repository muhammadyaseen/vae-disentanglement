import os
import numpy as np
import glob
import argparse

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
        label = os.path.basename(self.files_list[idx]).split("_")[2].replace(".jpg", "")
        label = np.array([label_to_sides[label]])
        label = label.astype('float')
        return image, label

    @staticmethod
    def generate_data(path, N, add_noise=False):

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

            # This noise offsets the figure from center, if enabled            
            if add_noise:        
                noise_mu, noise_s = 3, 1
                x_offset = noise_mu * np.random.randn() + noise_s  
                y_offset = noise_mu * np.random.randn() + noise_s
                
                circle_center = (W/2 + x_offset, H/2 + y_offset)
                bounding_circle = (circle_center, circle_radius)

            draw.regular_polygon(bounding_circle, n_sides[chosen_shape], 
                         rotation=0,
                         fill=fill_color, 
                         outline=outline_color)
            
            im.save(os.path.join(path, f'image_{i}_{chosen_shape}.jpg'))


if __name__ == "__main__":

    # if called as main script, we generate data using cmdline args

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--path',  '-p',
                        dest="datapath",
                        help =  'folder path where images will be saved')
    parser.add_argument('--num',  '-n',
                        dest="num",
                        type=int,
                        help =  'how many images to generate ?')
    parser.add_argument('--add-noise',  '-a',
                        dest="add_noise",
                        action='store_true',
                        help =  'add noise in generative process ?')

    args, _ = parser.parse_known_args()

    ThreeShapesDataset.generate_data(
        path=args.datapath,
        N=args.num,
        add_noise=args.add_noise
    )