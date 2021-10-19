import os
import numpy as np

import glob

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import PIL
from torchvision.transforms import transforms


class OneDimLatentDataset(Dataset):

    def __init__(self, root, split="train"):
        """
        Args:
            root (string): Directory with all the images.
        """
        self.root_dir = root
        self.files_list = glob.glob(os.path.join(self.root_dir,"*.jpg"))
        #print(self.files_list)
        print("Total images :", len(self.files_list))
        
        MAX_TRAIN_IDX = int( len(self.files_list) * 0.90 )
        print("Max train idx: ", MAX_TRAIN_IDX)
        
        if split == "train":
            self.files_list = self.files_list[: MAX_TRAIN_IDX]
            print("Selected for train: ",len(self.files_list))
        if split == "test":
            self.files_list = self.files_list[MAX_TRAIN_IDX:]
            print("Selected for val: ",len(self.files_list))
        #exit()

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files_list[idx]
        image = plt.imread(img_name)
        #image = torch.from_numpy(image).type(torch.FloatTensor).reshape(3,64,64)
        image = transforms.ToTensor()(image)
        #image = torch.unsqueeze(image, 0)

        label = int(self.files_list[idx].split("_")[1].replace(".jpg",""))
        label = np.array([label])
        label = label.astype('float')
        return image, label
