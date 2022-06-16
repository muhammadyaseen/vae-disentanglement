import os
import numpy as np
import glob
import gin.tf

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.distributions import normal
from common import constants as c


class OneDimLatentDataset(Dataset):

    def __init__(self, root, split="train",transforms=None):
        """
        Args:
            root (string): Directory with all the images.
        """
        self.root_dir = root
        self.files_list = glob.glob(os.path.join(self.root_dir,"*.jpg"))
        self.split = split
        self.trasnforms = transforms

        MAX_TRAIN_IDX = int( len(self.files_list) * 0.90 )
        # get shuffled indices of training samples
        self.train_indices = np.random.choice(len(self.files_list),
                         size=MAX_TRAIN_IDX,
                         replace=False)

        # disjoint (shuffled) validation set indices
        self.test_indices = np.random.permutation(
                                np.setdiff1d(
                                    range(len(self.files_list)), self.train_indices
                                )
                            )
        # if split == "train":
        #     self.files_list = self.files_list[: MAX_TRAIN_IDX]
        
        # if split == "test":
        #     self.files_list = self.files_list[MAX_TRAIN_IDX:]

    def __len__(self):
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "test":
            return len(self.test_indices)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.split == "train":
            idx = self.train_indices[idx]

        if self.split == "test":
            idx = self.test_indices[idx]
        
        img_name = self.files_list[idx]
        image = plt.imread(img_name)
        
        if self.trasnforms is not None:
            image = self.transforms(image)

        label = int(self.files_list[idx].split("_")[1].replace(".jpg",""))
        label = np.array([label])
        label = label.astype('float')
        return image, label

class ContinumDataset(Dataset):
    """
    This dataset has images which form a 'continum' from a perfect square 
    morphing into a circle, as border radius increases
    """
    def __init__(self, root, split="train", train_pct=0.90, transforms=None):
        """
        Args:
            root (string): Directory with all the images.
        """
        self.root_dir = root
        self.files_list = glob.glob(os.path.join(self.root_dir, "*.jpg"))
        self.split = split
        self.transforms = transforms
        MAX_TRAIN_IDX = int(len(self.files_list) * train_pct)
        
        # get shuffled indices of training samples
        self.train_indices = np.random.choice(len(self.files_list),
                         size=MAX_TRAIN_IDX,
                         replace=False)

        # disjoint (shuffled) validation set indices
        self.test_indices = np.random.permutation(
                                np.setdiff1d(
                                    range(len(self.files_list)), self.train_indices
                                )
                            )
        # TODO: this is NOT random. glob will have some specific ordering 
        # of file names. Should convert it to random
        # if split == "train":
        #     self.files_list = self.files_list[: MAX_TRAIN_IDX]

        # if split == "test":
        #     self.files_list = self.files_list[MAX_TRAIN_IDX:]

    def __len__(self):
        
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "test":
            return len(self.test_indices)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.split == "train":
            idx = self.train_indices[idx]

        if self.split == "test":
            idx = self.test_indices[idx]
        
        img_name = self.files_list[idx]
        image = plt.imread(img_name)
        
        if self.transforms is not None:
            image = self.transforms(image)

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

class ThreeShapesDataset(Dataset):

    def __init__(self, root, split="train", train_pct=0.90, transforms=None):
        """
        Args:
            root (string): Directory with all the images.
        """
        self.root_dir = root
        self.files_list = glob.glob(os.path.join(self.root_dir, "*.jpg"))
        self.transforms = transforms
        self.split = split

        MAX_TRAIN_IDX = int(len(self.files_list) * train_pct)
        
        # get shuffled indices of training samples
        self.train_indices = np.random.choice(len(self.files_list),
                         size=MAX_TRAIN_IDX,
                         replace=False)

        # disjoint (shuffled) validation set indices
        self.test_indices = np.random.permutation(
                                np.setdiff1d(
                                    range(len(self.files_list)), self.train_indices
                                )
                            )


    def __len__(self):
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "test":
            return len(self.test_indices)
        else:
            Exception("Unknown split type")

    def __getitem__(self, idx):

        label_to_sides = { 'triangle': 3,
                            'square': 4,
                            'circle': 500
                        }
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.split == "train":
            idx = self.train_indices[idx]

        if self.split == "test":
            idx = self.test_indices[idx]
        
        img_name = self.files_list[idx]
        image = plt.imread(img_name)
        
        if self.transforms is not None:
            image = self.transforms(image)
        
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

    @staticmethod
    def generate_npz_data(path, N, add_noise=False):

        # white, black, black
        fill_color, outline_color, bg_color = 255, 0, 0
        W, H = 64, 64
        circle_radius = 24
        circle_center = (W/2,H/2)
        bounding_circle = (circle_center, circle_radius)

        n_sides = [3, 4, 500]

        numpy_images = np.zeros((N, W, H))
        numpy_latents = np.zeros(N)
        one_third = N // 3
        shape_ranges = {
            'triangle': range(0, one_third),
            'square':   range(one_third, 2*one_third),
            'circle':   range(2*one_third, N)
        }
        
        for i in range(N):

            # single channel black/white image. Should use cmap='gray' when showing in matplotlib
            img = Image.new('L', (W, H), bg_color)
            draw = ImageDraw.Draw(img)

            which_shape = np.argmax([i in shape_range for _, shape_range in shape_ranges.items()])

            # This noise offsets the figure from center, if enabled            
            if add_noise:        
                noise_mu, noise_s = 3, 1
                x_offset = noise_mu * np.random.randn() + noise_s  
                y_offset = noise_mu * np.random.randn() + noise_s

                circle_center = (W/2 + x_offset, H/2 + y_offset)
                bounding_circle = (circle_center, circle_radius)

            draw.regular_polygon(bounding_circle, n_sides[which_shape], 
                        rotation=0,
                        fill=fill_color, 
                        outline=outline_color)
            
            numpy_images[i] = np.array(img)
            numpy_latents[i] = which_shape
        
        np.savez_compressed(
            path,
            images=numpy_images, 
            latents=numpy_latents,
            ranges=np.array([
                [0, one_third],
                [one_third, 2*one_third],
                [2*one_third, N]
            ])
        )
    
class PolynomialDataset(Dataset):

    def __init__(self, num_points=10000, **kwargs):
        
        self.num_points = num_points

        norm_dist_x = normal.Normal(loc=0, scale=3., validate_args=True)
        norm_dist_y = normal.Normal(loc=0, scale=1., validate_args=True)

        samples_x = norm_dist_x.sample(sample_shape=(num_points,1))
        samples_y = norm_dist_y.sample(sample_shape=(num_points,1))

        self.x_y_joint_samples = torch.cat([samples_x, samples_y], dim=1)

        # polynom: x^2 + 2*x + 3*y + xy
        self.polynomial_points = torch.cat([
                            torch.pow(samples_x, 2), 
                            2 * samples_x,
                            3 * samples_y, 
                            samples_x * samples_y
                        ], dim=1
        )

    def __len__(self):
        return self.num_points
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.polynomial_points[idx]
        latent = self.x_y_joint_samples[idx]

        return y, latent

class DSpritesDataset(Dataset):

    FILE_NAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    CORRELATED_FILE_NAME = 'dsprites_ndarray_co3sh3sc6or40x32y32_64x64.npz'
    CORRELATED_3C_FILE_NAME = 'dsprites_ndarray_co3sh3sc6or40x32y32_64x64x3.npz'
    COND_FILE_NAME = 'dsprites_ndarray_co1sh3sc6or40x1y32_64x64.npz'

    def __init__(self, root, split="train", train_pct=0.90, 
        transforms=None, correlated=False, colored=False,
        conditioned=False):
        """
        Args:
            root (string): Directory with the .npz file.
        """

        assert transforms is not None, "need to give transform"

        self.root_dir = root
        self.transforms = transforms
        self.split = split
        self.correlated = correlated
        self.colored = colored
        self.conditioned = conditioned

        file_to_load = self.FILE_NAME
        
        if self.correlated and self.colored:
            print("Loading correlated and colored dataset")
            file_to_load = self.CORRELATED_3C_FILE_NAME 
        elif self.conditioned:
            print("Loading conditioned dataset")
            file_to_load = self.COND_FILE_NAME
        
        dataset_zip = np.load(os.path.join(self.root_dir, file_to_load),
                              allow_pickle=True, encoding='latin1')

        self.images = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']

        metadata = dataset_zip['metadata'][()]
        self.latents_names = metadata['latents_names']

        MAX_TRAIN_IDX = int(len(self.images) * train_pct)
        print(f"Loaded {len(self.images)} images")

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

        if self.transforms is not None:
            image = self.transforms(image)

        latent = self.latents_values[idx]

        #print(image.shape)
        #print(latent.shape)
        return image, latent

class CorrelatedDSpritesDataset(Dataset):

    def __init__(self, correlation_strength, seed=0, split='train'):
        """
        Parameters
        ----------
        seed : int
            Random seed
        """
        from disentanglement_lib.data.ground_truth import util as gt_utils
        from disentanglement_lib.data.ground_truth import named_data
        
        self.name = 'dsprites_correlated'
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset = named_data.get_named_ground_truth_data('dsprites_full')
        self.iterator_len = self.dataset.images.shape[0]
        self.correlation_strength = correlation_strength
        self.split = split

        if self.split == 'test':
            # we only want a subset when loading in test/validation mode 
            # this gives ~3.5% of data
            self.dataset.factor_sizes = [1, 3, 6, 15, 10, 10]
            self.iterator_len = np.prod(self.dataset.factor_sizes)
        
        # get correlated space and update the space of normal dsprites dataset
        gin.bind_parameter("correlation_hyperparameter.line_width", self.correlation_strength)
        correlated_state_space = gt_utils.CorrelatedSplitDiscreteStateSpace(
                                            factor_sizes=self.dataset.factor_sizes,
                                            latent_factor_indices=self.dataset.latent_factor_indices, 
                                            corr_indices=[3, 4], # orientation, posX
                                            corr_type='line'
                                        )
        
        self.dataset.state_space = correlated_state_space

        print(f"Initialize [CorrelatedDSpritesDataset] with {self.iterator_len} examples. Shape {self.dataset.images.shape}.")

    @staticmethod
    def has_labels():
        return False

    def num_channels(self):
        return self.dataset.observation_shape[2]

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, item):
        
        assert item < self.iterator_len
        
        factors, observations = self.dataset.sample(1, random_state=self.random_state)
        
        #print(factors.size, observations.size)
        #print(factors[0].shape, observations[0].shape)
        
        # Convert output to CHW from HWC
        # `sample` function returns data with an extra `batch` axis for some reason so 
        # we have to index into it with [0] to return just one example
        return torch.from_numpy(np.moveaxis(observations[0], 2, 0), ).type(torch.FloatTensor), factors[0]
    
class ToyDataset(Dataset):

    FILE_NAME = 'toydata_3x3_uc.npz'

    def __init__(self, root, split="train", train_pct=0.90, transforms=None):
        """
        Args:
            root (string): Directory with the .npz file.
        """

        assert transforms is not None, "need to give transform"

        self.root_dir = root
        self.transforms = transforms
        self.split = split

        dataset_zip = np.load(os.path.join(self.root_dir, self.FILE_NAME),
                              allow_pickle=True, encoding='latin1')

        self.images = dataset_zip['images']
        self.latents_values = dataset_zip['latents_values']
        #self.latents_classes = dataset_zip['latents_classes']

        metadata = dataset_zip['metadata'][()]
        self.latents_names = metadata['latents_names']

        MAX_TRAIN_IDX = int(len(self.images) * train_pct)
        print(f"Loaded {len(self.images)} images")

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

        # turn into a 1-D vec
        #image = torch.Tensor(image.reshape(image.shape[0] **2 ))

        if self.transforms is not None:
            image = self.transforms(image)

        latent = self.latents_values[idx]

        return image, latent