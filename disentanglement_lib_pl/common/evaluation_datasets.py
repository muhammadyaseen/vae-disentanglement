from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import util
import numpy as np
from tensorflow.compat.v1 import gfile

class CorrelatedDSprites(dsprites.DSprites):
    """
    The ground-truth factors of variation are (in the default setting):
    0 - color (3 shades of gray)
    1 - shape (3 different values)
    2 - scale (6 different values)
    3 - orientation (40 different values)
    4 - position x (32 different values)
    5 - position y (32 different values)
    """
    COLOR = 0
    ORIENTATION = 3
    CORRELATED_DSPRITES_PATH = os.path.join(
        os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
        "dsprites_ndarray_co3sh3sc6or40x32y32_64x64.npz")
    
    def __init__(self, latent_factor_indices=None):
        # By default, all factors (including shape) are considered ground truth
        # factors.
        if latent_factor_indices is None:
            latent_factor_indices = list(range(6))
        self.latent_factor_indices = latent_factor_indices
        self.data_shape = [64, 64, 1]
        # Load the data so that we can sample from it.
        with gfile.Open(self.CORRELATED_DSPRITES_PATH, "rb") as data_file:
            data = np.load(data_file, allow_pickle=True)
            self.images = np.array(data["imgs"])
            self.factor_sizes = np.array(
                data["metadata"][()]["latents_sizes"], dtype=np.int64)
        self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state, in_pos=self.ORIENTATION, 
                                                        out_pos=self.COLOR, map_fn=self._orientation_to_color_map_fn)

    def sample_observations_from_factors(self, factors, random_state):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""

        # We have to ignore the color factor since it depends on rotation
        # and is not sequentially generated. Hence we reset it to full-white factor
        _factors = np.copy(factors) # Have to do this, otherwise we'd lose actual gt factor related to color
        _factors[:, 0] = np.zeros(len(_factors))
        all_factors = self.state_space.sample_all_factors(_factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)

    def _orientation_to_color_map_fn(orientation_values):

        def __get_color_from_orientation(orientation):
            
            FIFTY_PCT_BRIGHTNESS_RANGE = range(0,14)
            SEVENTY_FIVE_PCT_BRIGHTNESS_RANGE = range(14,27)
            FULL_BRIGHTNESS_RANGE = range(27,40)

            if orientation in FULL_BRIGHTNESS_RANGE:
                return 0
            if orientation in FIFTY_PCT_BRIGHTNESS_RANGE:
                return 1
            if orientation in SEVENTY_FIVE_PCT_BRIGHTNESS_RANGE:
                return 2
            raise NotImplemented()
        
        return np.vectorize(__get_color_from_orientation)(orientation_values)


def get_evaluation_dataset(dataset_name):

    if dataset_name == "dsprites_correlated":
        return CorrelatedDSprites([0, 1, 2, 3, 4, 5])
    
    raise NotImplementedError()
