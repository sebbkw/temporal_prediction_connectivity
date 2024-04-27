import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

class NetworkHierarchicalRecurrentDenoise (NetworkHierarchicalRecurrent):
    # Hidden units is an array of ints corresponding to
    # number of hidden units in each group
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrentDenoise, self).__init__(hyperparameters)

    def preprocess_data (self, data):
        x, y = data

        snr = 3 # 0
        noise = torch.rand(size=x.shape) * 1/(10**(snr/20))
        #noise = np.random.normal(size=x.shape) * 1/(10**(snr/20))
        #noise = gaussian_filter(noise, sigma=(0, 2, 0))
        #noise = torch.from_numpy(noise).type(torch.FloatTensor)
        x += noise

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
