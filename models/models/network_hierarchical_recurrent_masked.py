import torch
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

class NetworkHierarchicalRecurrentMasked (NetworkHierarchicalRecurrent):
    # Hidden units is an array of ints corresponding to
    # number of hidden units in each group
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrentMasked, self).__init__(hyperparameters)

    def preprocess_data (self, data):
        x, y = data

        #noise = torch.randn(x.shape)*0.5011
        #x += noise
        #y += noise

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
