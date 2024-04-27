import torch
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

class NetworkHierarchicalRecurrentAutoencoder (NetworkHierarchicalRecurrent):
    # Hidden units is an array of ints corresponding to
    # number of hidden units in each group
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrentAutoencoder, self).__init__(hyperparameters)

        self.lam_activity = hyperparameters['lam_activity']

    def preprocess_data (self, data):
        x, y = data

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
