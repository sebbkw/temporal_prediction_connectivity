import torch
import torch.nn as nn
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

class NetworkHierarchicalRecurrentSTAutoencoder (NetworkHierarchicalRecurrent):
    # Hidden units is an array of ints corresponding to
    # number of hidden units in each group
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrentSTAutoencoder, self).__init__(hyperparameters)

        self.output_units_groups     = [self.frame_size*5] + self.hidden_units_groups[:-1]
        self.output_units_groups_len = len(self.output_units_groups)
        self.output_units            = sum(self.output_units_groups)

        self.fc = nn.Linear(
            in_features = self.hidden_units,
            out_features = self.output_units
        )

        self.set_weight_masks()


    def preprocess_data (self, data):
        x, y = data

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
