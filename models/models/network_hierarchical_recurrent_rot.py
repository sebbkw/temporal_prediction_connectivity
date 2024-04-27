import torch
import torch.nn as nn

from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

class NetworkHierarchicalRecurrentRot (NetworkHierarchicalRecurrent):
    # Hidden units is an array of ints corresponding to
    # number of hidden units in each group
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrentRot, self).__init__(hyperparameters)

        self.output_units_groups     = [4]
        self.output_units_groups_len = len(self.output_units_groups)
        self.output_units            = sum(self.output_units_groups)

        self.fc = nn.Linear(
            in_features = self.hidden_units,
            out_features = self.output_units
        )

        self.initialize_weights_biases()
        self.set_weight_masks()

    def preprocess_data (self, data):
        x, y = data

        noise = torch.randn(x.shape)*0.5011
        x += noise

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
