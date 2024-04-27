import torch
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

class NetworkHierarchicalRecurrentTemporalPrediction (NetworkHierarchicalRecurrent):
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrentTemporalPrediction, self).__init__(hyperparameters)

    def preprocess_data (self, data):
        x, y = data

        noise = torch.randn(x.shape)*0.5011
        x += noise
        y += noise

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
