import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models.network_base import NetworkBase

def pad_matrix (m, pad_size):
    # repeat along last (feature) dimension
    m = torch.repeat_interleave(m.unsqueeze(0), pad_size, dim=0)

    # offseting by one for each repeat along the last dimensions
    m_stacked = []
    for i in range(pad_size):
        start_idx = i
        end_idx   = -(pad_size-i-1)

        if i != pad_size-1:
            m_stacked.append(m[start_idx, :, start_idx:end_idx])
        else:
            m_stacked.append(m[start_idx, :, start_idx:])

    # stack along last (feature) dimension
    m_stacked = torch.stack(m_stacked, dim=2)

    # now flatten last two dimensions
    m_stacked = torch.flatten(m_stacked, start_dim=2)

    return m_stacked

class NetworkFeedforward (NetworkBase):
    def __init__ (self, hyperparameters):
        super(NetworkFeedforward, self).__init__()

        self.loss                  = hyperparameters['loss']
        self.lam                   = hyperparameters['lam']
        self.frame_size            = hyperparameters['frame_size']
        self.warmup                = hyperparameters['warmup']
        self.device                = hyperparameters['device']

        self.beta_weights          = [float(i) for i in hyperparameters['beta_weights'].split(',') if len(i)]

        self.hidden_units_groups     = [int(i) for i in hyperparameters['hidden_units_groups'].split(',') if len(i)]
        self.hidden_units             = sum(self.hidden_units_groups)

        self.output_units_groups     = [self.frame_size]
        self.output_units_groups_len = len(self.output_units_groups)
        self.output_units            = sum(self.output_units_groups)

        self.padding = 5

        self.fc1 = nn.Linear(
            in_features = self.frame_size*self.padding,
            out_features = self.hidden_units
        )
        self.fc2 = nn.Linear(
            in_features = self.hidden_units,
            out_features = self.output_units
        )

        self.ReLU = nn.ReLU()

    def preprocess_data (self, data):
        x, y = data

        #noise = torch.randn(x.shape)*0.5011
        #x += noise
        #y += noise

        x = pad_matrix(x, self.padding)
        y = y[:, self.padding-1:]

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y

    def forward (self, inputs):
        # Forward pass
        h = self.ReLU(self.fc1(inputs))
        fc_outputs = self.fc2(h)

        return fc_outputs, [h]
