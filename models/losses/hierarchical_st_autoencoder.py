import torch.nn as nn
from utils import pad_matrix

def hierarchical_st_autoencoder (self, out, data):
    pad_size = 5

    predictions, hidden = out
    _, frame_targets    = data
    padded_targets      = pad_matrix (frame_targets, pad_size)

    MSEs = [nn.functional.mse_loss(predictions[:, pad_size-1:], padded_targets)]

    ret = { 'L1': self.L1() }

    loss = 0
    loss += ret['L1']

    for group_idx, (beta, MSE) in enumerate(zip(self.beta_weights, MSEs)):
        ret[f'mse{group_idx}'] = MSE
        loss += beta*MSE

    return loss, ret
