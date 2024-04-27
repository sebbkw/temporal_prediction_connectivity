#### Import modules ###

import sys
import numpy as np
import torch
import torch.nn as nn

from models.network_hierarchical_recurrent         import NetworkHierarchicalRecurrent
from models.network_hierarchical_recurrent_rot     import NetworkHierarchicalRecurrentRot
from models.network_hierarchical_recurrent_masked  import NetworkHierarchicalRecurrentMasked
from models.network_hierarchical_recurrent_denoise import NetworkHierarchicalRecurrentDenoise
from models.network_hierarchical_recurrent_autoencoder import NetworkHierarchicalRecurrentAutoencoder
from models.network_hierarchical_recurrent_st_autoencoder import NetworkHierarchicalRecurrentSTAutoencoder
from models.network_feedforward import NetworkFeedforward

DEVICE = 'cpu'

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

# Get model responses
def get_rnn_responses(stimuli_processed, model):  
    stimuli_reshaped = stimuli_processed.reshape(1, -1, 36*36)
    
    with torch.no_grad():
        if isinstance(model, NetworkFeedforward):
            padded_stim = pad_matrix(torch.tensor(stimuli_reshaped), 5).type(torch.FloatTensor)
            padded_stim = torch.cat([torch.zeros(1, 4, padded_stim.shape[2], dtype=torch.float), padded_stim], dim=1)
            _, hidden_states = model(padded_stim)
        else:
            _, hidden_states = model(torch.Tensor(stimuli_reshaped))
            
        if isinstance(model, NetworkFeedforward):
            rnn_model_responses = hidden_states[0].numpy()[:, :, :]
        else:
            rnn_model_responses = hidden_states.numpy()[:, :, :]
            
        print(rnn_model_responses.shape)
        
                    
    return rnn_model_responses[0]


# Load raw natural movie stimuli
mov_stimuli = np.load(
    '../rnn/neural_fitting/neural_data/VISp/715093703_stimuli_raw.npy',
    allow_pickle=True
).item()


paths = [
    #'./model_checkpoints/replication_Feb23-02-12/2000-epochs_model.pt',
    #'./model_checkpoints/rot_-6L1_Feb26-00-51/1000-epochs_model.pt',
    #'/media/seb/Elements/rnn_refactor_model_checkpoints/masked_-6L1_Feb24-18-25/2000-epochs_model.pt',
    #'/media/seb/Elements/rnn_refactor_model_checkpoints/denoise_-6L1_Mar 1-19-52/2000-epochs_model.pt',
    #'./model_checkpoints/autoencoder_-5L1_Apr18-19-35/2000-epochs_model.pt',
    #'/media/seb/Elements/rnn_refactor_model_checkpoints/autoencoder_-5.5L1_Mar 5-00-11/2000-epochs_model.pt',
    #'./model_checkpoints/ff_-6.5L1_Mar15-17-03/1350-epochs_model.pt'
    #'/home/seb/rnn_refactor/model_checkpoints/denoise_SNR3dataset_-6.0L1_Apr24-00-33/2000-epochs_model.pt'
    '/home/seb/rnn_refactor/model_checkpoints/st_autoencoder_-5.5L1_Apr24-11-02/2000-epochs_model.pt'
]

nets = [
    #NetworkHierarchicalRecurrent,
    #NetworkHierarchicalRecurrentRot,
    #NetworkHierarchicalRecurrentMasked,
    #NetworkHierarchicalRecurrentDenoise,
    #NetworkHierarchicalRecurrentAutoencoder,
    NetworkHierarchicalRecurrentSTAutoencoder,
    #NetworkFeedforward
]

names = ['st_autoencoder']

for path, net, name in zip(paths, nets, names):        
    print('Starting', path)
    
    model, _, _ = net.load(model_path=path, device=DEVICE)
        
    model_outputs = {}
    
    for stim, stim_data in mov_stimuli.items():
        stim_responses = get_rnn_responses(stim_data, model)
        
        print(stim_responses.shape)
        model_outputs[stim] = stim_responses
        
    save_path = f'/media/seb/Elements/rnn_neural_fitting/model_data/rnn_refactor_final_{name}_raw_noPCA.npy'
    np.save(save_path, model_outputs)
    print(save_path)
