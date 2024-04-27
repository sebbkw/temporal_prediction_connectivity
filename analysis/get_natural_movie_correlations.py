import sys
import torch
import math
import numpy as np
import scipy.stats as stats
    
from connectivity_functions import *
from plotting_functions import *
    
MODEL_PATH = ''
VPHYS_PATH  = ''
MODEL_NAME  = ''

sys.path.append("../")
from models.models.network_hierarchical_recurrent_temporal_prediction import NetworkHierarchicalRecurrentTemporalPrediction
from models.data.dataset import data_loader
from virtual_physiology.VirtualNetworkPhysiology import VirtualPhysiology
from plotting_functions import *

# Load network checkpoint
model, hyperparameters, _ = Network.load(
    model_path=MODEL_PATH, device='cpu'
)

# Instantiate new VirtualPhysiology object
vphys = VirtualPhysiology.load(
    data_path=VPHYS_PATH,
    model=model,
    hyperparameters=hyperparameters,
    frame_shape=(36,36),
    hidden_units=[36*36*2],
    device='cpu'
)

dataset_path = ""
test_data_loader = data_loader(
    dataset_path,
    split='validation',
    batch_size=100
)




all_hidden = []

for batch_n, data in enumerate(test_data_loader):
    print('Starting', batch_n)
    x, y = data
    
    with torch.no_grad():
        outputs, hidden_states = model(x)
        all_hidden.append(hidden_states.detach().cpu().numpy())

all_hidden = np.concatenate(all_hidden)

print(all_hidden.shape)





total_units   = 36*36*2
excit_units_n = int(total_units*0.9)
weight_matrix = model.rnn.weight_hh_l0[:excit_units_n, :excit_units_n].detach().numpy()
threshold = np.percentile(weight_matrix.reshape(-1), 95)

all_units     = []
for unit in vphys.data[0]:
    if unit['hidden_unit_index'] < excit_units_n:
        r = unit['gabor_r']
        sx, sy = unit['gabor_params'][4:6]
        
        if unit['gabor_r'] > 0.7 and sx>0.5 and sy>0.5:
            all_units.append(unit)
        

res_corr = []
is_connected_arr = []
conn_str = []

for pre_unit_idx, pre_unit in enumerate(all_units):
    pre_unit_res = all_hidden[:, :, pre_unit['hidden_unit_index']]
    if (pre_unit_idx % 10) == 0:
        print('Starting unit', pre_unit_idx, '/',  len(all_units))
        
    for post_unit_idx, post_unit in enumerate(all_units):        
        is_different_unit = pre_unit_idx != post_unit_idx
        is_connected, w   = get_is_connected(pre_unit, post_unit, weight_matrix, threshold)
        is_in_range       = get_is_in_range(pre_unit, post_unit, 'all')

        post_unit_res = all_hidden[:, :, post_unit['hidden_unit_index']]

        if is_different_unit:
            mn_cc = []
            
            for pre, post in zip(pre_unit_res[:100], post_unit_res[:100]):
                cc = stats.pearsonr(pre, post)[0]
                if not np.isnan(cc):
                    mn_cc.append(cc)
            
            res_corr.append(np.mean(mn_cc))
            is_connected_arr.append(int(is_connected))
            conn_str.append(w)
                
res_corr         = np.array(res_corr)
is_connected_arr = np.array(is_connected_arr)
conn_str         = np.array(conn_str)


np.save(f'./movie_correlation_data/nat_mov_conn_data_{MODEL_NAME}.npy', {
    'res_corr': res_corr,
    'is_connected_arr': is_connected_arr,
    'conn_str': conn_str
})
