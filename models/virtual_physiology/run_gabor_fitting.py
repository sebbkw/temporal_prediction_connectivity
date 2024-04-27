import sys
import torch
import numpy as np
import scipy.stats as stats

sys.path.append("../")
#from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent as Network
#from models.network_hierarchical_recurrent_autoencoder import NetworkHierarchicalRecurrentAutoencoder as Network
from models.network_hierarchical_recurrent_denoise import NetworkHierarchicalRecurrentDenoise as Network

from VirtualNetworkPhysiology import VirtualPhysiology

# Import gabor fitting module
sys.path.append("/media/seb/Elements/gabor_fit/")
from gabor_fit import fit

# Script wide variables
DEVICE     = 'cpu'

MODEL_PATH = ''
VPHYS_PATH = ''
SAVE_PATH  = ''

def get_mse_loss (y, y_est):
    return np.sum((y-y_est)**2)/ len(y)

# Load network checkpoint
model, hyperparameters, _ = Network.load(
    model_path=MODEL_PATH, device=DEVICE
)

# Instantiate new VirtualPhysiology object
vphys = VirtualPhysiology.load(
    data_path=VPHYS_PATH,
    model=model,
    hyperparameters=hyperparameters,
    frame_shape=(36, 36),
    hidden_units=[2592],
    device=DEVICE
)

# Get group 1 vphys data or input weights
physiology_data = vphys.data[0]
input_weights = model.rnn.weight_ih_l0.cpu().detach().numpy()

# Instantiate gabor fitting object
gf = fit.GaborFit()

# Loop through each unit and get relevant fitting params
for idx, unit in enumerate(physiology_data):
    print('\nStarting unit', idx, '/', len(physiology_data))
    unit_iw = unit['response_weighted_average'].reshape(36, 36, 1)

    _, pp, _ = gf.fit(unit_iw)
    x, y, theta, phi, freq, lvx, lvy = pp
    pp = fit.combine_params(*pp)
    x, y, theta, phi, lkx, lvx, lvy = pp

    fitted_gabor = gf.make_gabor(pp, *vphys.frame_shape)

    unit['gabor_x']         = y
    unit['gabor_y']         = x
    unit['gabor_params']    = (1, y, x, theta, lvx, lvy, freq[0], phi)
    unit['gabor_est']       = fitted_gabor[:, :, 0]
    unit['gabor_r']         = stats.pearsonr(fitted_gabor.flatten(), unit_iw.flatten())[0]
    unit['gabor_loss']      = get_mse_loss(fitted_gabor.flatten(), unit_iw.flatten())

vphys.save(SAVE_PATH)
