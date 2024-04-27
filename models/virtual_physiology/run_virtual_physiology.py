import sys
sys.path.append("../")
#from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent as Network
#from models.network_hierarchical_recurrent_autoencoder import NetworkHierarchicalRecurrentAutoencoder as Network
from models.network_hierarchical_recurrent_denoise import NetworkHierarchicalRecurrentDenoise as Network
from VirtualNetworkPhysiology import VirtualPhysiology

# Script wide variables
DEVICE     = 'cpu'

MODEL_PATH = ''
SAVE_PATH  = ''

# Load network checkpoint
model, hyperparameters, _ = NetworkHierarchicalRecurrentSTAutoencoder.load(
    model_path=MODEL_PATH, device=DEVICE
)

# Instantiate new VirtualPhysiology object
vphys = VirtualPhysiology(
    model=model,
    hyperparameters=hyperparameters,
    frame_shape=(36, 36),
    hidden_units=[2592],
    device=DEVICE
)


# Run virtual physiology methods
vphys.get_response_weighted_average(n_rand_stimuli=25000) \
     .get_grating_responses() \
     .get_grating_responses_parameters() \
     .save(SAVE_PATH)
