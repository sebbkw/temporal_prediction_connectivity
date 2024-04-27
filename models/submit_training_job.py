################################
# Imports and helper functions #
################################

import argparse, sys
import torch

import train

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to model checkpoint')
is_arg_required = not '--path' in sys.argv
parser.add_argument('--name', required=is_arg_required)
parser.add_argument('--L1', type=float, required=is_arg_required)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################
# Optionally load checkpoint #
##############################

if not args.path is None:
    train.main({'device': device}, args.path)
    exit()

##################################
# Define various hyperparameters #
##################################

hyperparameters = {
    "device"   : device,
    "save_dir" : 'model_checkpoints',
    "name"     : args.name,

    "batch_size"  : 100,
    "epochs"      : 2000,
    "lr"          : 10**-4,
    "checkpoints" : 500,

    "dataset" : 'singer',
    "loss"    : 'loss_hierarchical_st_autoencoder',
    "model"   : 'network_hierarchical_recurrent_st_autoencoder',

    "frame_size"            : 36*36,
    "hidden_units_groups"   : '2592,',
    "lam"                   : 10**args.L1,
    "warmup"                : 4,
    "beta_weights"          : '1,',
    "local_inhibitory_prop" : 0.1

}

if hyperparameters['loss'] == 'loss_hierarchical_autoencoder':
    hyperparameters['lam_activity'] = 10**-7

print('Using hyperparameters:')
for k, v in hyperparameters.items():
    print('\t', k, '\t', v)
print('')

train.main(hyperparameters)
