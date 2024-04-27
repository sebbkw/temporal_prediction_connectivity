################################
# Imports and helper functions #
################################

import argparse, sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import data_loader
from models.network_feedforward import NetworkFeedforward
from models.network_hierarchical_recurrent_temporal_prediction import NetworkHierarchicalRecurrentTemporalPredicton
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent
from models.network_hierarchical_recurrent_rot import NetworkHierarchicalRecurrentRot
from models.network_hierarchical_recurrent_masked import NetworkHierarchicalRecurrentMasked
from models.network_hierarchical_recurrent_denoise import NetworkHierarchicalRecurrentDenoise
from models.network_hierarchical_recurrent_autoencoder import NetworkHierarchicalRecurrentAutoencoder
from models.network_hierarchical_recurrent_st_autoencoder import NetworkHierarchicalRecurrentSTAutoencoder

models = {
    'network_feedforward'                  : NetworkFeedforward,
    'network_hierarchical_recurrent_temporal_prediction': NetworkHierarchicalRecurrentTemporalPredicton,
    'network_hierarchical_recurrent'       : NetworkHierarchicalRecurrent,
    'network_hierarchical_recurrent_rot'   : NetworkHierarchicalRecurrentRot,
    'network_hierarchical_recurrent_masked': NetworkHierarchicalRecurrentMasked,
    'network_hierarchical_recurrent_denoise': NetworkHierarchicalRecurrentDenoise,
    'network_hierarchical_recurrent_autoencoder': NetworkHierarchicalRecurrentAutoencoder,
    'network_hierarchical_recurrent_st_autoencoder': NetworkHierarchicalRecurrentSTAutoencoder
}

# Add paths to datasets here
datasets = {
    'singer'       : '',
    'singer_rot'   : ['', ''],
    'singer_masked': ['', '']
}

# Function to store running (i.e., per minibatch) loss, to be average per epoch
def append_running_loss (running_loss_history, loss, loss_components):
    running_loss_history["i"] += 1
    running_loss_history["loss"] += loss.detach().cpu().numpy()

    for k, v in loss_components.items():
        try:
            v = v.detach().cpu().numpy()
        except:
            pass

        if k in running_loss_history:
            running_loss_history[k] += v
        else:
            running_loss_history[k] = v

# Function to append averaged loss values
def append_epoch_loss (epoch_loss, running_loss_history, epoch):
    i = running_loss_history["i"]

    epoch_loss['epochs'].append(epoch)

    for key in running_loss_history.keys():
        if key != 'i':
            if not key in epoch_loss:
                epoch_loss[key] = []
            epoch_loss[key].append(running_loss_history[key] / i)

def main (hyperparameters={}, path=None):
    ###########################
    # Load / define the model #
    ###########################

    if path:
        # Load previous model and loss history
        Network = models['network_hierarchical_recurrent_st_autoencoder']
        model, hyperparameters, loss_history = Network.load(path, hyperparameters['device'])

        train_history = loss_history['train']
        valid_history = loss_history['validation']

        print("Loaded model and loss history from", path, '\n')
        print("Using", hyperparameters['device'])

    else:
        # Or start from scratch if path argument not passed
        Network         = models[hyperparameters['model']]
        model = Network(hyperparameters)
        model = model.to(hyperparameters['device'])
        print("Using", hyperparameters['device'])
        print("Creating new model\n")

        train_history = {'epochs': []}
        valid_history = {'epochs': []}

    train_data_loader = data_loader(
        datasets[hyperparameters["dataset"]],
        split='train',
        batch_size=hyperparameters['batch_size']
    )
    valid_data_loader = data_loader(
        datasets[hyperparameters["dataset"]],
        split='validation',
        batch_size=hyperparameters['batch_size']
    )
    print('Loaded data loaders\n')

    ###################
    # Train the model #
    ###################

    # Set up optimizer and scheduler (lr decay)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])

    hyperparameters['epochs'] = 2500
    hyperparameters['checkpoints'] = 50

    # Loop through each training epoch
    for epoch in range(len(train_history['epochs'])+1, hyperparameters["epochs"]+1):
        for mode in ['validation', 'train']:
            # Skip validation dataset for given flag
            # or if not on every 10th epoch
            if mode == 'validation':
                if epoch%10!=0:
                    continue
                curr_data_loader = valid_data_loader
            else:
                curr_data_loader = train_data_loader

            # Object holding loss variables for each epoch
            running_loss_history = {
                "i": 0, "loss": 0
            }

            # Load each minibatch from respective data loader
            for batch_n, data in enumerate(curr_data_loader):
                data = model.preprocess_data(data)

                if mode == 'train':
                    model.train()
                    optimizer.zero_grad()
                    out = model(data[0])
                    loss, loss_components = model.get_loss(out, data)
                else:
                    model.eval()
                    with torch.no_grad():
                        out = model(data[0])
                        loss, loss_components = model.get_loss(out, data)

                if mode == 'train':
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.01)
                    if 'mask_gradients' in dir(model):
                        model.mask_gradients()
                    optimizer.step()

                # Append loss for current minibatch to running totals
                if type(loss) == list:
                    loss = sum(loss)
                append_running_loss(running_loss_history, loss, loss_components)

            # Average across minibatches to get epoch loss
            if mode == 'train':
                append_epoch_loss(train_history, running_loss_history, epoch)
            else:
                append_epoch_loss(valid_history, running_loss_history, epoch)

        print('Epoch: {}/{}.............'.format(epoch, hyperparameters['epochs']), end=' ')
        print("Loss: {:.4f}.............".format(train_history['loss'][-1]), end=' ')
        if 'accuracy' in valid_history and epoch%10==0:
            print("Val accuracy: {:.4f}.............".format(valid_history['accuracy'][-1]), end=' ')
        elif 'accuracy' in train_history:
            print("Train accuracy: {:.4f}.............".format(train_history['accuracy'][-1]), end=' ')

        print(datetime.now().strftime("%H:%M:%S"))

        # Save check points every n epochs, after all epochs complete,
        # or if early stopping called
        if (epoch%hyperparameters['checkpoints']==0 or epoch==hyperparameters['epochs']):
            model.save(
                dir_name=hyperparameters['save_dir'],
                epoch=len(train_history['loss']),
                hyperparameters=hyperparameters,
                loss_history={ "train": train_history, "validation": valid_history }
            )

