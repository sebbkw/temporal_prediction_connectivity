import sys, time, os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import losses.hierarchical_temporal_prediction as hierarchical_temporal_prediction
import losses.hierarchical_mse as hierarchical_mse
import losses.hierarchical_autoencoder as hierarchical_autoencoder
import losses.cross_entropy as cross_entropy

class NetworkBase (nn.Module):
    def __init__ (self):
        super(NetworkBase, self).__init__()

    def save (self, dir_name, epoch, hyperparameters = {}, loss_history = None):
        # Check if folder already exists, i.e. this is not first save
        if "path" in hyperparameters and hyperparameters["path"]:
            # Get containing folder of previously loaded model
            dir_name = os.path.dirname(hyperparameters["path"])
        else:
            # Otherwise get new folder name for model
            dir_name = './{}/{}_{}L1{}{}'.format(
                dir_name,
                hyperparameters["name"],
                int(np.log10(hyperparameters['lam'])),
                "_" if len(hyperparameters["name"]) else "",
                time.strftime('%b%e-%H-%M')
            )

            # Now actually make the folder and add to hyperparameters dict
            os.makedirs(dir_name)
            hyperparameters["path"] = dir_name + '/'

            # Save a csv file with the network hyperparameters
            pd.DataFrame(hyperparameters, index=[0]).to_csv(dir_name + '/hyperparameters.csv')

        model_file_name = '{}/{}-epochs_model.pt'.format(dir_name, epoch)
        loss_file_name = '{}/{}-epochs_losshistory.pickle'.format(dir_name, epoch)

        torch.save(self.state_dict(), model_file_name)

        # Now set "path" in hyperparameters dict so next time model checkpoint
        # will be saved in same folder
        if not "path" in hyperparameters or not hyperparameters["path"]: 
            hyperparameters["path"] = model_file_name

        if loss_history:
            with open(loss_file_name, 'wb') as p:
                pickle.dump(loss_history, p, protocol=4)

        print('Saved model as ' + model_file_name)

    def L1 (self):
        weights = torch.empty(0, device=self.device)
        for name, params in self.named_parameters():
            if 'weight' in name:
                weights = torch.cat((weights, params.flatten()), 0)
        L1 = self.lam*weights.abs().sum()
        return L1

    def get_loss (self, out, data):
        return getattr(self, self.loss)(out, data)

    # Must return two variables: prediction, hidden_state
    # prediction must be a tensor of size (n_batches, n_tsteps, n_channels)
    # hidden_state is a list with each tensor element of size (n_batches, n_tsteps n_channels)
    def forward (self):
        raise NotImplementedError('Model must implement forward pass.')

    @classmethod
    def load (cls, model_path, device=torch.device('cpu'), plot_loss_history=False, plot_loglog=True):
        # Get directory containing this model (used to load hyperparams)
        model_dir = os.path.dirname(model_path)

        # Load hyperparameters from csv file in that dir
        hyperparameters = pd.read_csv(model_dir + '/hyperparameters.csv')
        hyperparameters = hyperparameters.to_dict('index')[0]
        hyperparameters['device'] = device

        # Instantiate model
        model = cls(hyperparameters)

        # Load previous state
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)

        # Get training history file name
        train_hist_path = model_path[:-9] + '_losshistory.pickle'

        # Now open saved loss history for plotting
        with open(train_hist_path, 'rb') as p:
            history = pickle.load(p)

            if plot_loss_history:
                cls.plot_loss_history(history, loglog=plot_loglog)

        return model, hyperparameters, history

    @staticmethod
    def plot_loss_history (history, loglog):
        for key in history['train'].keys():
            if key == 'epochs':
                continue

            train_y      = history['train'][key]
            train_epochs = history['train']['epochs']
            if loglog:
                plt.loglog(train_epochs, train_y, label='Train')
            else:
                plt.plot(train_epochs, train_y, label='Train')
                plt.yscale('log')

            if False: #'validation' in history and 'loss' in history['validation']:
                validation_epochs = history['validation']['epochs']
                if loglog:
                    plt.loglog(validation_epochs, history['validation'][key], label='Validation')
                else:
                    plt.plot(validation_epochs, history['validation'][key], label='Validation')
                    plt.yscale('log')

            plt.xlabel('Epoch');
            plt.ylabel('Mean loss');
            plt.title(key);
            plt.legend()
            plt.show()

NetworkBase.loss_hierarchical_temporal_prediction = hierarchical_temporal_prediction.hierarchical_temporal_prediction
NetworkBase.loss_hierarchical_mse                 = hierarchical_mse.hierarchical_mse
NetworkBase.loss_hierarchical_autoencoder         = hierarchical_autoencoder.hierarchical_autoencoder
NetworkBase.loss_cross_entropy                    = cross_entropy.cross_entropy