import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models.network_base import NetworkBase

class NetworkHierarchicalRecurrent (NetworkBase):
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalRecurrent, self).__init__()

        self.loss                  = hyperparameters['loss']
        self.lam                   = hyperparameters['lam']
        self.frame_size            = hyperparameters['frame_size']
        self.warmup                = hyperparameters['warmup']
        self.beta_weights          = [float(i) for i in hyperparameters['beta_weights'].split(',') if len(i)]
        self.local_inhibitory_prop = hyperparameters['local_inhibitory_prop']
        self.device                = hyperparameters['device']

        self.hidden_units_groups     = [int(i) for i in hyperparameters['hidden_units_groups'].split(',') if len(i)]
        self.hidden_units_groups_len = len(self.hidden_units_groups)
        self.hidden_units            = sum(self.hidden_units_groups)

        self.output_units_groups     = [self.frame_size] + self.hidden_units_groups[:-1]
        self.output_units_groups_len = len(self.output_units_groups)
        self.output_units            = sum(self.output_units_groups)

        self.rnn = nn.RNN(
            input_size = self.frame_size,
            hidden_size = self.hidden_units,
            num_layers = 1,
            nonlinearity = 'relu',
            batch_first = True
        )
        self.fc = nn.Linear(
            in_features = self.hidden_units,
            out_features = self.output_units
        )

        self.initialize_weights_biases()
        self.set_weight_masks()

    def initialize_weights_biases (self):
        # Initialise RNN weights with identity matrix
        nn.init.uniform_(self.rnn.bias_ih_l0, 0, 0)
        nn.init.uniform_(self.rnn.bias_hh_l0, 0, 0)
        nn.init.uniform_(self.fc.bias, 0, 0)

        ih_upper_bound = np.sqrt(2/(36*36*100))
        fc_upper_bound = np.sqrt(2/(36*36*36*36*2))
        nn.init.uniform_(self.rnn.weight_ih_l0, 0, ih_upper_bound)
        nn.init.uniform_(self.fc.weight, 0, fc_upper_bound)

        with torch.no_grad():
            self.state_dict()['rnn.weight_hh_l0'][:] = nn.Parameter(torch.eye(self.hidden_units, self.hidden_units)) / 100

    def set_weight_masks (self):
        def get_rnn_slice_from_group_n (group_n):
            start = sum(self.hidden_units_groups[:group_n])
            return slice(
                start,
                start + self.hidden_units_groups[group_n]
            )
        def get_fc_slice_from_group_n (group_n):
            start = sum(self.output_units_groups[:group_n])
            return slice(
                start,
                start + self.output_units_groups[group_n]

            )

        # Mask to zero out weights for input in recurrent layer
        self.rnn_ih_mask = torch.zeros(self.rnn.weight_ih_l0.shape).to(self.device)
        self.rnn_ih_mask[:self.hidden_units_groups[0], :] = 1

        # Mask to zero out weights for hidden connections in recurrent layer
        self.rnn_hh_mask = torch.zeros(self.rnn.weight_hh_l0.shape).to(self.device)
        for group_n in range(self.output_units_groups_len):
            # Recurrent connections
            curr_group_idxs = get_rnn_slice_from_group_n (group_n)
            self.rnn_hh_mask[curr_group_idxs, curr_group_idxs] = 1

            # Feedback connections (not applicable to group 0)
            if group_n > 0:
                prev_group_idxs = get_rnn_slice_from_group_n (group_n - 1)
                self.rnn_hh_mask[prev_group_idxs, curr_group_idxs] = 1

            # Feedforward connections (not applicable to last group)
            final_group_n = len(self.hidden_units_groups)-1
            if group_n < final_group_n:
                next_group_idxs = get_rnn_slice_from_group_n(group_n + 1)
                self.rnn_hh_mask[next_group_idxs, curr_group_idxs] = 1

            # Get idxs for inhibitory units for current group
            unit_ranges = self.get_inhibitory_idxs(group_n)
            excitatory_range, local_inhibitory_range = unit_ranges

            # Set only local connections for 'interneuron' units to 1
            self.rnn_hh_mask[curr_group_idxs, local_inhibitory_range] = 1
            if group_n > 0:
                self.rnn_hh_mask[prev_group_idxs, local_inhibitory_range] = 0
            if group_n < final_group_n:
                self.rnn_hh_mask[next_group_idxs, local_inhibitory_range] = 0

        # Mask to zero out RNN-FC weights
        self.fc_mask = torch.zeros(self.fc.weight.shape).to(self.device)
        for group_n in range(self.output_units_groups_len):
            presynaptic_idxs = get_rnn_slice_from_group_n(group_n)
            postsynaptic_idxs = get_fc_slice_from_group_n(group_n)

            self.fc_mask[postsynaptic_idxs, presynaptic_idxs] = 1

    def get_inhibitory_idxs (self, group_n):
        # Store number of units in each group
        # to get number of excitatory and inhibitory units
        n_group_units = self.hidden_units_groups[group_n]
        n_local_inhibitory = int(self.local_inhibitory_prop*n_group_units)
        n_excitatory = n_group_units - n_local_inhibitory

        # Excitatory neurons
        start_excitatory = sum(self.hidden_units_groups[:group_n])
        end_excitatory = start_excitatory + n_excitatory
        excitatory_range = slice(start_excitatory, end_excitatory)

        # Local inhibitory neurons
        end_local_inhibitory = end_excitatory + n_local_inhibitory
        local_inhibitory_range = slice(end_excitatory, end_local_inhibitory)

        return excitatory_range, local_inhibitory_range

    def preprocess_data (self, data):
        x, y = data

        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def forward (self, inputs):
        # Mask weights
        self.rnn.weight_ih_l0.data.mul_(self.rnn_ih_mask)
        self.rnn.weight_hh_l0.data.mul_(self.rnn_hh_mask)
        self.fc.weight.data.mul_(self.fc_mask)

        # Enforce Dale's law
        self.set_excitatory_inhibitory_weights()

        # Forward pass
        rnn_outputs, _ = self.rnn(inputs)
        fc_outputs = self.fc(rnn_outputs)

        return fc_outputs, rnn_outputs

    def mask_gradients (self):
        self.rnn.weight_ih_l0.grad.data.mul_(self.rnn_ih_mask)
        self.rnn.weight_hh_l0.grad.data.mul_(self.rnn_hh_mask)
        self.fc.weight.grad.data.mul_(self.fc_mask)

    def set_excitatory_inhibitory_weights (self):
        # Loop through each group
        for group_n in range(self.output_units_groups_len):
            unit_ranges = self.get_inhibitory_idxs(group_n)
            excitatory_range, local_inhibitory_range = unit_ranges

            # Set excitatory neuron weights as |W|
            excitatory_weights = self.rnn.weight_hh_l0.data[:, excitatory_range]
            self.rnn.weight_hh_l0.data[:, excitatory_range] = torch.abs(excitatory_weights)

            # Set local inhibitory neurons as -|W|
            local_inhibitory_weights = self.rnn.weight_hh_l0.data[:, local_inhibitory_range]
            self.rnn.weight_hh_l0.data[:, local_inhibitory_range] = -torch.abs(local_inhibitory_weights)
