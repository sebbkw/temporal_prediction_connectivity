import losses.hierarchical_offset_mse as l

WEIGHT_ACTIVITY = True

def hierarchical_autoencoder (self, out, data):
    _, hidden = out

    loss, ret = l.hierarchical_offset_mse(self, 0, out, data)

    ret['L1_activity'] = 0
    for group_idx, (group, beta) in enumerate(zip(self.hidden_units_groups, self.beta_weights)):
        low_offset = sum(self.hidden_units_groups[:group_idx])
        high_offset = low_offset + group

        if not WEIGHT_ACTIVITY:
            beta = 1

        activity = hidden[:, self.warmup+temporal_offset:, low_offset:high_offset].group.abs().sum()*beta
        ret['L1_activity'] += activity
        
    loss += ret['L1_activity']
    
    return loss, ret