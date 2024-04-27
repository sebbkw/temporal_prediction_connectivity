import torch.nn as nn

def hierarchical_offset_mse (self, temporal_offset, out, data):
    predictions, hidden = out
    _, frame_targets    = data

    ##############################
    # Predictions for each group #
    ##############################

    predictions_by_group = []
    for group_idx, group in enumerate(self.output_units_groups):
        low_offset = sum(self.output_units_groups[:group_idx])
        high_offset = low_offset + group

        if temporal_offset > 0:
            predictions_by_group.append(predictions[:, self.warmup:-temporal_offset, low_offset:high_offset])
        else:
            predictions_by_group.append(predictions[:, self.warmup:, low_offset:high_offset])

    ##########################
    # Targets for each group #
    ##########################

    targets_by_group = [frame_targets[:, self.warmup+temporal_offset:, :]]
    for group_idx, group in enumerate(self.hidden_units_groups):
        # Group 1 target is true future frame
        if group_idx == 0:
            continue

        # Group > 1 is the lower order group's hidden activity
        low_offset = sum(self.hidden_units_groups[:group_idx-1])
        high_offset = low_offset + self.hidden_units_groups[group_idx-1]

        targets_by_group.append(hidden[:, self.warmup+temporal_offset:, low_offset:high_offset])

    #######################
    # MSEs for each group #
    #######################

    MSEs = []
    for prediction, target in zip(predictions_by_group, targets_by_group):
        MSEs.append(nn.functional.mse_loss(prediction, target))

    #######################
    # Final weighted loss #
    #######################

    ret = { 'L1': self.L1() }

    loss = 0
    loss += ret['L1']

    for group_idx, (beta, MSE) in enumerate(zip(self.beta_weights, MSEs)):
        ret[f'mse{group_idx}'] = MSE
        loss += beta*MSE

    return loss, ret
