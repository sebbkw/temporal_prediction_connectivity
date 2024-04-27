import torch
import torch.nn as nn

def cross_entropy (self, out, data):
    predictions, _ = out
    _, targets     = data

    predictions_t = torch.transpose(predictions, 1, 2)
    cross_entropy = nn.functional.cross_entropy(predictions_t, targets)

    classes = torch.argmax(predictions_t, dim=1)
    accuracy = torch.sum(classes == targets).item()/torch.numel(targets)

    L1 = self.L1()

    return cross_entropy+L1, { 'accuracy': accuracy, 'L1': L1 }
