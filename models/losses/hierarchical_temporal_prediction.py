import losses.hierarchical_offset_mse as l

def hierarchical_temporal_prediction (self, out, data):
    return l.hierarchical_offset_mse(self, 1, out, data)
