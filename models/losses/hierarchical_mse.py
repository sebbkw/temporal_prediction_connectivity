import losses.hierarchical_offset_mse as l

def hierarchical_mse (self, out, data):
    return l.hierarchical_offset_mse(self, 0, out, data)
