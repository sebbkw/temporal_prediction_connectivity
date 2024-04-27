import torch

def pad_matrix (m, pad_size):
    # repeat along last (feature) dimension
    m = torch.repeat_interleave(m.unsqueeze(0), pad_size, dim=0)
            
    # offseting by one for each repeat along the last dimensions
    m_stacked = []
    for i in range(pad_size):
        start_idx = i
        end_idx   = -(pad_size-i-1)
        
        if i != pad_size-1:
            m_stacked.append(m[start_idx, :, start_idx:end_idx])
        else:
            m_stacked.append(m[start_idx, :, start_idx:])
    
    # stack along last (feature) dimension
    m_stacked = torch.stack(m_stacked, dim=2)
    
    # now flatten last two dimensions
    m_stacked = torch.flatten(m_stacked, start_dim=2)
    
    return m_stacked
