import numpy as np
import scipy.stats as stats

OSI_THRESH = 0.3
DSI_THRESH = 0.4

def get_is_in_range (pre_unit, post_unit, mode):
    if mode not in ['short_range', 'long_range', 'all']:
        raise NotImplementedError
    
    pre_x, pre_y = pre_unit['gabor_x'], pre_unit['gabor_y']
    post_x, post_y = post_unit['gabor_x'], post_unit['gabor_y']
    
    dist = ((pre_x-post_x)**2 + (pre_y-post_y)**2)**0.5
    
    if (mode == 'short_range') and (dist < 2.5):
        return True
    elif (mode == 'long_range') and (dist > 5) and (dist < 9.166):
        return True
    elif (mode == 'all') and (dist < 9.166):
        return True
    else:
        return False
    
def get_is_connected (pre_unit, post_unit, weight_matrix, threshold):
    weight = weight_matrix[post_unit['hidden_unit_index'], pre_unit['hidden_unit_index']]
    return weight > threshold, weight

def is_orientation_or_direction_selective(pre_unit, post_unit, mode):
    if mode == 'orientation':
        return (pre_unit['OSI']>OSI_THRESH)
    elif mode == 'direction':
        return (pre_unit['DSI']>DSI_THRESH)
    else:
        raise NotImplementedError
    
def get_translation_matrix (unit):
    x, y = unit['gabor_x'], unit['gabor_y']
    
    return np.array([
        [1, 0, -x],
        [0, 1, -y],
        [0, 0,  1]
    ])

def get_rotation_matrix (unit):
    x, y = unit['gabor_x'], unit['gabor_y']
    pref_ori = unit['preferred_orientation']
    theta = np.deg2rad(pref_ori-90)
    
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1]
    ])

def get_rotation_matrix_marques (unit):
    x, y = unit['gabor_x'], unit['gabor_y']
    pref_ori = unit['preferred_orientation']
    theta = np.deg2rad(pref_ori)
    
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1]
    ])

def get_column_vector (unit):
    x, y = unit['gabor_x'], unit['gabor_y']

    return np.array([
        [x], [y], [1]
    ])

def apply_matrix_transformation (units, T):
    units_transformed = []
    
    for unit in units:
        vector             = get_column_vector(unit)
        vector_transformed = T@vector
        units_transformed.append({
            'hidden_unit_index': unit['hidden_unit_index'],
            'preferred_orientation': unit['preferred_orientation'],
            'gabor_x': vector_transformed[0],
            'gabor_y': vector_transformed[1]           
        })
        
    return units_transformed

def get_axis (unit):
    x, y = unit['gabor_x'], unit['gabor_y']
    
    if (y>x) and (y>-x):
        quadrant = 2
    elif (y>-x) and (y<x):
        quadrant = 1
    elif (y<-x) and (y<x):
        quadrant = 4
    elif (y>x) and (y<-x):
        quadrant = 3
    else:
        quadrant = -1
        
    if (quadrant==2) or (quadrant==4):
        return 'coaxial'
    elif (quadrant==1) or (quadrant==3):
        return 'orthogonal'
    else:
        return np.nan
    
def apply_axis_filter (post_units, mode):
    if mode == 'all':
        return post_units
    elif (mode=='coaxial') or (mode=='orthogonal'):
        return [unit for unit in post_units if get_axis(unit)==mode]
    else:
        raise NotImplementedError

def get_axis_dir (unit):
    x, y = unit['gabor_x'], unit['gabor_y']
    
    if (y>x) and (y>-x):
        quadrant = 2
    elif (y>-x) and (y<x):
        quadrant = 1
    elif (y<-x) and (y<x):
        quadrant = 4
    elif (y>x) and (y<-x):
        quadrant = 3
    else:
        quadrant = -1
        
    if quadrant == -1:
        return np.nan
    else:
        return quadrant

def apply_axis_filter_dir (post_units, mode):
    if mode in [1, 2, 3, 4]:
        return [unit for unit in post_units if get_axis_dir(unit)==mode]
    else:
        raise NotImplementedError
        

def get_axis_marques (unit):
    x, y = unit['gabor_x'], unit['gabor_y']
    
    # https://www.desmos.com/calculator/3w4ctok5us
    
    tan_22 = np.tan(np.deg2rad(22.5))
    tan_67 = np.tan(np.deg2rad(67.5))
        
    if (y>-tan_22*x) and (y<tan_22*x):
        return 1
    elif (y>tan_22*x) and (y<tan_67*x):
        return 2
    elif (y>-tan_67*x) and (y>tan_67*x):
        return 3
    elif (y<-tan_67*x) and (y>-tan_22*x):
        return 4
    elif (y<-tan_22*x) and (y>tan_22*x):
        return 5
    elif (y<tan_22*x) and (y>tan_67*x):
        return 6
    elif (y<tan_67*x) and (y<-tan_67*x):
        return 7
    elif (y>-tan_67*x) and (y<-tan_22*x):
        return 8
    else:
        return np.nan
        
def apply_axis_filter_marques (post_units, mode):
    if mode in [1, 2, 3, 4, 5, 6, 7, 8]:
        return [unit for unit in post_units if get_axis_marques(unit)==mode]
    else:
        raise NotImplementedError
        
def get_orientation_differences (pre_unit, post_units, mode):
    diffs = []
    
    for post_unit in post_units:
        a1, a2 = sorted([pre_unit['preferred_orientation'], post_unit['preferred_orientation']])

        if mode == 'orientation':
            freq_diff = abs((a1%180)-(a2%180))
            freq_diff = min(freq_diff, 180-freq_diff)
        elif mode == 'direction':
            freq_diff = abs(a1-a2)
            freq_diff = min(freq_diff, 360-freq_diff)
        else:
            raise NotImplementedError
        
        diffs.append(freq_diff)
        
    return diffs

def get_binned_orientations (diffs, mode):
    if mode == 'orientation':
        bin_edges  = [0, 22.5, 67.5, 90]
        bin_labels = ["0", "45", "90"]
    elif mode == 'direction':
        bin_edges  = [0, 36, 72, 108, 144, 180]
        bin_labels = ["0", "45", "90", "135", "180"]
    else:
        raise NotImplementedError
      
    hist, _ = np.histogram(diffs, weights=np.ones_like(diffs)/len(diffs), bins=bin_edges)
    hist = hist/np.sum(hist)
    
    return hist, bin_labels

def bin_data (val_arr, max_val, weights, bins, min_val=0):
    bin_edges = np.linspace(min_val, max_val, bins+1)
    weight_arr = np.ones_like(val_arr)/len(val_arr)
    hist, _ = np.histogram(val_arr, bins=bin_edges, weights=weights)
    return hist/hist.sum(), bin_edges[:-1]+np.diff(bin_edges)[0]/2

def get_shuffled_binned_data_ho (orientation_differences, weights, bins, threshold):
    iters = 1000
    shuffled_results = np.zeros((len(bins)-1, iters))
    
    for i in range(iters):
        if i % 1000 == 0:
            print('Iteration', i)
        
        shuffled_weights = np.array(weights).copy()
        np.random.shuffle(shuffled_weights)
        temp_binned_data, _ = np.histogram(orientation_differences[shuffled_weights>threshold], bins=bins)
        shuffled_results[:, i] = temp_binned_data
        
    return shuffled_results

def get_shuffled_binned_data_iacoruso (post_unit_orientations, pre_unit_orientations, bins, total_units):
    iters = 1000
    shuffled_results = np.zeros((len(bins)-1, iters))
    
    pre_unit_orientations_copy = pre_unit_orientations.copy()
    
    for i in range(iters):
        if i % 100 == 0:
            print('Iteration', i)
            
        np.random.shuffle(pre_unit_orientations_copy)
                
        orientation_differences = []
        for a1, a2 in zip(post_unit_orientations, pre_unit_orientations_copy):
            freq_diff = abs((a1%180)-(a2%180))
            freq_diff = min(freq_diff, 180-freq_diff)
            orientation_differences.append(freq_diff)

        binned_data, _ = np.histogram(orientation_differences, bins=bins)

        shuffled_results[:, i] = binned_data/total_units
        
    return shuffled_results

def get_p_val (test, null):
    len_gt  = (test > null).sum()
    len_lte = (test <= null).sum()
    
    return 2*(min(len_gt, len_lte)/len(null))

def get_heat_maps (orientation_differences, coords, weights, bins, threshold, full=False):
    heat_maps = []
    binned_coords = [[] for b in range(len(bins)-1)]
    
    for d, c in zip(orientation_differences[weights>threshold], coords[weights>threshold]):
        binned_coord_indices = np.histogram([d], bins=bins)[0]
        if 1 in binned_coord_indices:
            binned_coords[np.argmax(binned_coord_indices)].append(c)
                
    for binned_coord_idx, binned_coord in enumerate(binned_coords): 
        if full:
            heat_map = np.zeros((102, 102))
        else:
            heat_map = np.zeros((20, 20))
        for coord in binned_coord:
            if full:
                x, y = int(coord['x'][0]+51), int(coord['y'][0]+51)
            else:
                x, y = int(coord['x'][0]+10), int(coord['y'][0]+10)
            heat_map[y, x] += 1
        #heat_map /= np.sum(heat_map.reshape(-1))
        heat_maps.append(heat_map)
        
    return np.array(heat_maps)
    
def get_shuffled_heat_map_data (orientation_differences, coords, weights, bins, threshold, full=False):
    iters = 100
    if full:
        shuffled_results = np.zeros((iters, len(bins)-1, 102, 102))
    else:
        shuffled_results = np.zeros((iters, len(bins)-1, 20, 20))
    
    for i in range(iters):
        if i % 10 == 0:
            print('Iteration', i)
        
        shuffled_weights = np.array(weights).copy()
        np.random.shuffle(shuffled_weights)
        
        heat_maps = get_heat_maps(orientation_differences, coords, shuffled_weights, bins, threshold, full=full)
        shuffled_results[i] = heat_maps
                   
    return np.mean(shuffled_results, axis=0)

def save_connectivity_data (model_name, measure_name, model, target):
    # Load existing dataset
    try:
        data = np.load('connectivity_data.npy', allow_pickle=True).item()
    except:
        data = {}
            
    # Save
    if not model_name in data:
        data[model_name] = {}
    
    if measure_name in data[model_name]:
        print("This will over-write existing data!")

    try:
        data[model_name][measure_name] = stats.pearsonr(model, target)[0]
    except:
        data[model_name][measure_name] = np.nan

    np.save('connectivity_data.npy', data)
    
def get_connectivity_data ():
    average_scores = {}
    
    data = np.load('connectivity_data.npy', allow_pickle=True).item()
                
    for model_name, model_data in data.items():
        cc_arr  = []
        
        for cc in model_data.values():
            if np.isfinite(cc):
                cc_arr.append(cc)
            else:
                cc_arr.append(0)
                        
        average_scores[model_name] = (np.mean(cc_arr), np.std(cc_arr)/(len(cc_arr)**0.5), cc_arr)
        
    return average_scores