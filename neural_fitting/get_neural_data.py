#### Import modules ###

import os
import numpy as np
import pandas as pd
import cv2
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#### Helper functions ####
def sahani_quick (data):
    # N = number of repeats
    # T = number of time bins
    
    N = data.shape[0]
    T = data.shape[1]
    
    # Total power = average( power in each trial )
    TP = np.nanmean(np.nanvar(data, axis=1, ddof=1))
    
    # Signal power
    SP = 1/(N-1) * (N * np.nanvar(np.nanmean(data, axis=0), ddof=1) - TP)
    
    if SP < 0:
        SP = np.nan
        
    # Noise power
    NP = TP-SP

    return SP, NP, TP

def get_spikes (scene_presentations, units, n_repeats):
    # Get spikes aligned to each frame onset
    spikes = session.presentationwise_spike_times(
        stimulus_presentation_ids=scene_presentations.index.values,
        unit_ids=units.index.values
    )
    
    # Add count column to spikes dataframe
    spikes["count"] = np.zeros(spikes.shape[0])
    # For each stimulus presentation and neuron, count all the spikes
    spikes_count = spikes.groupby(["stimulus_presentation_id", "unit_id"]).count()

    # Now reshape into a matrix with rows as presentation ids and columns as unit ids
    stimulus_x_unit_id = pd.pivot_table(
        spikes_count,
        values="count",
        index="stimulus_presentation_id",
        columns="unit_id",
        fill_value=0.0,
        aggfunc=np.sum
    )
    
    # Append missing rows for which no neuron fired 
    missing_rows = list(set(scene_presentations.index.values).difference(stimulus_x_unit_id.index.values))
    for row in missing_rows:
        empty_row = stimulus_x_unit_id.iloc[0].copy()
        empty_row[:] = 0
        stimulus_x_unit_id.loc[row] = empty_row

    # And sort by the index (presentation id)
    stimulus_x_unit_id = stimulus_x_unit_id.sort_index()

    # Now add frame column
    stimulus_x_unit_id['frame'] = scene_presentations['frame'].values

    # Sort rows by frame number
    stimulus_x_unit_id = stimulus_x_unit_id.sort_values('frame')
    
    # Get unit ids from pivot table (excluding frame column)
    unit_ids = stimulus_x_unit_id.columns.values[:-1]

    # Convert to numpy array, discarding dummy frame number column
    stimulus_x_unit_id_numpy = stimulus_x_unit_id.to_numpy()[:, :-1]
        
    # Split into two blocks so that summing procedure only takes
    # frames that were actually contiguous in time
    n_rows = len(stimulus_x_unit_id_numpy)
    block_1 = stimulus_x_unit_id_numpy[:n_rows//2]
    block_2 = stimulus_x_unit_id_numpy[n_rows//2:]
    
    frame_counts = []
    for block in [block_1, block_2]:
        for curr_row_idx in range(len(block)):
            summed_row = np.sum(block[curr_row_idx:curr_row_idx+1], axis=0)
            frame_counts.append(summed_row)
            
    frame_counts = np.array(frame_counts)

    # Chunk response to each frame into blocks of repeats
    chunked_frame_counts = []
    for row_idx in range(0, len(frame_counts), n_repeats):
        chunked_frame_counts.append(
            frame_counts[row_idx:row_idx+n_repeats]
        )
    chunked_frame_counts = np.array(chunked_frame_counts)

    # And reshape into order (unit, repeat, frame)
    chunked_frame_counts = np.transpose(chunked_frame_counts, axes=[2, 1, 0])

    # Average across repeats
    mean_frame_counts = np.nanmean(chunked_frame_counts, axis=1)
    
    return mean_frame_counts, chunked_frame_counts, unit_ids


def process_im (im, bp):    
    # Crop to 180x180
    im = im[:, 90:90+180]
    
    if bp:
        im = whiten_and_filter_image(im)
    
    # Resize
    h = 36 
    w = 36
    #im = cv2.resize(im, (w, h))
        
    im = im.flatten()
    im = (im - np.mean(im)) / np.std(im)
        
    return im


################
# Main routine #
################

bandpass = False
chunk_size = 300
brain_area = 'VISp'
cache_dir = '' # Cache dir here

# Create save dir
#save_dir = f'./neural_data/{brain_area}'
save_dir = f'./neural_data/{brain_area}'
if not os.path.exists(save_dir):
     os.makedirs(save_dir)
     print('Created:', save_dir)

    
# Load cache file
print('Loading cache dir, will download if it does not already exist')
manifest_path = os.path.join(cache_dir, 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# Get sessions
sessions = cache.get_session_table()

filtered_sessions = sessions[(sessions.session_type == 'brain_observatory_1.1') & \
                             (sessions.full_genotype == 'wt/wt') & \
                             ([brain_area in acronyms for acronyms in 
                               sessions.ecephys_structure_acronyms])]

filtered_sessions_idxs = filtered_sessions.index.values

for session_id in filtered_sessions_idxs:
    if True:
        # Load up a particular recording session
        print('\n\nLoading session data for session', session_id)
        session = cache.get_session_data(session_id)

        # Get unit ids in brain area
        units = session.units[session.units['ecephys_structure_acronym'] == brain_area]

        # Get stimulus tables
        print('\n\nGet stimulus table for movie 1')
        stim_table_one = session.get_stimulus_table("natural_movie_one")
        print('Get stimulus table movie 3')
        stim_table_three = session.get_stimulus_table("natural_movie_three")

        print('\n\nGetting processed spike data')
        try:
            spikes_1_mean, spikes_1_trials, unit_ids_1 = get_spikes(stim_table_one, units, 20)
            print('Got spike data for movie 1')
            spikes_3_mean, spikes_3_trials, unit_ids_3 = get_spikes(stim_table_three, units, 10)

            spikes_1_trials = spikes_1_trials[:, :10]

            spikes_1_trials = spikes_1_trials.astype('float')
            spikes_3_trials = spikes_3_trials.astype('float')

            print('Got spike data for movie 3')
        except Exception as e:
            print('Error:', e)
            continue

        assert(np.array_equal(unit_ids_1, unit_ids_3))

        print('\n\nProcessing neural data')

        def chunk_data (stim_name, mean, trials):
            N = trials.shape[2]

            stim_names   = []
            trial_chunks = []
            mean_chunks  = []

            for i in range(0, N, chunk_size):
                chunk_slice = slice(i, i+chunk_size)

                stim_names.append(stim_name + '_' + str(i))
                mean_chunks.append(mean[:, chunk_slice])
                trial_chunks.append(trials[:, :, chunk_slice])

            return stim_names, mean_chunks, trial_chunks

        movie_1_stims, movie_1_mean_chunks, movie_1_trial_chunks = chunk_data('movie_one', spikes_1_mean, spikes_1_trials)
        movie_3_stims, movie_3_mean_chunks, movie_3_trial_chunks = chunk_data('movie_three', spikes_3_mean, spikes_3_trials)

        all_stims  = [*movie_1_stims,        *movie_3_stims]
        all_mean   = [*movie_1_mean_chunks,  *movie_3_mean_chunks]
        all_trials = [*movie_1_trial_chunks, *movie_3_trial_chunks]

        neural_data = {}
        for unit_idx, unit_id in enumerate(unit_ids_1):
            SP_1, NP_1, _ = sahani_quick (spikes_1_trials[unit_idx])
            SP_3, NP_3, _ = sahani_quick (spikes_3_trials[unit_idx])
            noise_ratio = np.mean([NP_1/SP_1, NP_3/SP_3])

            if noise_ratio > 60:
                continue

            neural_data[unit_id] = { 'stimuli': [], 'spikes_trials': {}, 'spikes_mean': {}, 'noise_ratio': noise_ratio }

            for stim, mean, trials in zip(all_stims, all_mean, all_trials):
                neural_data[unit_id]['stimuli'].append(stim)
                neural_data[unit_id]['spikes_mean'][stim]= mean[unit_idx]
                neural_data[unit_id]['spikes_trials'][stim] = trials[unit_idx]

    if True:
        # Load movie frames
        print('\n\nLoading movie frames, may take a few minutes')

        raw_movie_frames = {
            'movie_one': cache.get_natural_movie_template(1),
            'movie_three': cache.get_natural_movie_template(3)
        }

        stimuli_data = {}
        for movie_key, movie in raw_movie_frames.items():
            print('Processing frames for', movie_key)

            processed_data = np.array([process_im(im, bp=bandpass) for im in movie])
            for i in range(0, len(processed_data), chunk_size):
                stimuli_data[f'{movie_key}_{i}'] = processed_data[i:i+chunk_size]

    # Save data
    print('\n\nSaving data')
    np.save(f'{save_dir}/{session_id}_stimuli_20x40_raw.npy', stimuli_data)
    np.save(f'{save_dir}/{session_id}_wt_10repeats_neural.npy', neural_data)
