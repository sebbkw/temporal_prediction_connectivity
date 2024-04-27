import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Takes orientation tuning curve at max tf and sf
# Returns direction selectivity (DSI)
def get_DSI (orientations, tuning_curve):    
    orient_pref_idx = np.where(tuning_curve == np.max(tuning_curve))[0][0]
    orient_pref = orientations[orient_pref_idx]
    orient_pref_resp = tuning_curve[orient_pref_idx]

    orient_opp = (orient_pref + 180) % 360
    orient_opp_idx = np.where(orientations == orient_opp)[0][0]
    orient_opp_resp = tuning_curve[orient_opp_idx]

    DSI = (orient_pref_resp - orient_opp_resp) / (orient_pref_resp + orient_opp_resp)
    #DSI = 1 - (orient_opp_resp/orient_pref_resp)

    return DSI

# Takes orientation tuning curve at max tf and sf
# Returns orientation selectivity (OSI)
def get_OSI (orientations, tuning_curve):    
    orient_pref_idx = np.where(tuning_curve == np.max(tuning_curve))[0][0]
    orient_pref = orientations[orient_pref_idx]
    orient_pref_resp = tuning_curve[orient_pref_idx]

    orient_orth1 = (orient_pref + 90) % 360
    orient_orth1_idx = np.where(orientations == orient_orth1)[0][0]
    orient_orth2 = (orient_pref - 90) % 360
    orient_orth2_idx = np.where(orientations == orient_orth2)[0][0]
    orient_orth_resp = (tuning_curve[orient_orth1_idx]+tuning_curve[orient_orth2_idx]) / 2

    OSI = (orient_pref_resp - orient_orth_resp) / (orient_pref_resp + orient_orth_resp)

    return OSI

def get_responses (unit_data):
    responses = np.zeros((len(orientation_arr), len(temporal_frequency_arr)))

    for ori_idx, ori in enumerate(orientation_arr):
        for tf_idx, tf in enumerate(temporal_frequency_arr):
            spikes = unit_data[
                (unit_data['orientation']==ori) & (unit_data['temporal_frequency']==tf)
            ].spike_count.values[0]
            
            responses[ori_idx, tf_idx] = spikes
            
    max_response = np.max(responses)
    
    ori_idx, tf_idx = [idx[0] for idx in np.where(responses == max_response)]
    tuning_curve = responses[:, tf_idx]
    
    return responses, tuning_curve


cache_dir = '/media/seb/ee7d6f3e-3390-444a-b0b3-131b80f2a7f8/ecephys_cache_dir/'
brain_area = 'VISp'  

temporal_frequency_arr = np.array([1, 2, 4, 8, 15])
orientation_arr        = np.array([0, 45, 90, 135, 180, 225, 270, 315])

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

print('Total sessions =', len(filtered_sessions_idxs))


unit_id_arr   = []
responses_arr = []
pref_ori_arr  = []

OSI_arr       = []
DSI_arr       = []

for session_id in filtered_sessions_idxs:
    # Load up a particular recording session
    print('Loading session data for session', session_id)
    session = cache.get_session_data(session_id)
    print('Got session data')

    # Get unit ids in brain area
    units = session.units[(session.units['snr'] > 4) & (session.units['ecephys_structure_acronym'] == brain_area)]
    unit_ids = units.index.values

    drifting_gratings_presentation_ids = session.stimulus_presentations.loc[
        (session.stimulus_presentations['stimulus_name'] == 'drifting_gratings')
    ].index.values

    stats = session.conditionwise_spike_statistics(
        stimulus_presentation_ids=drifting_gratings_presentation_ids,
        unit_ids=unit_ids
    )

    stats = pd.merge(stats, session.stimulus_conditions, left_on="stimulus_condition_id", right_index=True)

    for unit_id_idx, (unit_id, _) in enumerate(stats.index):
        if unit_id_idx%100 == 0:
            print(unit_id_idx, '/',  len(stats.index))

        unit_data = stats.loc[unit_id]
        responses, tuning_curve = get_responses (unit_data)

        responses_arr.append(responses)
        pref_ori_arr.append(orientation_arr[np.argmax(tuning_curve)])
        unit_id_arr.append(unit_id)        

        OSI_arr.append(get_OSI(orientation_arr, tuning_curve))
        DSI_arr.append(get_DSI(orientation_arr, tuning_curve)  )
    
    np.save('./v1_data/drifting_grating_tuning.npy', {
        'unit_id' : unit_id_arr,
        'response': responses_arr,
        'pref_ori': pref_ori_arr,
        'OSI'     : OSI_arr,
        'DSI'     : DSI_arr
    })
