import pickle, math, os

import numpy as np
import torch

import scipy
import scipy.stats
from scipy import ndimage
import scipy.optimize as opt
from scipy import signal
import scipy.fft as fft

class VirtualPhysiology:
    # model               trained model
    # hyperparameters     model hyperparameters
    # hidden_unit_range   range object of hidden units for analysis
    # device              device for tensors (cpu or cuda)
    def __init__ (self, model, hyperparameters, frame_shape, hidden_units, device):
        self.data = []

        self.model = model
        self.model.eval()

        self.hyperparameters = hyperparameters
        self.warmup = hyperparameters["warmup"]
        self.frame_size = hyperparameters["frame_size"]
        self.t_steps = 50

        self.frame_shape = frame_shape
        self.hidden_units = hidden_units
        self.device = device

        self.data = []
        for group in self.hidden_units:
            self.data.append([])

        self.osi_thresh = 0.4
        self.dsi_thresh = 0.3
        self.mean_response_offset = 5

        min_n, max_n = min(frame_shape), max(frame_shape)
        self.spatial_frequencies = [i/max_n for i in range(1, min_n//2+1)] # Cycles / pixel
        self.orientations = np.arange(0, 360, 5) # Degrees
        self.temporal_frequencies = np.linspace(1/self.t_steps, 1/4, self.t_steps//4) # Cycles / frame
        

    @classmethod
    def load (cls, data_path, model, hyperparameters, frame_shape, hidden_units, device):
        virtual_physiology = cls(
            model=model,
            hyperparameters=hyperparameters,
            frame_shape=frame_shape,
            hidden_units=hidden_units,
            device=device
        )

        with open(data_path, 'rb') as handler:
            virtual_physiology.data = pickle.load(handler)

        return virtual_physiology

    # relative_path = use file name as part of data_path relative to model dir
    def save (self, data_path):            
        with open(data_path, 'wb') as p:
            pickle.dump(self.data, p, protocol=4)

        with open(data_path + '.params', 'wb') as p:
            params = {
                "t_steps": self.t_steps,
                "spatial_frequencies": self.spatial_frequencies,
                "orientations": self.orientations,
                "temporal_frequencies": self.temporal_frequencies
            }
            pickle.dump(params, p, protocol=4)


    def get_unit_data (self, unit_idx):
        for group in self.data:
            for unit_data in group:
                if unit_data["hidden_unit_index"] == unit_idx:
                    return unit_data
        return False


    def get_group_from_unit_idx (self, unit_idx):
        for group_idx in range(len(self.hidden_units)):
            if unit_idx <  np.sum(self.hidden_units[:group_idx+1]):
                return group_idx
        return False

    def get_response_weighted_average (self, n_rand_stimuli=100):
        # Pre-allocate lists of lists to hold gaussian noise stimuli and associated unit activity
        stimuli = []
        unit_activity = []
        for _ in range (np.sum(self.hidden_units)):
            stimuli.append([])
            unit_activity.append([])

        noise_shape = (n_rand_stimuli, self.warmup+self.t_steps, self.frame_size)
        noise = np.random.normal(loc=0, scale=1, size=noise_shape)
        noise = torch.Tensor(noise).to(self.device)

        with torch.no_grad():
            _, hidden_state = self.model(noise)


        response = hidden_state.detach().numpy().reshape(n_rand_stimuli*(self.warmup+self.t_steps), -1)
        noise = noise.detach().numpy().reshape(-1, self.frame_size)

        rwa_arr = []
        for unit_idx, unit_responses in enumerate(response.T):
            group_idx = self.get_group_from_unit_idx(unit_idx)

            if unit_idx % 100 == 0:
                print('Processing RWA for unit', unit_idx)

            if np.sum(unit_responses):
                if group_idx == 0:
                    rwa = np.average(noise, axis=0, weights=unit_responses)
                else:
                    rwa = np.average(noise[:-group_idx], axis=0, weights=unit_responses[group_idx:])

                self.data[group_idx].append({
                    "hidden_unit_index": unit_idx,
                    "response_weighted_average": rwa
                })

        print('Finished averaging stimuli')

        return self

    # sf = cycles per pixel
    # tf = cycles per second
    # speed = tf/sf = pixels per second
    def get_grating_stimuli(self, spatial_frequency, orientation, temporal_frequency, grating_amplitude, frames):
        y_size, x_size = self.frame_shape

        theta = (orientation-90) * np.pi/180
        x, y = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size))
        x_theta = x * np.cos(theta) + y * np.sin(theta)

        phase_shift = 2*np.pi*temporal_frequency
        phases = np.arange(frames)*phase_shift

        grating_frames = []
        for phase in phases:
            grating_frames.append( grating_amplitude * np.sin(2*spatial_frequency*np.pi*x_theta - phase) )


        gratings = np.array(grating_frames).reshape(1, frames, y_size*x_size)
        gratings = (gratings-np.mean(gratings))/np.std(gratings) ##
        gratings = torch.Tensor(gratings).to(self.device)

        return gratings

    def get_grating_responses (self):
        # Add array to data dictionary structures containing
        # complete response (for each grating phase) and mean response (averaged across phases)
        # for each spatial frequency/orientation/tf combination
        for group_data in self.data:
            for unit_data in group_data:
                unit_data["grating_responses"] = np.zeros((
                    len(self.spatial_frequencies),
                    len(self.orientations),
                    len(self.temporal_frequencies),
                    self.t_steps
                ))
                unit_data["mean_grating_responses"] = np.zeros((
                    len(self.spatial_frequencies),
                    len(self.orientations),
                    len(self.temporal_frequencies)
                ))

        # Keep track of progress for display purposes
        param_count = 0
        try:
            total_params = self.data[0][0]["mean_grating_responses"].size
        except:
            total_params = self.data[1][0]["mean_grating_responses"].size            

        # Loop through each parameter combination for each unit
        for sf_idx, sf in enumerate(self.spatial_frequencies):
            for ori_idx, ori in enumerate(self.orientations):
                for tf_idx, tf in enumerate(self.temporal_frequencies):

                    # Generate gratings for particular param combination
                    gratings = self.get_grating_stimuli(sf, ori, tf, grating_amplitude=1, frames=self.warmup+self.t_steps)

                    # Feedforward pass through network
                    with torch.no_grad():
                        _, hidden_state = self.model(gratings)

                    # Loop through unit responses at each time step
                    for group_data in self.data:
                        for unit_data in group_data:
                            unit_idx = unit_data["hidden_unit_index"]

                            # Discard warm up period of network's response to gratings
                            unit_responses = hidden_state[0, self.warmup:, unit_idx].cpu().numpy()

                            unit_data["grating_responses"][sf_idx, ori_idx, tf_idx] = unit_responses
                            unit_data["mean_grating_responses"][sf_idx, ori_idx, tf_idx] = np.mean(unit_responses)

                    if param_count % 100 == 99:
                        print("Finished param combination {}/{}".format(param_count+1, total_params)) 
                    param_count += 1

        print("Finished tuning curve")

        return self

    # Takes orientation tuning curve at max tf and sf
    # Returns direction selectivity (DSI)
    def get_DSI (self, tuning_curve):    
        orient_pref_idx = np.where(tuning_curve == np.max(tuning_curve))[0][0]
        orient_pref = self.orientations[orient_pref_idx]
        orient_pref_resp = tuning_curve[orient_pref_idx]

        orient_opp = (orient_pref + 180) % 360
        orient_opp_idx = np.where(self.orientations == orient_opp)[0][0]
        orient_opp_resp = tuning_curve[orient_opp_idx]

        DSI = (orient_pref_resp - orient_opp_resp) / (orient_pref_resp + orient_opp_resp)
        #DSI = 1 - (orient_opp_resp/orient_pref_resp)

        return DSI


    # Takes orientation tuning curve at max tf and sf
    # Returns orientation selectivity (OSI)
    def get_OSI (self, tuning_curve):    
        orient_pref_idx = np.where(tuning_curve == np.max(tuning_curve))[0][0]
        orient_pref = self.orientations[orient_pref_idx]
        orient_pref_resp = tuning_curve[orient_pref_idx]

        orient_orth1 = (orient_pref + 90) % 360
        orient_orth1_idx = np.where(self.orientations == orient_orth1)[0][0]
        orient_orth2 = (orient_pref - 90) % 360
        orient_orth2_idx = np.where(self.orientations == orient_orth2)[0][0]
        orient_orth_resp = (tuning_curve[orient_orth1_idx]+tuning_curve[orient_orth2_idx]) / 2

        OSI = (orient_pref_resp - orient_orth_resp) / (orient_pref_resp + orient_orth_resp)

        return OSI

    def get_orientation_tuning_curve(self, unit_data):
        mean_grating_responses = unit_data["mean_grating_responses"]

        # Get indices of max grating response (mean across time steps)
        max_mean_grating_response = np.max(mean_grating_responses)
        unit_data["max_mean_grating_response"] = max_mean_grating_response
        sf_idx, ori_idx, tf_idx = [idx[0] for idx in np.where(mean_grating_responses == max_mean_grating_response)]

        orientation_tuning_curve = mean_grating_responses[sf_idx, :, tf_idx]
        return orientation_tuning_curve


    # Gets OSI, DSI, CV and modulation ratio for each unit
    def get_grating_responses_parameters (self):
        for group_idx, group_data in enumerate(self.data):
            for unit_i, unit_data in enumerate(group_data):
                grating_responses = unit_data["grating_responses"]
                mean_grating_responses = unit_data["mean_grating_responses"]

                # Get indices of max grating response (mean across time steps)
                max_mean_grating_response = np.max(mean_grating_responses)
                unit_data["max_mean_grating_response"] = max_mean_grating_response
                sf_idx, ori_idx, tf_idx = [idx[0] for idx in np.where(mean_grating_responses == max_mean_grating_response)]

                # Convert these indices into the underlying parameters
                max_sf = unit_data["preferred_sf"] = self.spatial_frequencies[sf_idx]
                max_ori = unit_data["preferred_orientation"] =  self.orientations[ori_idx]
                max_tf = unit_data["preferred_tf"] = self.temporal_frequencies[tf_idx]

                # Response to moving grating for parameters that give maximum response
                optimum_grating_response = grating_responses[sf_idx, ori_idx, tf_idx]
                unit_data["optimum_grating_response"] = optimum_grating_response

                # Get CV and DSI measures from curve
                # given max spatial frequency and temporal frequency parameters
                orientation_curve = mean_grating_responses[sf_idx, :, tf_idx]
                unit_data["OSI"] = self.get_OSI(orientation_curve)
                unit_data["DSI"] = self.get_DSI(orientation_curve)

                if unit_i % 50 == 49:
                    print("Finished unit {} / {}, group {} / {}".format(
                        unit_i+1, len(group_data), group_idx+1, len(self.data)
                    ))

        return self

    # Filter unresponsive units and units where curve fitting failed
    def filter_unit_data (self, group_idx):
        group_data = self.data[group_idx]

        # Reject low mean response units (< 1% of mean, max mean response)
        all_max_mean = []
        for group in self.data:
            for u in group:
                all_max_mean.append(u["max_mean_grating_response"])

        mean_responses = [u["max_mean_grating_response"] for u in group_data]

        response_threshold = 0.01 * np.mean(all_max_mean)
        n_filtered = len(np.where(mean_responses < response_threshold)[0])
        print("{} / {} units below response threshold".format(n_filtered, len(group_data)))

        # Reject units where curve fitting failed (modulation_ratio set as False)
        n_filtered = len([u for u in group_data if not u["modulation_ratio"]])
        print("{} / {} units failed to fit curve for modulation ratio estimate".format(n_filtered, len(group_data)))

        # Now actually filter those units out
        return [u for u in group_data if u["modulation_ratio"] and u["max_mean_grating_response"] >= response_threshold]

    # Filter unresponsive units (in place method)
    def filter_nonresponding_units (self):
        for group_idx, group in enumerate(self.data):
            filtered = []
            for u in group:
                max_mean = np.max(u["mean_grating_responses"])
                if max_mean > 0:
                    filtered.append(u)

            self.data[group_idx] = filtered

            original_n = self.hidden_units[group_idx]
            filtered_n = len(self.data[group_idx])
            print(f"{filtered_n} / {original_n} units kept after filtering non-responsive units")

        return self


    def get_moving_bar_stimuli (self, direction, x, y, bar_amplitude=1, bar_size=5, frames_len = 20):
        # Include warmup period for network in stimulus
        # Where first n frames will be gray (all 0's)
        total_frames = self.warmup + frames_len

        stimuli = np.zeros((total_frames, self.frame_shape[0], self.frame_shape[1]))

        for i in range(frames_len):
            bar_position = 0
            square = np.ones((bar_size, bar_size))*bar_amplitude

            if direction == 0 or direction == 270:
                bar_position = (bar_size - i) % bar_size
            else:
                bar_position = i%(bar_size) 

            if direction == 0 or direction == 180:
                square[bar_position, :] = -bar_amplitude
            else:
                square[:, bar_position] = -bar_amplitude

            stimulus = np.zeros((self.frame_shape[0], self.frame_shape[1]))
            stimulus[y:y+bar_size, x:x+bar_size] = square

            stimuli[self.warmup+i, :, :] = stimulus

        # Reshape frame into flat array and convert into Tensor object
        stimuli = stimuli.reshape(total_frames, self.frame_size)
        stimuli = torch.Tensor(stimuli).unsqueeze(0).to(self.device)

        return stimuli
