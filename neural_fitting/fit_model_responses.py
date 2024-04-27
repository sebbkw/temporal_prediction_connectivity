##################
# Module imports #
##################

import os, argparse, sys
from itertools import product
import numpy as np

from scipy.optimize import curve_fit
from scipy import stats

from sklearn import linear_model
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

####################################
# Fit parameters and CLI arguments #
####################################

# Model name and offset
parser = argparse.ArgumentParser()
parser.add_argument('--ln', help='Train LN model', action='store_true')

is_arg_required = not '--ln' in sys.argv

parser.add_argument('--model_name', type=str, help='The name of the model.', required=is_arg_required)
parser.add_argument('--offset', type=int, help='Offset', required=is_arg_required)
parser.add_argument('--padding', type=int, help='Padding', required=is_arg_required)
parser.add_argument('--cell_start', type=int, help='Starting cell index', required=True)
parser.add_argument('--cell_end', type=int, help='Final cell index', required=True)
args = parser.parse_args()

should_fit_ln_model = args.ln
model_name = args.model_name
model_offset = args.offset
model_padding = args.padding
model_cell_start = args.cell_start
model_cell_end = args.cell_end

model_file_suffix = 'wt_60noise_200components'

# Print the arguments
if should_fit_ln_model:
    print('Fitting LN model')
else:
    print('Model name:', args.model_name)


#######################################
# Grid search parameters and CV folds #
#######################################

np.random.seed(0)

# Cross-validation folds
test_folds = [3, 8, 13]

train_folds = [
    [ 2, 12, 7,  1,  9, 11, 10,  5],
    [ 6, 12, 7,  1,  9, 11, 10,  5],
    [ 6,  4, 7,  1,  9, 11, 10,  5],
    [ 6,  4, 14, 1,  9, 11, 10,  5],
    [ 6,  4, 14, 2,  9, 11, 10,  5],
    [ 6,  4, 14, 2, 12, 11, 10,  5],
    [ 6,  4, 14, 2, 12,  7, 10,  5],
    [ 6,  4, 14, 2, 12,  7,  1,  5],
    [ 6,  4, 14, 2, 12,  7,  1,  9]
]
val_folds = [
    [ 6,  4, 14],
    [ 4, 14,  2],
    [14,  2, 12],
    [ 2, 12,  7],
    [12,  7,  1],
    [ 7,  1,  9],
    [ 1,  9, 11],
    [ 9, 11, 10],
    [11, 10,  5],
]
train_val_folds = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14]


L1_arr = np.logspace(1, -5, 40)

p0_arr = list(product([2, 1], [1, 0.5, 0.1], [1, 0], [0.5, 0, -0.5]))

############################
# Custom fitting functions #
############################

# CC_norm for prediction versus spikes
# R    = (trials, time bins)
# yhat = predicted spike rate
def get_cc_norm (R, yhat):
    N, T = R.shape

    y = np.nanmean(R, axis=0)

    Ey = np.mean(y)
    Eyhat = np.mean(yhat)
    Vy = np.sum((y-Ey)*(y-Ey))/T
    Vyhat = np.sum((yhat-Eyhat)*(yhat-Eyhat))/T
    Cyyhat = np.sum((y-Ey)*(yhat-Eyhat))/T

    SP = (np.nanvar(np.nansum(R,axis=0), ddof=1)-np.nansum(np.nanvar(R,axis=1, ddof=1)))/(N*(N-1))

    cc_norm = Cyyhat/np.sqrt(SP*Vyhat)

    if SP <= 0:
        return np.nan
    else:
        return cc_norm

# Sigmoidal non-linearity to fit to linear filter output
def sigmoid (x, a, b, c, d):
    return np.maximum(a/(1+np.exp(-(x-c)/b)) + d, 0)

@ignore_warnings(category=ConvergenceWarning)
def fit_sigmoid (linear_y_hat_tv, y_tv, p0):
    fitted_sigmoid_params, _ = curve_fit(
        sigmoid, linear_y_hat_tv, y_tv, p0=p0
    )
    return fitted_sigmoid_params

# Mean MSE across cross-folds for a given L1
PCA_TRANSFORMED_DATA = {}
@ignore_warnings(category=ConvergenceWarning)
def fit_ln_model (X, Y, y, L1, folds):
    cc_arr_nl = []
    cc_arr_l  = []
    clf_arr   = []

    fold_predictions = []
    fold_spikes      = []

    # Loop through each fold
    for train, val in folds:
        # Standardize data Fit linear filter
        if repr(train) in PCA_TRANSFORMED_DATA:
            X_train, X_val = PCA_TRANSFORMED_DATA[repr(train)]
        else:
            print('\t\t\tScaling')
            scaler = StandardScaler()
            X_train = scaler.fit_transform (X[train])
            X_val   = scaler.transform(X[val])

            # PCA transform
            print('\t\t\tPCA fitting')
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            PCA_TRANSFORMED_DATA[repr(train)] = X_train, X_val

        print('\t\t\tLinear fitting')
        clf = linear_model.Lasso(alpha=L1)
        clf.fit(X_train, y[train])
        clf_arr.append(clf)

        # Get linear predictions
        linear_y_hat_train = clf.predict(X_train)
        linear_y_hat_val   = clf.predict(X_val)

        # Fit sigmoid
        sigmoid_cc_arr = []
        sigmoid_params = []
        for p0 in p0_arr:
            try:
                fitted_params = fit_sigmoid (linear_y_hat_train, y[train], p0=p0)
            except Exception as e:
                print('\t\t\t\tError fitting sigmoid:', e)
                continue

            # Get non-linear prediction and CC on val set
            sigmoid_cc = get_cc_norm(Y[val].T, sigmoid(clf.predict(X_val), *fitted_params))
            sigmoid_cc_arr.append(sigmoid_cc)
            sigmoid_params.append(fitted_params)

        try:
            # Use initial params which yield best val set performance
            best_fitted_params = sigmoid_params[np.nanargmax(sigmoid_cc_arr)]
            # Get final non-linear prediction and CC on val set
            nonlinear_y_hat_val = sigmoid(clf.predict(X_val), *best_fitted_params)
        except Exception as e:
            # Use ReLU non-linearity
            print('\t\t\t\tError fitting sigmoid:', e)
            nonlinear_y_hat_val = relu(clf.predict(X_val))

        cc_arr_nl.append( get_cc_norm(Y[val].T, nonlinear_y_hat_val) )
        cc_arr_l.append( get_cc_norm(Y[val].T, linear_y_hat_val) )

        fold_predictions.append(nonlinear_y_hat_val)
        fold_spikes.append(y[val])

    return np.nanmean(cc_arr_nl), np.nanmean(cc_arr_l), fold_predictions, fold_spikes

def padded_matrix (cochleagram, padding_size):
    cochleagram_lst = []
    for t in range(padding_size, cochleagram.shape[0]):
        cochleagram_lst.append(cochleagram[t-padding_size:t].flatten())
    return np.array(cochleagram_lst)

def get_concatenated_data (
    model_responses, spiking_responses, padding, spike_offset
):
    model_responses_lst          = []
    spiking_responses_trials_lst = []
    spiking_responses_lst        = []

    for stim_name in spiking_responses['stimuli']:
        # Get model responses
        model_r = model_responses[stim_name]
        # Pad
        model_r = padded_matrix(model_r, padding)
        # Discard last time points due to spike offset
        if spike_offset:
            model_r = model_r[:-spike_offset]

        # Get spiking responses
        spiking_r        = spiking_responses['spikes_mean'][stim_name]
        spiking_trials_r = spiking_responses['spikes_trials'][stim_name] 
        # Discard first n tsteps due to padding
        spiking_r       = spiking_r[padding:]
        spiking_trials_r = spiking_trials_r[:, padding:]
        # Discard first spike_offset tsteps
        spiking_r       = spiking_r[spike_offset:]
        spiking_trials_r = spiking_trials_r[:, spike_offset:]

        # Append all to list to later concatenate
        model_responses_lst.append(model_r)
        spiking_responses_lst.append(spiking_r)
        spiking_responses_trials_lst.append(spiking_trials_r)

    return (
        np.concatenate(model_responses_lst, axis=0),
        np.concatenate(spiking_responses_trials_lst, axis=1).T,
        np.concatenate(spiking_responses_lst, axis=0)
    )

# Get cross validation folds (stim indices) from file indices
def get_cv_folds (X, stim_lst):
    per_file_size = X.shape[0]//len(stim_lst)

    CV_iterator = []

    for t_fold, v_fold in zip(train_folds, val_folds):
        t_idxs = []
        for f in t_fold:
            t_idxs += np.arange((f-1)*per_file_size, (f)*per_file_size).tolist()

        v_idxs = []
        for f in v_fold:
            v_idxs += np.arange((f-1)*per_file_size, (f)*per_file_size).tolist()

        CV_iterator.append((t_idxs, v_idxs))

    return CV_iterator

# Get cross validation folds (stim indices) from file indices
def get_cv_folds_test (X, stim_lst):
    per_file_size = X.shape[0]//len(stim_lst)

    CV_iterator = []

    for t_fold, v_fold in zip([train_val_folds], [test_folds]):
        t_idxs = []
        for f in t_fold:
            t_idxs += np.arange((f-1)*per_file_size, (f)*per_file_size).tolist()

        v_idxs = []
        for f in v_fold:
            v_idxs += np.arange((f-1)*per_file_size, (f)*per_file_size).tolist()

        CV_iterator.append((t_idxs, v_idxs))

    return CV_iterator

# Get dataset from list of cross validation folds
def get_dataset_from_folds (folds, X, Y, y, stim_lst):
    per_file_size = X.shape[0]//len(stim_lst)

    tv_idxs = []
    for f in folds:
        tv_idxs += np.arange((f-1)*per_file_size, (f)*per_file_size).tolist()
    X_tv = X[tv_idxs]
    Y_tv = Y[tv_idxs]
    y_tv = y[tv_idxs]

    return X_tv, Y_tv, y_tv

##########################
# Main fitting procedure #
##########################

try:
    if should_fit_ln_model:
        os.mkdir(f'./fits/LN_{model_file_suffix}')
    else:
        os.mkdir(f'./fits/{model_name}_{model_offset}offset_{model_padding}padding_{model_file_suffix}')
except FileExistsError:
    pass

cc_arr     = []
failed_arr = []

if should_fit_ln_model:
    model_responses = np.load(
        './neural_data/VISp/715093703_stimuli_raw.npy',
        allow_pickle=True
    ).item()
else:
    model_responses = np.load(
        f'./model_data/{model_name}.npy',
        allow_pickle=True
    ).item()

cell_data = np.load(
    './neural_data/VISp/wt_60noise_sessions_neural.npy',
    allow_pickle=True
).item()

for cell_idx, (cell_name, spiking_data) in enumerate(cell_data.items()):
    if cell_idx not in range(model_cell_start, model_cell_end):
        continue

    print('Starting cell', cell_name, f'({cell_idx}/{len(cell_data.keys())})')

    stim_lst = spiking_data['stimuli']
    noise    = spiking_data['noise_ratio']

    if should_fit_ln_model:
        padding      = model_padding
        spike_offset = model_offset
    else:
        padding      = model_padding
        spike_offset = model_offset
        
    X, Y, y = get_concatenated_data(model_responses, spiking_data, padding=padding, spike_offset=spike_offset)
    
    print(X.shape, Y.shape, len(y))
    assert(X.shape[0] == len(y))
    assert(X.shape[0] == Y.shape[0])
    
    X_tv, Y_tv, y_tv = get_dataset_from_folds(train_val_folds, X, Y, y, stim_lst)
    cv_folds = get_cv_folds(X, stim_lst)
    cv_folds_test = get_cv_folds_test(X, stim_lst)

    # Find best lambda
    mean_L1_scores_nl    = []
    mean_L1_scores_l     = []
    all_fold_predictions = []

    for L1 in L1_arr:
        print('\t\tFitting with L1 =', L1)
        try:
            nl_cc, l_cc, fold_predictions, fold_spikes = fit_ln_model(X, Y, y, L1, cv_folds)
            print('\t\t\tNon-linear CC', nl_cc, 'Linear CC', l_cc)
        except Exception as e:
            print('\t\t\tError for current L1', e)
            nl_cc, l_cc, fold_predictions = np.nan, np.nan, np.nan

        mean_L1_scores_nl.append( nl_cc )
        mean_L1_scores_l.append( l_cc )
        all_fold_predictions.append( fold_predictions )

    best_L1    = L1_arr[np.nanargmax(mean_L1_scores_nl)]
    best_cc_nl = np.nanmax(mean_L1_scores_nl)
    best_cc_l  = mean_L1_scores_l[np.nanargmax(mean_L1_scores_nl)]
    best_fold_predictions = all_fold_predictions[np.nanargmax(mean_L1_scores_nl)]

    nl_cc_test, l_cc_test, _, _ = fit_ln_model(X, Y, y, best_L1, cv_folds_test)

    if np.isfinite(best_cc_nl):
        cc_arr.append({
            'cell_name': cell_name, 'noise': noise,
            'cc_norm': best_cc_nl, 'cc_norm_linear': best_cc_l,
            'cc_norm_test': nl_cc_test, 'cc_norm_linear_test': l_cc_test,
            'best_L1': best_L1, 'y_hat': best_fold_predictions, 'y': fold_spikes
        })
        print('\tCross validated CC_norm =', best_cc_nl)

        if should_fit_ln_model:
            np.save(f'./fits/LN_{model_file_suffix}/{model_cell_start}.npy', cc_arr)
        else:
            np.save(f'./fits/{model_name}_{model_offset}offset_{model_padding}padding_{model_file_suffix}/{model_cell_start}.npy', cc_arr)
    else:
        failed_arr.append({
            'cell_name': cell_name, 'noise': noise,
            'coef': clf.coef_, 'error': 'NaN cc_norm'
        })
