import os
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="The dir name pattern to match.")
args = parser.parse_args()

fits = []

model_names = glob.glob('./fits/' + args.dir + '/*.npy')

print('Starting', args.dir)

fits = []
for file_name in model_names:
    print('Adding', file_name)

    try:
        fits.append(np.load(file_name, allow_pickle=True))
    except:
        print('Error')

fits_filtered = []
cell_filtered = []

for f in fits:
    for c in f:
        c_name = c['cell_name']
        if not c_name in cell_filtered:
            cell_filtered.append(c_name)

            del c['y_hat']
            del c['y']

            fits_filtered.append(c)

save_file = './processed_fits/' + args.dir + '.npy'
print('Saving', save_file, 'with', len(fits_filtered), 'cells')
np.save(save_file, fits_filtered)
