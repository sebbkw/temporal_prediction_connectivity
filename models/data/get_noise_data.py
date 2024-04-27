import numpy as np

np.random.seed(0)

SAVE_DIR  = ''
FILE_NAME = ''

dataset = np.load(f'{SAVE_DIR}{FILE_NAME}.npy', mmap_mode='r')
size = 8

noise_dataset = np.zeros_like(dataset)

for x_i, x in enumerate(dataset):
    print('Starting clip', x_i, '/', dataset.shape[0])

    snr = 3
    noise = np.random.normal(size=x.shape) * 1/(10**(snr/20))
    x_noise = x + noise

    noise_dataset[x_i] = x_noise

print('x:', noise_dataset.shape)
print('y:', dataset.shape)

np.save(f'{SAVE_DIR}noise_x_{FILE_NAME}.npy', noise_dataset)
np.save(f'{SAVE_DIR}noise_y_{FILE_NAME}.npy', dataset)
