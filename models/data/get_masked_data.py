import numpy as np

np.random.seed(0)

SAVE_DIR  = ''
FILE_NAME = ''

dataset = np.load(f'{SAVE_DIR}{FILE_NAME}.npy', mmap_mode='r')
size = 8

masked_dataset = np.zeros_like(dataset)

for x_i, x in enumerate(dataset):
    print('Starting clip', x_i, '/', dataset.shape[0])

    x_2d = x.copy().reshape(x.shape[0], 36, 36)

    for i in range(8):
        x_offset = np.random.randint(0, 36-size)
        y_offset = np.random.randint(0, 36-size)

        x_2d[:, y_offset:y_offset+size, x_offset:x_offset+size] = np.zeros((size, size))

    masked_dataset[x_i] = x_2d.reshape(-1, 36*36)

print('x:', masked_dataset.shape)
print('y:', dataset.shape)

np.save(f'{SAVE_DIR}masked_x_{FILE_NAME}.npy', masked_dataset)
np.save(f'{SAVE_DIR}masked_y_{FILE_NAME}.npy', dataset)
