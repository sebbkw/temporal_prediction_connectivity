import os
import numpy as np
import torch
import torch.utils.data

class DatasetUnsupervised (torch.utils.data.Dataset):
    def __init__ (self, path, mmap=False):
        self.path = path

        if mmap:
            self.dataset = np.load(path, mmap_mode='r+')
        else:
            self.dataset = np.load(path)

    def __len__ (self):
        return len(self.dataset)

    def __getitem__ (self, i):
        clip = self.dataset[i]
        target = clip.copy()

        clip = torch.from_numpy(clip)
        clip = clip.type(torch.FloatTensor)

        target = torch.from_numpy(target)
        target = target.type(torch.FloatTensor)

        return clip, target

class DatasetSupervised (torch.utils.data.Dataset):
    def __init__ (self, paths, mmap=False):
        self.path = paths[0]

        if mmap:
            self.dataset_input  = np.load(paths[0], mmap_mode='r+')
            self.dataset_targets = np.load(paths[1], mmap_mode='r+')
        else:
            self.dataset_input  = np.load(paths[0])
            self.dataset_targets = np.load(paths[1])

        assert(len(self.dataset_input) == len(self.dataset_targets))

    def __len__ (self):
        return len(self.dataset_input)

    def __getitem__ (self, i):
        clip = self.dataset_input[i]
        target = self.dataset_targets[i]

        clip = torch.from_numpy(clip)
        clip = clip.type(torch.FloatTensor)

        target = torch.from_numpy(target)
        if len(target.shape) == 2:
            target = target.type(torch.FloatTensor)

        return clip, target

def data_loader (dataset_path, split, batch_size=256, num_workers=1, pin_memory=True, shuffle=True, mmap=True):
    if type(dataset_path) == list:
        full_dataset = DatasetSupervised(dataset_path, mmap)
    else:
        full_dataset = DatasetUnsupervised(dataset_path, mmap)

    len_full_dataset = 44000 #len(full_dataset)
    train_size = int(40/44 * len_full_dataset) #int(len(full_dataset)*0.9)
    test_size = len_full_dataset - train_size

    all_idxs = range(0, len_full_dataset)
    test_idxs = np.linspace(0, len_full_dataset-1, test_size, dtype=int)
    train_idxs = list(set(all_idxs).difference(test_idxs))

    if split == 'train':
        dataset_split = torch.utils.data.Subset(full_dataset, train_idxs)
    elif split == 'validation':
        dataset_split = torch.utils.data.Subset(full_dataset, test_idxs)
    elif split == 'full':
        dataset_split = torch.utils.data.Subset(full_dataset, all_idxs)
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset_split, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print("Dataset ({}) length: {}".format(split, len(dataset_split)))

    return data_loader

