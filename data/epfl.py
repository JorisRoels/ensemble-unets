
import numpy as np
import os
import torch.utils.data as data
from torch.utils.data import DataLoader

from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input

class EPFL(data.Dataset):

    def __init__(self, input_shape, train=True, frac=1.0, orientation=(0,1,2),
                 len_epoch=1000, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = 'data'
        self.orientation = orientation

        if self.train:
            self.data = read_tif(os.path.join('../', self.data_dir, 'training.tif'), dtype='uint8')
            self.labels = read_tif(os.path.join('../', self.data_dir, 'training_groundtruth.tif'), dtype='int')
        else:
            self.data = read_tif(os.path.join('../', self.data_dir, 'testing.tif'), dtype='uint8')
            self.labels = read_tif(os.path.join('../', self.data_dir, 'testing_groundtruth.tif'), dtype='int')

        # orient the data
        self.data = np.transpose(self.data, orientation)
        self.labels = np.transpose(self.labels, orientation)

        # normalize data to [0,1] interval
        self.data = normalize(self.data, 0, 255)
        self.labels = normalize(self.labels, 0, 255)

        # optionally: use only a fraction of the data
        s = int(frac * self.data.shape[0])
        self.data = self.data[:s,:,:]
        self.labels = self.labels[:s,:,:]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):

        return self.len_epoch

# get train and test loaders
def get_loaders(input_shape, train_batch_size=1, test_batch_size=1, len_epoch=1000,
                train_xtransform=None, train_ytransform=None, test_xtransform=None, test_ytransform=None,
                train_frac=1.0, test_frac=1.0, orientation=(0,1,2)):

    # load datasets
    train_set = EPFL(input_shape, transform=train_xtransform, target_transform=train_ytransform,
                     len_epoch=len_epoch, train=True, frac=train_frac, orientation=orientation)
    test_set = EPFL(input_shape, transform=test_xtransform, target_transform=test_ytransform,
                    len_epoch=len_epoch, train=False, frac=test_frac, orientation=orientation)

    # build loaders
    train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size)

    return train_loader, test_loader

# compute first order statistics, useful for normalization
def get_stats():

    data_dir = 'data'

    # read train data
    data = read_tif(os.path.join('../', data_dir, 'training.tif'), dtype='uint8')

    # normalize data to [0,1] interval
    data = normalize(data, 0, 255)

    # compute first order statistics
    mu = np.mean(data)
    std = np.std(data)

    return mu, std