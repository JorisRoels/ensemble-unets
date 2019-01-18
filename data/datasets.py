
import os
import numpy as np
import torch.utils.data as data
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input, sample_unlabeled_input

class LabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, label_path, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, dtypes=('uint8','uint8')):

        self.data_path = data_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data = read_tif(data_path, dtype=dtypes[0])
        self.labels = read_tif(label_path, dtype=dtypes[1])

        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.preprocess = preprocess
        if preprocess == 'z':
            self.data = normalize(self.data, mu, std)
        elif preprocess == 'unit':
            self.data = normalize(self.data, 0, 255)
        self.labels = normalize(self.labels, 0, 255)

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target[np.newaxis, ...]
        else:
            return input, target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

    # rotates data: (zyx -> yxz, yxz -> xzy, xzy -> zyx)
    def rotate_data(self):
        self.data = np.transpose(self.data, (1, 2, 0))
        self.labels = np.transpose(self.labels, (1, 2, 0))

    # sets the input_shape parameter
    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

class UnlabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, input_shape, len_epoch=100, preprocess='z', transform=None, dtype='uint8'):

        self.data_path = data_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform

        self.data = read_tif(data_path, dtype=dtype)

        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.preprocess = preprocess
        if preprocess == 'z':
            self.data = normalize(self.data, mu, std)
        elif preprocess == 'unit':
            self.data = normalize(self.data, 0, 255)

    def __getitem__(self, i):

        # get random sample
        input = sample_unlabeled_input(self.data, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...]
        else:
            return input

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

class EPFLTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None):
        super(EPFLTrainDataset, self).__init__(os.path.join('../data', 'epfl', 'training.tif'),
                                               os.path.join('../data', 'epfl', 'training_groundtruth.tif'),
                                               input_shape,
                                               len_epoch=len_epoch,
                                               preprocess=preprocess,
                                               transform=transform,
                                               target_transform=target_transform)

class EPFLTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None):
        super(EPFLTestDataset, self).__init__(os.path.join('../data', 'epfl', 'testing.tif'),
                                              os.path.join('../data', 'epfl', 'testing_groundtruth.tif'),
                                              input_shape,
                                              len_epoch=len_epoch,
                                              preprocess=preprocess,
                                              transform=transform,
                                              target_transform=target_transform)

class EPFLPixelTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None):
        super(EPFLPixelTrainDataset, self).__init__(os.path.join('../data', 'epfl', 'training.tif'),
                                               os.path.join('../data', 'epfl', 'training_groundtruth.tif'),
                                               input_shape,
                                               len_epoch=len_epoch,
                                                    preprocess=preprocess,
                                               transform=transform,
                                               target_transform=target_transform)

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class EPFLPixelTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None):
        super(EPFLPixelTestDataset, self).__init__(os.path.join('../data', 'epfl', 'testing.tif'),
                                                    os.path.join('../data', 'epfl', 'testing_groundtruth.tif'),
                                                    input_shape,
                                                    len_epoch=len_epoch,
                                                   preprocess=preprocess,
                                                    transform=transform,
                                                    target_transform=target_transform)

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class EMBLMitoTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLMitoTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

class EMBLMitoTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLMitoTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                  preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

class EMBLERTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLERTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'er_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                 preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

class EMBLERTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLERTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                  os.path.join('../data', 'embl', 'er_labels.tif'),
                                                  input_shape,
                                                  len_epoch=len_epoch,
                                                preprocess=preprocess,
                                                  transform=transform,
                                                  target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

class EMBLMitoPixelTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLMitoPixelTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                        preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class EMBLMitoPixelTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLMitoPixelTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                       preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class EMBLERPixelTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLERPixelTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'er_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                      preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class EMBLERPixelTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(EMBLERPixelTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                  os.path.join('../data', 'embl', 'er_labels.tif'),
                                                  input_shape,
                                                  len_epoch=len_epoch,
                                                     preprocess=preprocess,
                                                  transform=transform,
                                                  target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class VNCTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(VNCTrainDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                              preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

class VNCTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(VNCTestDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                             preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

class VNCPixelTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(VNCPixelTrainDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class VNCPixelTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(VNCPixelTestDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                  preprocess=preprocess,
                                                  transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class MEDTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(MEDTrainDataset, self).__init__(os.path.join('../data', 'med', 'data.tif'),
                                              os.path.join('../data', 'med', 'labels.tif'),
                                              input_shape,
                                              len_epoch=len_epoch,
                                              preprocess=preprocess,
                                              transform=transform,
                                              target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

class MEDTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(MEDTestDataset, self).__init__(os.path.join('../data', 'med', 'data.tif'),
                                             os.path.join('../data', 'med', 'labels.tif'),
                                             input_shape,
                                             len_epoch=len_epoch,
                                             preprocess=preprocess,
                                             transform=transform,
                                             target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

class MEDPixelTrainDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(MEDPixelTrainDataset, self).__init__(os.path.join('../data', 'med', 'data.tif'),
                                                   os.path.join('../data', 'med', 'labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   preprocess=preprocess,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, :s]
        self.labels = self.labels[:, :, :s]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

class MEDPixelTestDataset(LabeledVolumeDataset):

    def __init__(self, input_shape, len_epoch=100, preprocess='z', transform=None, target_transform=None, split=0.5):
        super(MEDPixelTestDataset, self).__init__(os.path.join('../data', 'med', 'data.tif'),
                                                  os.path.join('../data', 'med', 'labels.tif'),
                                                  input_shape,
                                                  len_epoch=len_epoch,
                                                  preprocess=preprocess,
                                                  transform=transform,
                                                  target_transform=target_transform)

        s = int(split * self.data.shape[2])
        self.data = self.data[:, :, s:]
        self.labels = self.labels[:, :, s:]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target