
"""
    This is a script that trains 2D U-Nets in the XY, YZ and XZ direction from scratch
    Usage:
        python train.py --method 2D
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader

from data.datasets import *
from networks.unet import UNet2D
from util.preprocessing import get_augmenters_2d
from util.io import imwrite3D
from util.losses import CrossEntropyLoss
from util.validation import segment
from util.metrics import jaccard, dice, accuracy_metrics

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)
parser.add_argument("--data", help="Dataset for training", type=str, default="epfl") # options: 'epfl', 'embl_mito', 'embl_er', vnc, med
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=10)

# network parameters
parser.add_argument("--input_size_xy", help="Size of the XY blocks that propagate through the network", type=str, default="512,512")
parser.add_argument("--input_size_zx", help="Size of the XZ blocks that propagate through the network", type=str, default="160,160")
parser.add_argument("--input_size_yz", help="Size of the YZ blocks that propagate through the network", type=str, default="160,160")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=16)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)", type=int, default=4)
parser.add_argument("--group_norm", help="Use group normalization instead of batch normalization", type=int, default=0)
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=1)
parser.add_argument("--class_weight", help="Percentage of the reference class", type=float, default=(0.5))

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=1)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)

args = parser.parse_args()
args.input_size_xy = [int(item) for item in args.input_size_xy.split(',')]
args.input_size_zx = [int(item) for item in args.input_size_zx.split(',')]
args.input_size_yz = [int(item) for item in args.input_size_yz.split(',')]
weight = torch.FloatTensor([1-args.class_weight, args.class_weight]).cuda()
loss_fn_seg = CrossEntropyLoss(weight=weight)

"""
    Setup logging directory
"""
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
net_dirs = ['unets']
subdirs = ['XY', 'ZX', 'YZ']
for dir in net_dirs:
    if not os.path.exists(os.path.join(args.log_dir, dir)):
        os.mkdir(os.path.join(args.log_dir, dir))
    for subdir in subdirs:
        if not os.path.exists(os.path.join(args.log_dir, dir, subdir)):
            os.mkdir(os.path.join(args.log_dir, dir, subdir))
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    for subdir in subdirs:
        if not os.path.exists(os.path.join(args.write_dir, subdir)):
            os.mkdir(os.path.join(args.write_dir, subdir))
        os.mkdir(os.path.join(args.write_dir, subdir, 'segmentation_last_checkpoint'))
        os.mkdir(os.path.join(args.write_dir, subdir, 'segmentation_best_checkpoint'))

"""
    Load the data
"""
input_shape = (1, args.input_size_xy[1], args.input_size_xy[0])
# load data
print('[%s] Loading data' % (datetime.datetime.now()))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise==1))
if args.data == 'epfl':
    train = EPFLTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform)
    test = EPFLTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
elif args.data == 'vnc':
    train = VNCTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform)
    test = VNCTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
elif args.data == 'med':
    train = MEDTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform)
    test = MEDTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
else:
    if args.data == 'embl_mito':
        train = EMBLMitoTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform)
        test = EMBLMitoTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
    else:
        train = EMBLERTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform)
        test = EMBLERTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.test_batch_size)

"""
    Setup optimization for training
"""
print('[%s] Setting up optimization for training' % (datetime.datetime.now()))
net = UNet2D(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
input_size = args.input_size_xy

"""
    Train the network for each orientation
"""
print('[%s] Training 2D U-Nets from scratch...' % (datetime.datetime.now()))
for i in range(3):
    log_dir = os.path.join(args.log_dir, net_dirs[0], subdirs[i])
    if args.write_dir is not None:
        write_dir = os.path.join(args.write_dir, subdirs[i])
    print('[%s] Initiating training of network %d...' % (datetime.datetime.now(), i+1))
    # actual training
    net.train_net(train_loader=train_loader, test_loader=test_loader,
                  loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
                  epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
                  log_dir=log_dir)

    """
        Validate the trained network
    """
    print('[%s] Validating trained network %d...' % (datetime.datetime.now(), i+1))
    test_data = test.data
    test_labels = test.labels
    # segmentation_last_checkpoint = segment(test_data, net, input_size, batch_size=args.test_batch_size)
    segmentation_last_checkpoint = np.zeros_like(test_data)
    j = jaccard(segmentation_last_checkpoint, test_labels)
    d = dice(segmentation_last_checkpoint, test_labels)
    a, p, r, f = accuracy_metrics(segmentation_last_checkpoint, test_labels)
    print('[%s] Results last checkpoint:' % (datetime.datetime.now()))
    print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
    print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
    print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
    print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
    print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
    print('[%s]     F-score: %f' % (datetime.datetime.now(), f))
    net = torch.load(os.path.join(log_dir, 'best_checkpoint.pytorch'))
    # segmentation_best_checkpoint = segment(test_data, net, input_size, batch_size=args.test_batch_size)
    segmentation_best_checkpoint = np.zeros_like(test_data)
    j = jaccard(segmentation_best_checkpoint, test_labels)
    d = dice(segmentation_best_checkpoint, test_labels)
    a, p, r, f = accuracy_metrics(segmentation_best_checkpoint, test_labels)
    print('[%s] Results best checkpoint:' % (datetime.datetime.now()))
    print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
    print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
    print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
    print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
    print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
    print('[%s]     F-score: %f' % (datetime.datetime.now(), f))

    """
        Write out the results
    """
    if args.write_dir is not None:
        print('[%s] Writing the output' % (datetime.datetime.now()))
        imwrite3D(segmentation_last_checkpoint, os.path.join(write_dir, 'segmentation_last_checkpoint'), rescale=True)
        imwrite3D(segmentation_best_checkpoint, os.path.join(write_dir, 'segmentation_best_checkpoint'), rescale=True)

    """
        Prep next network training if necessary
    """
    if i<2:
        print('[%s] Setting up optimization for training next network' % (datetime.datetime.now()))
        # re-initialize for next run
        net = UNet2D(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        # rotating dataset for next run
        train.rotate_data()

        # adjusting input shape for next run
        if i==0:
            train.set_input_shape((1, args.input_size_zx[1], args.input_size_zx[0]))
            test.set_input_shape((1, args.input_size_zx[1], args.input_size_zx[0]))
            input_size = args.input_size_zx
        else:
            train.set_input_shape((1, args.input_size_yz[1], args.input_size_yz[0]))
            test.set_input_shape((1, args.input_size_yz[1], args.input_size_yz[0]))
            input_size = args.input_size_yz

print('[%s] Finished!' % (datetime.datetime.now()))