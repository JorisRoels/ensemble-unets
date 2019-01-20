
"""
    This is a script that tests multiple U-Nets as an ensemble
    Usage:
        python train.py --method 2D --net /path/to/network.pytorch
"""

"""
    Necessary libraries
"""
import numpy as np
import os
import argparse
import datetime

from util.validation import segment
from util.tools import load_net
from util.io import imwrite3D, read_tif
from util.preprocessing import normalize
from util.metrics import accuracy_metrics, jaccard, dice

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--method", help="Specifies 2D or 3D U-Net", type=str, default="2D")

# logging parameters
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)

# data parameters
parser.add_argument("--data", help="Path to the data (should be tif file)", type=str, default="../data/epfl/testing.tif")
parser.add_argument("--data_labels", help="Path to the data labels (should be tif file)", type=str, default="../data/epfl/testing_groundtruth.tif")

# network parameters
parser.add_argument("--nets", help="Path to the network", type=str, default="checkpoint1.pytorch,checkpoint2.pytorch,checkpoint3.pytorch")

# optimization parameters
parser.add_argument("--input_size_xy", help="Size of the XY blocks that propagate through the network", type=str, default="512,512")
parser.add_argument("--input_size_zx", help="Size of the XZ blocks that propagate through the network", type=str, default="160,160")
parser.add_argument("--input_size_yz", help="Size of the YZ blocks that propagate through the network", type=str, default="160,160")
parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
parser.add_argument("--crf_iterations", help="Number of CRF post-processing iterations (not applied if 0)", type=int, default=0)

args = parser.parse_args()
args.nets = [str(item) for item in args.nets.split(',')]
args.input_size_xy = [int(item) for item in args.input_size_xy.split(',')]
args.input_size_zx = [int(item) for item in args.input_size_zx.split(',')]
args.input_size_yz = [int(item) for item in args.input_size_yz.split(',')]

"""
    Setup writing directory
"""
print('[%s] Setting up write directories' % (datetime.datetime.now()))
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

"""
    Load and normalize the data
"""
print('[%s] Loading and normalizing the data' % (datetime.datetime.now()))
test_data = read_tif(args.data, dtype='uint8')
mu = np.mean(test_data)
std = np.std(test_data)
test_data = normalize(test_data, mu, std)
if len(test_data.shape)<3:
    test_data = test_data[np.newaxis, ...]

segmentation_cum = np.zeros_like(test_data)
orientations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
orientations_inv = ((0, 1, 2), (2, 0, 1), (1, 2, 0))
input_sizes = (args.input_size_xy, args.input_size_zx, args.input_size_yz)
for i, o in enumerate(orientations):

    """
        Load the network
    """
    print('[%s] Loading network' % (datetime.datetime.now()))
    net = load_net(args.nets[i])

    """
        Segmentation
    """
    print('[%s] Starting segmentation' % (datetime.datetime.now()))
    segmentation_cum += np.transpose(segment(np.transpose(test_data, o), net, input_sizes[i], batch_size=args.batch_size, crf_iterations=args.crf_iterations, mu=mu, std=std), orientations_inv[i])

segmentation = segmentation_cum/3

"""
    Validate the segmentation
"""
print('[%s] Validating segmentation' % (datetime.datetime.now()))
test_data_labels = read_tif(args.data_labels, dtype='uint8')
test_data_labels = normalize(test_data_labels, 0, 255)
j = jaccard(segmentation, test_data_labels)
d = dice(segmentation, test_data_labels)
a, p, r, f = accuracy_metrics(segmentation, test_data_labels)
print('[%s] RESULTS:' % (datetime.datetime.now()))
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
    imwrite3D(segmentation, args.write_dir, rescale=True)

print('[%s] Finished!' % (datetime.datetime.now()))
