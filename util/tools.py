
import numpy as np
import torch

# generate an input and target sample of certain shape from a labeled dataset
def sample_labeled_input(data, labels, input_shape):

    # randomize seed
    np.random.seed()

    # generate random position
    x = np.random.randint(0, data.shape[0]-input_shape[0]+1)
    y = np.random.randint(0, data.shape[1]-input_shape[1]+1)
    z = np.random.randint(0, data.shape[2]-input_shape[2]+1)

    # extract input and target patch
    input = data[x:x+input_shape[0], y:y+input_shape[1], z:z+input_shape[2]]
    target = labels[x:x+input_shape[0], y:y+input_shape[1], z:z+input_shape[2]]

    return input, target

# load a network
def load_net(model_file):
    return torch.load(model_file)