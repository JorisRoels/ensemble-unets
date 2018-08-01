
import torch.optim as optim
import os

import data.epfl as epfl
from networks.blocks import *
from networks.unet import UNet2D, AutoEncoder2D
from util.preprocessing import get_augmenters
from util.tools import load_net
from util.losses import JaccardLoss
from train import train_autoencoder, train_unet2d

# 1) trains 2D U-Nets in the XY, YZ and XZ direction from scratch as a baseline
# 2) trains 2D U-Net autoencoders in the 3 directions and uses these to initialize the segmentation U-Net encoders
#    the decoder of the U-Net is initialized with that of another pretrained network if available
def train_models():

    # data parameters
    Nxy, Nxz, Nyz = 128, 128, 128       # input size of the data in the three different directions
    in_channels = 1                     # number of input channels
    out_channels = 2                    # number of output classes

    # network parameters
    depth = 4                           # depth of the networks
    feature_scale = 4                   # feature scaling factor for a less wide U-Net (1 corresponds to the original network)
    deconv = True                       # use deconvolution layers in the upsampling path
    batch_norm = True                   # use batch normalization
    skip_connections = True             # use skip connection in the U-Nets

    # optimization parameters
    loss_unet = JaccardLoss()           # loss function for the unet networks
    loss_autoencoder = nn.MSELoss()     # loss function for the autoencoders
    learning_rate = 0.001               # learning rate for the U-Net optimization
    step_size_lr = 10                   # number of iterations between each learning rate decay
    lr_reduction_factor = 0.9           # multiplicative learning rate decay factor
    len_epoch = 100                     # number of iterations in one training epoch
    epochs = 10                         # number of epochs to train the networks
    train_batch_size = 4                # training batch size
    test_batch_size = 1                 # test batch size

    # logging parameters
    log_dir = 'logs'                    # logging directory
    test_freq = 1                       # number of epochs between each test phase
    print_stats = 10                    # number of iterations between each time training information is printed
    write_images_freq = 50              # number of epochs between each time images are logged

    # setup logging directories
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    net_dirs = ['unets_scratch', 'autoencoders', 'unets_finetuned']
    for dir in net_dirs:
        if not os.path.exists(os.path.join(log_dir, dir)):
            os.mkdir(os.path.join(log_dir, dir))
        subdirs = ['0', '1', '2']
        for subdir in subdirs:
            if not os.path.exists(os.path.join(log_dir, dir, subdir)):
                os.mkdir(os.path.join(log_dir, dir, subdir))

    # setup augmentation and dataloaders
    mu, std = epfl.get_stats()
    train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters(mu, std)
    train_loader_xy, test_loader_xy = \
        epfl.get_loaders((1, Nxy, Nxy), train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                         len_epoch=len_epoch,
                         train_xtransform=train_xtransform, train_ytransform=train_ytransform,
                         test_xtransform=test_xtransform, test_ytransform=test_ytransform, orientation=(0,1,2))
    train_loader_xz, test_loader_xz = \
        epfl.get_loaders((1, Nxz, Nxz), train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                         len_epoch=len_epoch,
                         train_xtransform=train_xtransform, train_ytransform=train_ytransform,
                         test_xtransform=test_xtransform, test_ytransform=test_ytransform, orientation=(1,2,0))
    train_loader_yz, test_loader_yz = \
        epfl.get_loaders((1, Nyz, Nyz), train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                         len_epoch=len_epoch,
                         train_xtransform=train_xtransform, train_ytransform=train_ytransform,
                         test_xtransform=test_xtransform, test_ytransform=test_ytransform, orientation=(2,0,1))
    train_loaders = [train_loader_xy, train_loader_xz, train_loader_yz]
    test_loaders = [test_loader_xy, test_loader_xz, test_loader_yz]

    # for each orientation: train 2D U-Nets from scratch (baseline)
    print('Training 2D U-Nets from scratch...')
    for i in range(3):
        print('Network %d' % (i+1))
        net = UNet2D(depth, feature_scale, in_channels, out_channels, deconv, batch_norm, skip_connections)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_lr, gamma=lr_reduction_factor)
        train_unet2d.train(net, train_loaders[i], test_loaders[i], loss_unet, optimizer, scheduler=scheduler, epochs=epochs, test_freq=test_freq,
                           print_stats=print_stats, log_dir=os.path.join(log_dir, net_dirs[0], str(i)), write_images_freq=write_images_freq)

    # for each orientation: train 2D U-Net autoencoders from scratch
    print('Training 2D U-Net autoencoders from scratch...')
    for i in range(3):
        print('Network %d' % (i+1))
        net = AutoEncoder2D(depth, feature_scale, in_channels, deconv, batch_norm)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_lr, gamma=lr_reduction_factor)
        train_autoencoder.train(net, train_loaders[i], test_loaders[i], loss_autoencoder, optimizer, scheduler=scheduler, epochs=epochs, test_freq=test_freq,
                                print_stats=print_stats, log_dir=os.path.join(log_dir, net_dirs[1], str(i)), write_images_freq=write_images_freq)

    # initialize the contractive paths of the first 2D U-Net with the contractive path of the trained autoencoder and finetune the network
    print('Finetuning 2D U-Nets from autoencoder...')
    print('Network %d' % (1))
    autoencoder = load_net(os.path.join(log_dir, net_dirs[1], str(0), 'checkpoint.pytorch'))
    net = UNet2D(autoencoder, out_channels, skip_connections)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_lr, gamma=lr_reduction_factor)
    train_unet2d.train(net, train_loaders[0], test_loaders[0], loss_unet, optimizer, scheduler=scheduler, epochs=epochs, test_freq=test_freq,
                       print_stats=print_stats, log_dir=os.path.join(log_dir, net_dirs[2], str(0)), write_images_freq=write_images_freq)

    # initialize the contractive paths of the other 2D U-Nets with the contractive path of the trained autoencoders
    # initialize the expansive path of the other 2D U-Nets with the expansive path of the previous trained 2D U-Net
    # finetune the network
    for i in range(1,3):
        print('Network %d' % (i+1))
        autoencoder = load_net(os.path.join(log_dir, net_dirs[1], str(i), 'checkpoint.pytorch'))
        unet_previous = load_net(os.path.join(log_dir, net_dirs[2], str(i-1), 'checkpoint.pytorch'))
        net = UNet2D(autoencoder, unet_previous, out_channels, skip_connections)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_lr, gamma=lr_reduction_factor)
        train_unet2d.train(net, train_loaders[i], test_loaders[i], loss_unet, optimizer, scheduler=scheduler, epochs=epochs, test_freq=test_freq,
                           print_stats=print_stats, log_dir=os.path.join(log_dir, net_dirs[2], str(i)), write_images_freq=write_images_freq)

# main function
if __name__ == "__main__":
    train_models()