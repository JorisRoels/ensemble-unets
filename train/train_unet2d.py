
import datetime
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

def train_epoch(net, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

    # make sure network is on the gpu and in training mode
    net.cuda()
    net.train()

    # keep track of the average loss during the epoch
    loss_cum = 0.0
    cnt = 0

    # start epoch
    for i, data in enumerate(loader):

        # get the inputs
        x, y = data[0].cuda(), data[1].cuda()

        # zero the gradient buffers
        net.zero_grad()

        # forward prop
        y_pred = net(x)

        # compute loss
        loss = loss_fn(y_pred, y)
        loss_cum += loss.data.cpu().numpy()
        cnt += 1

        # backward prop
        loss.backward()

        # apply one step in the optimization
        optimizer.step()

        # print statistics if necessary
        if i % print_stats == 0:
            print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                  % (datetime.datetime.now(), epoch, i, len(loader.dataset), loss))

    # don't forget to compute the average and print it
    loss_avg = loss_cum / cnt
    print('[%s] Epoch %5d - Average train loss: %.6f'
          % (datetime.datetime.now(), epoch, loss_avg))

    # log everything
    if writer is not None:

        # always log scalars
        writer.add_scalar('train/loss', loss_avg, epoch)

        if write_images:
            # write images
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
            y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
            writer.add_image('train/x', x, epoch)
            writer.add_image('train/y', y, epoch)
            writer.add_image('train/y_pred', y_pred, epoch)

    return loss_avg

def test_epoch(net, loader, loss_fn, epoch, writer=None, write_images=False):

    # make sure network is on the gpu and in training mode
    net.cuda()
    net.eval()

    # keep track of the average loss during the epoch
    loss_cum = 0.0
    cnt = 0

    # start epoch
    for i, data in enumerate(loader):

        # get the inputs
        x, y = data[0].cuda(), data[1].cuda()

        # forward prop
        y_pred = net(x)

        # compute loss
        loss = loss_fn(y_pred, y)
        loss_cum += loss.data.cpu().numpy()
        cnt += 1

    # don't forget to compute the average and print it
    loss_avg = loss_cum / cnt
    print('[%s] Epoch %5d - Average test loss: %.6f'
          % (datetime.datetime.now(), epoch, loss_avg))

    # log everything
    if writer is not None:

        # always log scalars
        writer.add_scalar('test/loss', loss_avg, epoch)

        if write_images:
            # write images
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
            y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
            writer.add_image('test/x', x, epoch)
            writer.add_image('test/y', y, epoch)
            writer.add_image('test/y_pred', y_pred, epoch)

    return loss_avg

def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1):

    # log everything if necessary
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    test_loss_min = np.inf
    for epoch in range(epochs):

        print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

        # train the model for one epoch
        train_epoch(net=net, loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                    print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

        # adjust learning rate if necessary
        if scheduler is not None:
            scheduler.step(epoch=epoch)

            # and keep track of the learning rate
            writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

        # test the model for one epoch is necessary
        if epoch % test_freq == 0:
            test_loss = test_epoch(net=net, loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer, write_images=True)

            # and save model if lower test loss is found
            if test_loss < test_loss_min:
                test_loss_min = test_loss
                torch.save(net, os.path.join(log_dir, 'best_checkpoint.pytorch'))

        # save model every epoch
        torch.save(net, os.path.join(log_dir, 'checkpoint.pytorch'))

    writer.close()
