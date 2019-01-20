
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import progressbar as pb
from util.tools import gaussian_window
from util.crf import dense_crf

# sliding window iterator
def sliding_window(image, step_size, window_size):

    # define range
    zrange = [0]
    while zrange[-1] < image.shape[0] - window_size[0]:
        zrange.append(zrange[-1] + step_size[0])
    zrange[-1] = image.shape[0] - window_size[0]
    yrange = [0]
    while yrange[-1] < image.shape[1] - window_size[1]:
        yrange.append(yrange[-1] + step_size[1])
    yrange[-1] = image.shape[1] - window_size[1]
    xrange = [0]
    while xrange[-1] < image.shape[2] - window_size[2]:
        xrange.append(xrange[-1] + step_size[2])
    xrange[-1] = image.shape[2] - window_size[2]

    # loop over the range
    # bar = pb.ProgressBar(max_value=len(zrange)*len(yrange)*len(xrange))
    # i = 0
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # yield the current window
                yield (z, y, x, image[z:z+window_size[0], y:y+window_size[1], x:x+window_size[2]])
                # i += 1
                # bar.update(i)

# segment a data set with a given network with a sliding window
# data is assumed a 3D volume
# input shape should come as
#   - 2D (y_dim, x_dim)
#   - 3D (z_dim, y_dim, x_dim)
def segment(data, net, input_shape, batch_size=1, in_channels=1, step_size=None, crf_iterations=0, mu=0, std=1):

    # make sure we compute everything on the gpu and in evaluation mode
    net.cuda()
    net.eval()

    # 2D or 3D
    is2d = len(input_shape) == 2

    # upsampling might be necessary depending on the network
    interp = nn.Upsample(size=input_shape, mode='bilinear', align_corners=True)

    # set step size to half of the window if necessary
    if step_size == None:
        if is2d:
            step_size = (1, input_shape[0]//2, input_shape[1]//2)
        else:
            step_size = (input_shape[0]//2, input_shape[1]//2, input_shape[2]//2)

    # gaussian window for smooth block merging
    if is2d:
        g_window = gaussian_window((1,input_shape[0],input_shape[1]), sigma=input_shape[-1]/4)
    else:
        g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)

    # symmetric extension only necessary along z-axis if multichannel 2D inputs
    if is2d and in_channels>1:
        z_pad = in_channels // 2
        padding = ((z_pad, z_pad), (0, 0), (0, 0))
        data = np.pad(data, padding, mode='symmetric')
    else:
        z_pad = 0

    # allocate space
    seg_cum = np.zeros(data.shape)
    counts_cum = np.zeros(data.shape)

    # define sliding window
    if is2d:
        sw = sliding_window(data, step_size=step_size, window_size=(in_channels, input_shape[0],input_shape[1]))
    else:
        sw = sliding_window(data, step_size=step_size, window_size=input_shape)

    # start prediction
    batch_counter = 0
    if is2d:
        batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1]))
    else:
        batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1], input_shape[2]))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        if not is2d: # add channel in case of 3D processing, in 2D case, it's already there
            inputs = inputs[np.newaxis, ...]
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:

            # convert to tensors
            inputs = torch.FloatTensor(batch).cuda()

            # forward prop
            outputs = net(inputs)
            if input_shape[0] != outputs.size(2) or input_shape[1] != outputs.size(3):
                outputs = interp(outputs)
            outputs = F.softmax(outputs, dim=1)

            # cumulate segmentation volume
            for b in range(batch_size):
                (z_b, y_b, x_b) = positions[b, :]
                # take into account the gaussian filtering
                if is2d:
                    seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
                    counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
                else:
                    seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
                    counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

            # reset batch counter
            batch_counter = 0

    # don't forget last batch
    # convert to tensors
    inputs = torch.FloatTensor(batch).cuda()

    # forward prop
    outputs = net(inputs)
    if input_shape[0] != outputs.size(2) or input_shape[1] != outputs.size(3):
        outputs = interp(outputs)
    outputs = F.softmax(outputs, dim=1)

    # cumulate segmentation volume
    for b in range(batch_counter):
        (z_b, y_b, x_b) = positions[b, :]
        # take into account the gaussian filtering
        if is2d:
            seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
            counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
        else:
            seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
            counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

    # crop out the symmetric extension and compute segmentation
    segmentation = np.divide(seg_cum[0:counts_cum.shape[0]-2*z_pad, :, :],
                             counts_cum[0:counts_cum.shape[0] - 2*z_pad, :, :])

    # CRF post-processing if necessary
    if crf_iterations>0:
        data_denorm = data*std + mu
        segmentation = crf(data_denorm, segmentation, iterations=crf_iterations)

    return segmentation

# segment a data set with a given network with a sliding window
# data is assumed a 3D volume
def segment_pixels(data, net, input_shape, batch_size=1, in_channels=1, crf_iterations=0, mu=0, std=1):

    # make sure we compute everything on the gpu and in evaluation mode
    net.cuda()
    net.eval()

    # symmetric extension only necessary along z-axis if multichannel 2D inputs
    padding = ((in_channels//2, in_channels//2),
               (input_shape[0]//2, input_shape[0]//2),
               (input_shape[1]//2, input_shape[1]//2))
    data = np.pad(data, padding, mode='symmetric')

    # allocate space
    segmentation = np.zeros(data.shape)

    # define sliding window
    sw = sliding_window(data, step_size=(1,1,1), window_size=(in_channels, input_shape[0],input_shape[1]))

    # start prediction
    batch_counter = 0
    batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1]))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z+in_channels//2, y+input_shape[0]//2, x+input_shape[1]//2]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:

            # convert to tensors
            inputs = torch.FloatTensor(batch).cuda()

            # forward prop
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)

            # cumulate segmentation volume
            for b in range(batch_size):
                (z_b, y_b, x_b) = positions[b, :]
                segmentation[z_b, y_b, x_b] += outputs.data.cpu().numpy()[b, 1]

            # reset batch counter
            batch_counter = 0
    
    segmentation = segmentation[in_channels//2:segmentation.shape[0]-in_channels//2,
                        input_shape[0]//2:segmentation.shape[1]-input_shape[0]//2,
                        input_shape[1]//2:segmentation.shape[2]-input_shape[1]//2]
    
    data = data[in_channels//2:data.shape[0]-in_channels//2,
                        input_shape[0]//2:data.shape[1]-input_shape[0]//2,
                        input_shape[1]//2:data.shape[2]-input_shape[1]//2]

    # CRF post-processing if necessary
    if crf_iterations>0:
        data_denorm = data*std + mu
        segmentation = crf(data_denorm, segmentation, iterations=crf_iterations)

    return segmentation

# post-processing crf
# window size should be 2D
def crf(data, segmentation_probs, window_size=(512,512), iterations=10):

    step_size = (1, window_size[0]//2, window_size[1]//2)

    # gaussian window for smooth block merging
    g_window = gaussian_window((1,window_size[0],window_size[1]), sigma=window_size[-1]/4)

    # prep data
    seg_probs = segmentation_probs
    if len(data.shape)==2:
        data = data[np.newaxis,...]
        seg_probs = segmentation_probs[np.newaxis,...]
    probs = np.zeros((2, seg_probs.shape[0], seg_probs.shape[1], seg_probs.shape[2]))
    probs[0] = seg_probs
    probs[1] = 1-seg_probs
    # probs = -np.log(probs)

    sw = sliding_window(data, step_size=step_size, window_size=(1,window_size[0],window_size[1]))

    # allocate space
    seg_cum = np.zeros(data.shape)
    counts_cum = np.zeros(data.shape)

    for (z, y, x, inputs) in sw:

        # inputs_prepped = np.zeros((inputs.shape[1],inputs.shape[2],3)) # grayscale to rgb hack
        inputs_prepped = inputs[0]
        # inputs_prepped[...,1] = inputs[0]
        # inputs_prepped[...,2] = inputs[0]

        probs_block = probs[:, z, y:y + window_size[0], x:x + window_size[1]]

        # compute CRF
        seg_crf = dense_crf(inputs_prepped.astype(np.uint8), probs_block, iterations=iterations)

        # cumulate results
        seg_cum[z:z + 1, y:y + window_size[0], x:x + window_size[1]] += np.multiply(g_window, seg_crf[0])
        counts_cum[z:z + 1, y:y + window_size[0], x:x + window_size[1]] += g_window

    segmentation = np.divide(seg_cum, counts_cum)

    return segmentation
