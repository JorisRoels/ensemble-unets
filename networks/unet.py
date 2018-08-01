
from networks.blocks import *

# helper class, because parameters in lists are not detected in modules
class AttrProxy(object):
    # translates index lookups into attribute lookups
    def __init__(self, module, prefix, n):
        self.module = module
        self.prefix = prefix
        self.n = n

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

    def __len__(self):
        return self.n

# 2D U-Net
# supports random initialization, initialization from an autoencoder and initialization from another U-Net
class UNet2D(nn.Module):

    def __init__(self, *args):
        super(UNet2D, self).__init__()
        if len(args) == 3:
            self.init_from_autoencoder(args[0], args[1], args[2])
        elif len(args) == 4:
            self.init_from_autoencoder_and_unet(args[0], args[1], args[2], args[3])
        else:
            self.init_default(args[0], args[1], args[2], args[3], args[4], args[5], args[6])

    def init_from_autoencoder_and_unet(self, autoencoder, unet, out_channels, skip_connections):

        self.deconv = autoencoder.deconv
        self.in_channels = autoencoder.in_channels
        self.batch_norm = autoencoder.batch_norm
        self.feature_scale = autoencoder.feature_scale
        self.depth = autoencoder.depth
        self.in_channels = autoencoder.in_channels
        self.out_channels = out_channels
        self.skip_connections = skip_connections

        n_filters_out = int(64 / self.feature_scale)
        n_filters_in = self.in_channels

        # contractive path
        for i in range(self.depth):
            self.add_module('conv' + str(i), autoencoder.convs[i])
            self.add_module('max_pool' + str(i), autoencoder.max_pools[i])
            n_filters_in = n_filters_out
            n_filters_out = int(n_filters_out*2)
        self.convs = AttrProxy(self, 'conv', self.depth)
        self.max_pools = AttrProxy(self, 'max_pool', self.depth)

        # center
        self.center = ConvBlock2D(n_filters_in, n_filters_out, self.batch_norm)

        # expansive path
        for i in range(self.depth):
            self.add_module('upconv' + str(i), unet.upconvs[i])
            n_filters_out = n_filters_in
            n_filters_in = int(n_filters_in/2)
        self.upconvs = AttrProxy(self, 'upconv', self.depth)

        # output layer
        self.output = unet.output

    def init_from_autoencoder(self, autoencoder, out_channels, skip_connections):

        self.deconv = autoencoder.deconv
        self.in_channels = autoencoder.in_channels
        self.batch_norm = autoencoder.batch_norm
        self.feature_scale = autoencoder.feature_scale
        self.depth = autoencoder.depth
        self.in_channels = autoencoder.in_channels
        self.out_channels = out_channels
        self.skip_connections = skip_connections

        n_filters_out = int(64 / self.feature_scale)
        n_filters_in = self.in_channels

        # contractive path
        for i in range(self.depth):
            self.add_module('conv' + str(i), autoencoder.convs[i])
            self.add_module('max_pool' + str(i), autoencoder.max_pools[i])
            n_filters_in = n_filters_out
            n_filters_out = int(n_filters_out*2)
        self.convs = AttrProxy(self, 'conv', self.depth)
        self.max_pools = AttrProxy(self, 'max_pool', self.depth)

        # center
        self.center = ConvBlock2D(n_filters_in, n_filters_out, self.batch_norm)

        # expansive path
        for i in range(self.depth):
            if self.skip_connections:
                self.add_module('upconv' + str(i), UpSamplingConcatBlock2D(n_filters_out, n_filters_in, self.deconv))
            else:
                self.add_module('upconv' + str(i), UpSamplingBlock2D(n_filters_out, n_filters_in, self.deconv))
            n_filters_out = n_filters_in
            n_filters_in = int(n_filters_in/2)
        self.upconvs = AttrProxy(self, 'upconv', self.depth)

        # output layer
        self.output = nn.Conv2d(n_filters_out, self.out_channels, 1)

    def init_default(self, depth, feature_scale, in_channels, out_channels, deconv, batch_norm, skip_connections):

        self.deconv = deconv
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        self.feature_scale = feature_scale
        self.depth = depth
        self.out_channels = out_channels
        self.skip_connections = skip_connections

        n_filters_out = int(64 / self.feature_scale)
        n_filters_in = in_channels

        # contractive path
        for i in range(depth):
            self.add_module('conv' + str(i), ConvBlock2D(n_filters_in, n_filters_out, self.batch_norm))
            self.add_module('max_pool' + str(i), nn.MaxPool2d(kernel_size=2))
            n_filters_in = n_filters_out
            n_filters_out = int(n_filters_out*2)
        self.convs = AttrProxy(self, 'conv', depth)
        self.max_pools = AttrProxy(self, 'max_pool', depth)

        # center
        self.center = ConvBlock2D(n_filters_in, n_filters_out, self.batch_norm)

        # expansive path
        for i in range(depth):
            if self.skip_connections:
                self.add_module('upconv' + str(i), UpSamplingConcatBlock2D(n_filters_out, n_filters_in, self.deconv))
            else:
                self.add_module('upconv' + str(i), UpSamplingBlock2D(n_filters_out, n_filters_in, self.deconv))
            n_filters_out = n_filters_in
            n_filters_in = int(n_filters_in/2)
        self.upconvs = AttrProxy(self, 'upconv', depth)

        # output layer
        self.output = nn.Conv2d(n_filters_out, out_channels, 1)

    def forward(self, inputs):

        # contractive path
        outputs = inputs
        convs = []
        for i in range(self.depth):
            outputs = self.convs[i](outputs)
            convs.append(outputs)
            outputs = self.max_pools[i](outputs)

        # center
        outputs = self.center(outputs)

        # expansive path
        convs.reverse()
        for i in range(self.depth):
            if self.skip_connections:
                outputs = self.upconvs[i](convs[i],outputs)
            else:
                outputs = self.upconvs[i](outputs)

        # output layer
        outputs = self.output(outputs)

        return outputs

# 2D U-Net autoencoder (basically just omits the skip connections)
class AutoEncoder2D(nn.Module):

    def __init__(self, depth, feature_scale, in_channels, deconv, batch_norm):
        super(AutoEncoder2D, self).__init__()
        self.deconv = deconv
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        self.feature_scale = feature_scale
        self.depth = depth

        n_filters_out = int(64 / self.feature_scale)
        n_filters_in = in_channels

        # contractive path
        for i in range(depth):
            self.add_module('conv' + str(i), ConvBlock2D(n_filters_in, n_filters_out, self.batch_norm))
            self.add_module('max_pool' + str(i), nn.MaxPool2d(kernel_size=2))
            n_filters_in = n_filters_out
            n_filters_out = int(n_filters_out*2)
        self.convs = AttrProxy(self, 'conv', depth)
        self.max_pools = AttrProxy(self, 'max_pool', depth)

        # center
        self.center = ConvBlock2D(n_filters_in, n_filters_out, self.batch_norm)

        # expansive path
        for i in range(depth):
            self.add_module('upconv' + str(i), UpSamplingBlock2D(n_filters_out, n_filters_in, self.deconv))
            n_filters_out = n_filters_in
            n_filters_in = int(n_filters_in/2)
        self.upconvs = AttrProxy(self, 'upconv', depth)

        # output layer
        self.output = nn.Conv2d(n_filters_out, in_channels, in_channels)

    def forward(self, inputs):

        # contractive path
        outputs = inputs
        for i in range(self.depth):
            outputs = self.convs[i](outputs)
            outputs = self.max_pools[i](outputs)

        # center
        outputs = self.center(outputs)

        # expansive path
        for i in range(self.depth):
            outputs = self.upconvs[i](outputs)

        # output layer
        outputs = self.output(outputs)
        outputs = F.sigmoid(outputs) # normalize to [0,1] interval

        return outputs

# 3D U-Net autoencoder (basically just omits the skip connections)
class AutoEncoder3D(nn.Module):

    def __init__(self, depth, feature_scale, in_channels, deconv, batch_norm):
        super(AutoEncoder3D, self).__init__()
        self.deconv = deconv
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        self.feature_scale = feature_scale
        self.depth = depth

        n_filters_out = int(64 / self.feature_scale)
        n_filters_in = in_channels

        # contractive path
        for i in range(depth):
            self.add_module('conv' + str(i), ConvBlock3D(n_filters_in, n_filters_out, self.batch_norm))
            self.add_module('max_pool' + str(i), nn.MaxPool3d(kernel_size=2))
            n_filters_in = n_filters_out
            n_filters_out = int(n_filters_out*2)
        self.convs = AttrProxy(self, 'conv', depth)
        self.max_pools = AttrProxy(self, 'max_pool', depth)

        # center
        self.center = ConvBlock3D(n_filters_in, n_filters_out, self.batch_norm)

        # expansive path
        for i in range(depth):
            self.add_module('upconv' + str(i), UpSamplingBlock3D(n_filters_out, n_filters_in, self.deconv))
            n_filters_out = n_filters_in
            n_filters_in = int(n_filters_in/2)
        self.upconvs = AttrProxy(self, 'upconv', depth)

        # output layer
        self.output = nn.Conv3d(n_filters_out, in_channels, 1)

    def forward(self, inputs):

        # contractive path
        outputs = inputs
        for i in range(self.depth):
            outputs = self.convs[i](outputs)
            outputs = self.max_pools[i](outputs)

        # center
        outputs = self.center(outputs)

        # expansive path
        for i in range(self.depth):
            outputs = self.upconvs[i](outputs)

        # output layer
        outputs = self.output(outputs)
        outputs = F.sigmoid(outputs) # normalize to [0,1] interval

        return outputs