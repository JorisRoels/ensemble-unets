
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2D convolutional layer with relu activation
class Conv2DRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding=0, bias=True, dilation=1):
        super(Conv2DRelu, self).__init__()

        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=padding, stride=stride, bias=bias, dilation=dilation),
                                  nn.ReLU(inplace=True) ,)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D deconvolution layer with relu activation
class Deconv2DRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2,  stride=2, padding=0, bias=True):
        super(Deconv2DRelu, self).__init__()

        self.unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                     padding=padding, stride=stride, bias=bias),
                                  nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D convolutional layer with batch normalization
class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding=0, bias=True, dilation=1):
        super(Conv2DBatchNorm, self).__init__()

        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=padding, stride=stride, bias=bias, dilation=dilation),
                                  nn.BatchNorm2d(int(out_channels)) ,)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D deconvolution layer with batch normalization
class Deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2,  stride=2, padding=0, bias=True):
        super(Deconv2DBatchNorm, self).__init__()

        self.unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                     padding=padding, stride=stride, bias=bias),
                                  nn.BatchNorm2d(int(out_channels)) ,)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D convolution layer with batch normalization and relu activation
class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding=0, bias=True, dilation=1):
        super(Conv2DBatchNormRelu, self).__init__()

        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=padding, stride=stride, bias=bias, dilation=dilation),
                                  nn.BatchNorm2d(int(out_channels)),
                                  nn.ReLU(inplace=True) ,)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D deconvolution layer with batch normalization and relu activation
class Deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2,  stride=2, padding=0, bias=True):
        super(Deconv2DBatchNormRelu, self).__init__()

        self.unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                     padding=padding, stride=stride, bias=bias),
                                  nn.BatchNorm2d(int(out_channels)),
                                  nn.ReLU(inplace=True) ,)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# block of two subsequent 2D convolutional layers with batch normalization and relu activation
class ConvBlock2D(nn.Module):

    def __init__(self, in_size, out_size, batch_norm, kernel_size=3, padding=1):
        super(ConvBlock2D, self).__init__()

        if batch_norm:
            self.conv1 = Conv2DBatchNormRelu(in_size, out_size, kernel_size=kernel_size, padding=padding)
            self.conv2 = Conv2DBatchNormRelu(out_size, out_size, kernel_size=kernel_size, padding=padding)
        else:
            self.conv1 = Conv2DRelu(in_size, out_size, kernel_size=kernel_size, padding=padding)
            self.conv2 = Conv2DRelu(out_size, out_size, kernel_size=kernel_size, padding=padding)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

# 2D block that upsamples the input and adds a convolutional block
class UpSamplingBlock2D(nn.Module):

    def __init__(self, in_size, out_size, deconv):
        super(UpSamplingBlock2D, self).__init__()

        if deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv = ConvBlock2D(out_size, out_size, False)

    def forward(self, inputs):

        outputs = self.up(inputs)

        return self.conv(outputs)

# 2D block that upsamples the input, concatenates with another layer and adds a convolutional block
class UpSamplingConcatBlock2D(nn.Module):

    def __init__(self, in_size, out_size, deconv):
        super(UpSamplingConcatBlock2D, self).__init__()

        if deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv = ConvBlock2D(in_size, out_size, False)

    def forward(self, inputs1, inputs2):

        outputs2 = self.up(inputs2)

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))

