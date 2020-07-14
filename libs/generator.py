from torch import nn
import torch.nn.functional as F
import torch


def upconv(in_channels,out_channels, kernel_size, stride=2, padding=0, dilation = 1, output_padding = 0, batch_norm=True):
    layers = []
    if stride > 1:
        layers.append(nn.Upsample(scale_factor=stride))

    conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation = dilation, output_padding = output_padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.linear_bn = upconv(noise_size, conv_dim * 4, 4, padding=1)
        self.upconv1 = upconv(conv_dim * 4, conv_dim * 2, 4, stride = 2)
        self.upconv2 = upconv(conv_dim * 2, conv_dim, 4, stride = 2)
        self.upconv3 = upconv(conv_dim, conv_dim, 4, stride = 2, dilation = 1)
        self.upconv4 = upconv(conv_dim, conv_dim, 4, stride = 2, dilation = 2, output_padding = 1)
        self.upconv5 = upconv(conv_dim, 3, 4, stride = 2, dilation = 2, output_padding = 0, batch_norm=False)

    def forward(self, z):
        batch_size, dim = z.shape
        out = z.view(batch_size, dim, 1, 1)
        out = F.relu(self.linear_bn(out))
        out = F.relu(self.upconv1(out))
        out = F.relu(self.upconv2(out))
        out = F.relu(self.upconv3(out))
        out = F.relu(self.upconv4(out))
        out = torch.sigmoid(F.relu(self.upconv5(out)))
        return out
