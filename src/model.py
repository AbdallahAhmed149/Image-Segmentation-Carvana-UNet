"""U-Net architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder
        self.conv_up1 = DoubleConv(256 + 512, 256)
        self.conv_up2 = DoubleConv(128 + 256, 128)
        self.conv_up3 = DoubleConv(64 + 128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def _up_and_concat(x, skip):
        # interpolate to the skip's size -> works even if H/W aren't divisible by 8
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.conv_up1(self._up_and_concat(x4, x3))
        x = self.conv_up2(self._up_and_concat(x, x2))
        x = self.conv_up3(self._up_and_concat(x, x1))
        return self.outc(x)
