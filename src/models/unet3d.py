import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        # Output
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        # Up 1
        x = self.up1(x4)
        # handle padding issues if tensor size isn't perfectly divisible by 2
        diffD = x3.size(2) - x.size(2)
        diffH = x3.size(3) - x.size(3)
        diffW = x3.size(4) - x.size(4)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                      diffH // 2, diffH - diffH // 2,
                      diffD // 2, diffD - diffD // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        
        # Up 2
        x = self.up2(x)
        diffD = x2.size(2) - x.size(2)
        diffH = x2.size(3) - x.size(3)
        diffW = x2.size(4) - x.size(4)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                      diffH // 2, diffH - diffH // 2,
                      diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        
        # Up 3
        x = self.up3(x)
        diffD = x1.size(2) - x.size(2)
        diffH = x1.size(3) - x.size(3)
        diffW = x1.size(4) - x.size(4)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                      diffH // 2, diffH - diffH // 2,
                      diffD // 2, diffD - diffD // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        return logits
