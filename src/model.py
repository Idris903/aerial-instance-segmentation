import torch


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False  # BatchNorm adds its own shift, so a conv bias would be redundant
            ),
            torch.nn.BatchNorm2d(out_channels),
            # inplace: overwrite the input tensor instead of allocating a new one, to save memory
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Encoder block: halves the spatial size, then extracts richer features
class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            DoubleConv(in_channels,
                       out_channels)
        )

    def forward(self, x):
        return self.down(x)


# Decoder block: doubles the spatial size, then the skip connection
# restores the fine spatial detail that pooling threw away
class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = torch.nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(in_channels,
                               out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # skip first, upsampled second
        return self.conv(x)


class Head(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU(inplace=True),
            # No padding needed: a 1x1 kernel is always centred on its own pixel,
            # so every position already gets an output
            torch.nn.Conv2d(
                64,
                out_channels,
                kernel_size=1,
            )
        )

    def forward(self, x):
        return self.head(x)


class Detector(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.doubleConv = DoubleConv(in_channels,
                                     64)

        self.down1 = Down(64,
                          128)

        self.down2 = Down(128,
                          256)

        self.down3 = Down(256,
                          512)

        self.down4 = Down(512,
                          1024)

        self.up1 = Up(1024,
                      512)

        self.up2 = Up(512,
                      256)

        self.heatmap = Head(256,
                            num_classes)

        self.size = Head(256,
                         2)

        self.offset = Head(256,
                           2)

        torch.nn.init.constant_(self.heatmap.head[-1].bias, -2.19)

    def forward(self, x):
        # Shapes are (channels, height, width) for a 512 x 512 input
        x1 = self.doubleConv(x)  # (64, 512, 512)
        x2 = self.down1(x1)      # (128, 256, 256)
        x3 = self.down2(x2)      # (256, 128, 128)
        x4 = self.down3(x3)      # (512, 64, 64)
        x5 = self.down4(x4)      # (1024, 32, 32)

        # up1 and up2 run in fp32. Under autocast (AMP) they would run in fp16, and up1's
        # first conv sums 1024 x 3 x 3 = 9,216 products per output, which can exceed fp16's
        # maximum (65,504) and become inf. In my previous model (a U-Net), that inf
        # corrupted BatchNorm's running statistics.
        with torch.autocast(device_type=x.device.type, enabled=False):
            # .float() converts the encoder outputs, which are fp16 under autocast
            x = self.up1(x5.float(), x4.float())  # (512, 64, 64)
            x = self.up2(x, x3.float())           # (256, 128, 128); x is already fp32

        heatmap = self.heatmap(x)
        size = self.size(x)
        offset = self.offset(x)

        return {"heatmap": heatmap, "size": size, "offset": offset}
        
        
                      
        
        
        