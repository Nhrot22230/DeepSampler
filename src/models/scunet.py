import torch
import torch.nn as nn


class SCUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, init_channels=64, depth=5):
        super(SCUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = init_channels
        for _ in range(depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = ch
            ch *= 2

        # Decoder
        ch = init_channels * (2 ** (depth - 1))
        for _ in range(depth - 1):
            self.decoder.append(
                nn.Sequential(
                    # Upsampling block
                    nn.ConvTranspose2d(
                        ch, ch // 2, 5, stride=2, padding=2, output_padding=1
                    ),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    # Convolution block
                    nn.Conv2d(ch, ch // 2, 3, padding=1),  # After concatenation
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(inplace=True),
                )
            )
            ch = ch // 2

        # Final 1x1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(init_channels, out_channels, 1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []

        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Decoder path
        for i, dec in enumerate(self.decoder):
            x = dec[0](x)  # Upsampling block
            x = torch.cat([x, skips[-(i + 2)]], dim=1)  # Skip connection
            x = dec[1:](x)  # Convolution block

        # Final convolution
        return self.final_conv(x)
