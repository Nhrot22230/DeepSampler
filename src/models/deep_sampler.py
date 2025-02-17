import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class EncoderBlock(nn.Module):
    """
    Simplified Encoder Block.

    This block performs a single convolution followed by batch normalization,
    ReLU activation, and dropout. The output before pooling is saved as a skip connection,
    then a max pooling operation is applied.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder block.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, in_channels, height, width].

        Returns:
            tuple: A tuple (skip, pooled) where:
                - skip (torch.Tensor): The output from the convolution (before pooling), for skip.
                - pooled (torch.Tensor): The output after max pooling.
        """
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return skip, x


class DecoderBlock(nn.Module):
    """
    Simplified Decoder Block.

    This block performs upsampling using a transposed convolution, then concatenates
    the corresponding skip connection from the encoder, and finally applies a convolution
    block to fuse the features.

    Args:
        in_channels (int): Number of input channels from the previous decoder block.
        out_channels (int): Number of output channels.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder block.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder block, with shape [B, C, H, W].
            skip (torch.Tensor): Skip connection tensor from the corresponding encoder block,
                                 with shape [B, out_channels, H_skip, W_skip].

        Returns:
            torch.Tensor: Output tensor after upsampling, concatenation, and convolution,
                          with shape [B, out_channels, H_out, W_out].
        """
        x = self.up(x)
        # If spatial dimensions don't match, pad x to match the size of skip.
        if x.size() != skip.size():
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(
                x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        # Concatenate skip connection along the channel dimension.
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DeepSampler(nn.Module):
    """
    DeepSampler Network with a Latent Space Transformer.

    This encoder-decoder architecture (similar to U-Net) consists of:
      - A specified number of encoder and decoder blocks (defined by 'depth').
      - A bottleneck that doubles the number of channels.
      - A transformer that processes the latent space.
      - A final layer that maps to the desired number of output channels.

    Dynamic padding is applied to ensure the spatial dimensions are divisible by 2^depth.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 4,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        """
        Initializes the DeepSampler network.

        Args:
            input_channels (int): Number of channels in the input.
            output_channels (int): Desired number of output channels.
            base_channels (int): Number of channels for the first encoder block.
            depth (int): Number of encoder/decoder blocks.
            dropout (float): Dropout rate.
            transformer_heads (int): Number of heads in the transformer encoder.
            transformer_layers (int): Number of transformer encoder layers.
        """
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        encoder_channels = []

        # Build encoder blocks.
        in_ch = input_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(EncoderBlock(in_ch, out_ch, dropout=dropout))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck: doubles the number of channels.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
        )
        bottleneck_channels = in_ch * 2

        # Transformer operating in the latent space.
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=bottleneck_channels,
                nhead=transformer_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )

        # Build decoder blocks (in reverse order).
        for i in reversed(range(depth)):
            out_ch = encoder_channels[i]
            self.decoders.append(
                DecoderBlock(bottleneck_channels, out_ch, dropout=dropout)
            )
            bottleneck_channels = out_ch

        # Final convolution: maps to the desired output channels.
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepSampler network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, input_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, output_channels, H, W],
                          cropped to the original spatial dimensions.
        """
        # Record original dimensions.
        orig_h, orig_w = x.size(2), x.size(3)
        pad_factor = 2**self.depth

        # Calculate new dimensions divisible by pad_factor.
        new_h = (
            orig_h
            if orig_h % pad_factor == 0
            else orig_h + (pad_factor - orig_h % pad_factor)
        )
        new_w = (
            orig_w
            if orig_w % pad_factor == 0
            else orig_w + (pad_factor - orig_w % pad_factor)
        )

        # Compute dynamic padding (left, right, top, bottom).
        padding = (
            (new_w - orig_w) // 2,
            (new_w - orig_w) - (new_w - orig_w) // 2,
            (new_h - orig_h) // 2,
            (new_h - orig_h) - (new_h - orig_h) // 2,
        )
        x = F.pad(x, padding)

        # Encoder: process input through encoder blocks and save skip connections.
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck.
        x = self.bottleneck(x)

        # Transformer: reshape latent space and process.
        b, c, h_lat, w_lat = x.shape
        # Flatten spatial dimensions: [B, C, h_lat * w_lat] then permute to [B, seq_len, C]
        x_flat = x.view(b, c, h_lat * w_lat).permute(0, 2, 1)
        x_trans = self.transformer(x_flat)
        # Reshape back to [B, C, h_lat, w_lat]
        x = x_trans.permute(0, 2, 1).view(b, c, h_lat, w_lat)

        # Decoder: process through decoder blocks using skip connections.
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        # Final convolution to get output channels.
        x = self.final_conv(x)
        # Crop the output to the original dimensions.
        return x[:, :, :orig_h, :orig_w]
