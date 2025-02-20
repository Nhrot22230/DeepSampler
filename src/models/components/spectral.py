import torch
import torch.nn as nn
from torch import Tensor


class SpectralEncoderBlock(nn.Module):
    """
    Spectrum-aware encoder block with frequency-wise attention.

    This module processes an input tensor through a convolutional block to extract
    spectral features, followed by an attention mechanism that operates along the frequency dim.
    The attention module first reduces the number of channels, applies multi-head attention to
    capture inter-frequency relationships, and then restores the original channel dimension.
    The output is a residual combination of the convolutional features and the
    attention-enhanced features.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        dropout (float): Dropout probability used in the convolutional and dropout layers.

    Attributes:
        conv (nn.Sequential): Sequential module consisting of a 2D convolution, batch normalization,
                              GELU activation, and dropout.
        attn (nn.Sequential): Sequential module that performs channel reduction, multi-head
                              attention, and channel restoration.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.attn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 8, 1),
            nn.MultiheadAttention(embed_dim=out_ch // 8, num_heads=4, batch_first=True),
            nn.Conv2d(out_ch // 8, out_ch, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes the forward pass of the spectral encoder block.

        The input tensor is first processed by the convolutional block to extract features.
        It is then permuted to rearrange its dimensions so that multi-head attention can be applied
        across the frequency dimension. After attention processing, the tensor is permuted back to
        its original layout, and a residual connection is added between the convolutional and
        attention outputs.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_ch, frequency, time].

        Returns:
            Tensor: Output tensor of shape [batch_size, out_ch, new_frequency, new_time] after
                    combining convolutional and attention features.
        """
        x = self.conv(x)
        attn_map = self.attn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + attn_map


class SpectralDecoderBlock(nn.Module):
    """
    Upsampling decoder block with frequency-aware transposed convolutions.

    This module upsamples the input feature maps using a transposed convolutional block and then
    merges the upsampled features with a corresponding skip connection from an earlier layer. The
    merged tensor is further processed by a convolutional block to refine the combined features.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        dropout (float): Dropout probability applied in the transposed convolution block.

    Attributes:
        upconv (nn.Sequential): Sequential module for upsampling, consisting of a transposed conv,
                                batch normalization, GELU activation, and dropout.
        merge (nn.Sequential): Sequential module that merges the upsampled features with the skips
                               using convolution, batch normalization, and GELU activation.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=(3, 5),
                stride=(1, 2),
                padding=(1, 2),
                output_padding=(0, 1),
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        Executes the forward pass of the spectral decoder block.

        The input tensor is first upsampled using transposed convolutions. It is then concatenated
        along the channel dimension with a skip connection tensor that carries information from an
        earlier encoding stage. The concatenated tensor is passed through a merging block to blend
        the features from both paths.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_ch, frequency, time] to be upsampled.
            skip (Tensor): Skip connection tensor of shape [batch_size, out_ch, frequency', time']
                           providing additional context from the encoder.

        Returns:
            Tensor: Output tensor of shape [batch_size, out_ch, frequency, time] after merging.
        """
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.merge(x)


class SpectrogramTransformer(nn.Module):
    """
    Lightweight transformer for spectral processing.

    This module applies multiple transformer encoder layers to process spectral data.
    It reshapes the input spectrogram tensor to a sequence format suitable for the transformer,
    applies a series of transformer encoder layers to capture temporal and frequency dependencies,
    and then reshapes the output back to the original spectrogram format.

    Args:
        dim (int): Dimensionality of the input features (channels after encoding).
        depth (int): Number of transformer encoder layers to apply (default: 4).
        heads (int): Number of attention heads in each transformer encoder layer (default: 4).

    Attributes:
        layers (nn.ModuleList): A list of transformer encoder layers configured with GELU,
                                feedforward expansion, and layer normalization applied
                                before attention.
    """

    def __init__(self, dim: int, depth: int = 4, heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * 4,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes the forward pass of the spectrogram transformer.

        The input tensor, representing a spectrogram with dimensions corresponding to batch size,
        channels, frequency bins, and time frames, is first permuted and flattened to create a
        sequence of feature vectors. Each transformer encoder layer processes the sequence to
        model dependencies. Finally, the sequence is reshaped back to
        the original spectrogram format.

        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, frequency, time].

        Returns:
            Tensor: Output tensor of shape [batch_size, channels, frequency, time] after
                    transformer encoding.
        """
        B, C, F, T = x.shape
        x = x.permute(0, 3, 2, 1).flatten(0, 1)
        for layer in self.layers:
            x = layer(x)
        return x.view(B, T, F, C).permute(0, 3, 2, 1)
