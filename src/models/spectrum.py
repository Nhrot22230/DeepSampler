import torch.nn as nn
from torch import Tensor
from src.models.components.spectral import (
    SpectralEncoderBlock,
    SpectralDecoderBlock,
    SpectrogramTransformer,
)
import torch.nn.functional as F


class Spectrum(nn.Module):
    """
    Modern music source separation architecture combining multi-scale spectral processing,
    lightweight attention mechanisms, and frequency-aware transformations.

    This architecture employs an encoder-decoder design to separate musical sources from a mixture.
    The encoder path processes the input spectrogram through a series of spectral encoder blocks,
    each extracting multi-scale spectral features. The encoded representations are downsampled in
    the time dimension using average pooling and stored as skip connections for later refinement.
    A bottleneck module, consisting of 1x1 convolutions and a lightweight spectrogram transformer,
    further refines the features by modeling global dependencies. The decoder path then upsamples
    the bottleneck output using spectral decoder blocks, merging the upsampled features with the
    corresponding encoder skip connections to reconstruct the separated source spectrograms.
    Finally, an output layer produces a complex mask with alternating channels representing the real
    and imaginary components, used to separate the sources.

    Args:
        n_sources (int): Number of sources to separate. The output complex mask will have
                         n_sources * 2 channels.
        base_channels (int): Number of base channels for the initial encoder block.
        depth (int): Number of encoder and decoder blocks.
        bottleneck_dim (int): Number of channels in the bottleneck transformation.
        transformer_depth (int): Number of transformer encoder layers in the bottleneck module.
    """

    def __init__(
        self,
        n_sources: int = 4,
        base_channels: int = 32,
        depth: int = 5,
        bottleneck_dim: int = 256,
        transformer_depth: int = 2,
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        ch = base_channels
        for idx in range(depth):
            self.encoders.append(SpectralEncoderBlock(1 if idx == 0 else ch // 2, ch))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch // 2, bottleneck_dim, 1),
            SpectrogramTransformer(bottleneck_dim, depth=transformer_depth),
            nn.Conv2d(bottleneck_dim, ch // 2, 1),
        )

        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.decoders.append(SpectralDecoderBlock(ch // 2, ch // 4))
            ch //= 2

        self.output = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1),
            nn.Conv2d(base_channels, n_sources * 2, 1),
            nn.Tanh(),
        )

    def forward(self, mix: Tensor) -> Tensor:
        """
        Executes the forward pass of the DeepSpectrumSeparator.

        The input is expected to be a mixture spectrogram with shape
        [batch_size, 1, frequency, time].
        The encoder path sequentially processes the input through spectral encoder blocks,
        storing each intermediate feature map as a skip connection. After each encoder block,
        average pooling is applied to reduce the time dimension. The bottleneck module applies a
        1x1 convolution to adjust channel dimensions, processes the features using a lightweight
        transformer to capture global dependencies, and then restores the channel dimension with
        another 1x1 convolution.

        In the decoder path, each spectral decoder block upsamples the bottleneck output and merges
        it with the corresponding skip connection from the encoder via concatenation along the
        channel dimension. Finally, an output layer generates a complex mask with alternating
        channels representing the real and imaginary components. The mask is then recombined into a
        complex-valued tensor using the relation:
        output = real + 1j * imaginary.

        Args:
            mix (Tensor): Input tensor of shape [batch, 1, freq, time] representing the
                          mixture spectrogram.

        Returns:
            Tensor: Complex mask tensor computed as real + 1j * imag, used for source separation.
        """
        skips = []
        x = mix

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.avg_pool2d(x, (2, 1))

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        mask = self.output(x)
        real = mask[:, ::2]
        imag = mask[:, 1::2]
        return real + 1j * imag
