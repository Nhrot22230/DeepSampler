import torch
import torch.nn as nn

from src.utils.train.tensor_logger import TensorLogger

from .components.attention import FreqAttention, TransformerModule
from .components.decoder import DecoderBlock
from .components.encoder import EncoderBlock


class DeepSampler(nn.Module):
    """Enhanced audio transformer combining AST principles (Gong et al., 2021)
    with selective attention mechanisms for source separation.

    Args:
        in_ch: Input channels (spectrogram bands)
        out_ch: Output sources
        base_ch: Base channels
        depth: Network depth
        dropout: Dropout probability
        t_layers: Transformer layers
        t_heads: Transformer heads
        t_dropout: Transformer dropout
        debug: Tensor logging
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 4,
        base_ch: int = 16,
        depth: int = 4,
        dropout: float = 0.2,
        t_layers: int = 2,
        t_heads: int = 2,
        t_dropout: float = 0.2,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.logger = TensorLogger(debug=debug, prefix="DINOs")

        # Channel progression follows M2M-AST design
        channels = [base_ch * (2**i) for i in range(depth + 1)]
        # Input processing (AST-style spectrogram embedding)
        self.init_conv = nn.Conv2d(in_ch, channels[0], kernel_size=1, padding=0)
        self.freq_attention = FreqAttention(channels=channels[0])

        # Encoder with progressive downsampling
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    dropout=dropout,
                    debug=debug,
                )
                for i in range(depth)
            ]
        )

        # Transformer module with selective attention
        self.transformer = TransformerModule(
            dim=channels[-1],
            layers=t_layers,
            heads=t_heads,
            dropout=t_dropout,
            debug=debug,
        )

        # Decoder with attention-guided skip connections
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    in_ch=channels[depth - i],
                    skip_ch=channels[depth - i - 1],
                    out_ch=channels[depth - i - 1],
                    dropout=dropout,
                    debug=debug,
                )
                for i in range(depth)
            ]
        )

        # Output projection with ASP (Audio Spectrogram Projection)
        self.final_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.log("input", x)

        # Initial processing with frequency attention
        x = self.logger.log("init_processed", self.init_conv(x))  # [B, BASE_CH, H, W]
        x = self.logger.log("freq_attention", self.freq_attention(x))  # [B, C, F, 1]

        # Encoder path with skip connections
        skips = []
        for i, enc in enumerate(self.encoders):
            skip, x = enc(x)
            skips.append(skip)
            self.logger.log(f"enc_{i + 1}_out", x)

        # Transformer processing (AST-style sequence modeling)
        x = self.logger.log("trans_in", x)
        x = self.transformer(x)
        x = self.logger.log("trans_out", x)

        # Decoder path with selective skip connections
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
            self.logger.log(f"dec_{i}_out", x)

        return self.logger.log("output", self.final_conv(x))
