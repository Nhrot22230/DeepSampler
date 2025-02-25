import torch
import torch.nn as nn


class TDFBlock(nn.Module):
    """
    Time-Distributed Fully-connected Network with bottleneck architecture
    Processes each frequency vector independently across channels and time steps
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int = 1,
        bottleneck_factor: int = 1,
        min_units: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.bottleneck_factor = bottleneck_factor
        self.min_units = min_units

        layers = []
        current_dim = in_features

        # Build hidden layers with bottleneck
        for _ in range(num_layers - 1):
            hidden_dim = max(in_features // bottleneck_factor, min_units)
            layers += [
                nn.Linear(current_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            current_dim = hidden_dim

        # Final layer maps back to original dimension
        layers += [
            nn.Linear(current_dim, in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
        ]

        self.tdf = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (B, C, F, T)
        B, C, F, T = x.shape

        # Reshape for time-distributed processing
        x_processed = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x_processed = x_processed.reshape(-1, F)  # (B*T*C, F)

        # Apply TDF transformations
        x_processed = self.tdf(x_processed)

        # Restore original shape
        x_processed = x_processed.view(B, T, C, F)
        return x_processed.permute(0, 2, 3, 1)  # (B, C, F, T)


class TDCBlock(nn.Module):
    """
    Time-Distributed Convolutions (TDC) with dense connections
    Implements a dense block structure where each layer's output is concatenated
    with all previous outputs for subsequent layers
    """

    def __init__(self, num_layers: int, growth_rate: int, kernel_size: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size

        # Each layer takes all previous features as input
        self.layers = nn.ModuleList(
            [self._make_dense_layer(i) for i in range(num_layers)]
        )

    def _make_dense_layer(self, layer_num: int):
        """Create a composite layer with dense connections"""
        in_channels = self.growth_rate * (layer_num + 1)  # Accumulated channels
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                self.growth_rate,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.growth_rate),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Input shape: (B*T, C, F)
        features = [x]

        for layer in self.layers:
            # Concatenate all previous features along channel dimension
            new_input = torch.cat(features, dim=1)
            new_features = layer(new_input)
            features.append(new_features)

        # Return concatenation of all features except original input
        return torch.cat(features[1:], dim=1)


class TFCBlock(nn.Module):
    """
    Time-Frequency Convolutions (TFC) dense block
    Implements dense connections where each layer receives all previous features
    """

    def __init__(
        self,
        num_layers: int,
        in_ch: int,
        growth_rate: int,
        kernel_size: tuple = (3, 3),
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size

        # Create composite layers with proper channel growth
        self.layers = nn.ModuleList()
        current_channels = in_ch
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(
                    current_channels,
                    growth_rate,
                    kernel_size=kernel_size,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(growth_rate, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )
            self.layers.append(layer)
            current_channels += growth_rate  # Update input channels for next layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features along channel dimension
            x_in = torch.cat(features, dim=1)
            x_out = layer(x_in)
            features.append(x_out)

        # Return concatenation of all layer outputs (excluding original input)
        return torch.cat(features[1:], dim=1)


class TFCTDFBlock(nn.Module):
    """
    TFC-TDF v3 Block combining:
    - Time-Frequency Convolutions (TFC)
    - Time-Distributed Fully-connected (TDF)
    - Residual connections
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_tfc_layers: int = 3,
        kernel_size: tuple = (3, 3),
        bottleneck_factor: int = 16,
        min_units: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels

        # First TFC block with transition
        self.tfc1 = TFCBlock(num_tfc_layers, in_channels, growth_rate, kernel_size)
        tfc1_out_channels = in_channels + growth_rate * num_tfc_layers
        self.tfc1_transition = nn.Conv2d(tfc1_out_channels, in_channels, 1)

        # TDF block (processes frequency dimension)
        self.tdf = TDFBlock(
            in_features=in_channels,  # Frequency dimension
            num_layers=2,  # Paper uses 2-layer bottleneck
            bottleneck_factor=bottleneck_factor,
            min_units=min_units,
        )

        # Second TFC block with transition
        self.tfc2 = TFCBlock(num_tfc_layers, in_channels, growth_rate, kernel_size)
        tfc2_out_channels = in_channels + growth_rate * num_tfc_layers
        self.tfc2_transition = nn.Conv2d(tfc2_out_channels, in_channels, 1)

        # Final residual connection
        self.res_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        residual = self.res_conv(x)

        # First TFC block
        x = self.tfc1(x)
        x = self.tfc1_transition(x)

        # TDF processing
        x = self.tdf(x)

        # Second TFC block
        x = self.tfc2(x)
        x = self.tfc2_transition(x)

        # Add residual connection
        return x + residual
