import pytest
import torch
from torch import Tensor

# Import the classes under test.
# Adjust the import path to match your project structure.
from src.models.dino_sampler import TDCBlock, TDFBlock, TFCBlock, TFCTDFBlock

BATCH_SIZE = 2

##############################################
# TDFBlock Tests
##############################################


@pytest.mark.parametrize(
    "num_layers, bottleneck_factor, min_units", [(1, 1, 16), (2, 2, 16), (3, 2, 32)]
)
def test_tdf_block(num_layers: int, bottleneck_factor: int, min_units: int) -> None:
    """
    GIVEN a random input tensor with shape [BATCH_SIZE, channels, in_features, TIME],
          where in_features (frequency dimension) is set to 64 and channels is arbitrary (e.g., 4)
    WHEN the tensor is processed by a TDFBlock constructed with the specified num_layers,
         bottleneck_factor, and min_units parameters that apply a fully-connected transformation
         across the frequency dimension in a time-distributed manner
    THEN the output tensor should have the same shape as the input
         and a backward pass should successfully compute gradients.
    """
    in_features = 64
    channels = 4
    TIME = 8
    x: Tensor = torch.randn(BATCH_SIZE, channels, in_features, TIME, requires_grad=True)
    block = TDFBlock(in_features, num_layers, bottleneck_factor, min_units)
    y: Tensor = block(x)
    # The TDFBlock is expected to preserve the input shape.
    assert y.shape == x.shape, f"Expected shape {x.shape}, but got {y.shape}"
    loss = y.mean()
    loss.backward()
    assert x.grad is not None, "Gradients did not flow back to the input tensor"


##############################################
# TDCBlock Tests
##############################################


@pytest.mark.parametrize(
    "num_layers, growth_rate, kernel_size", [(1, 16, 3), (3, 8, 3), (5, 4, 5)]
)
def test_tdc_block(num_layers: int, growth_rate: int, kernel_size: int) -> None:
    """
    GIVEN a random input tensor with shape [BATCH_SIZE * TIME_FRAMES, growth_rate, FEATURE_LENGTH],
          where the channel dimension equals the growth_rate (ensuring compatibility with TDCBlock)
    WHEN the tensor is processed by the TDCBlock configured with the specified num_layers,
         growth_rate, and kernel_size, which applies a series of 1-D convolutions with dense
         concatenation
    THEN the output tensor should have shape
         [BATCH_SIZE * TIME_FRAMES, num_layers * growth_rate, FEATURE_LENGTH],
         and a backward pass should successfully compute gradients.
    """
    TIME_FRAMES = 4
    FEATURE_LENGTH = 64
    batch_time = BATCH_SIZE * TIME_FRAMES
    x: Tensor = torch.randn(batch_time, growth_rate, FEATURE_LENGTH, requires_grad=True)
    block = TDCBlock(num_layers, growth_rate, kernel_size)
    y: Tensor = block(x)
    expected_channels = num_layers * growth_rate
    expected_shape = (batch_time, expected_channels, FEATURE_LENGTH)
    assert (
        y.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {y.shape}"
    loss = y.mean()
    loss.backward()
    assert x.grad is not None, "Expected gradients to flow back to the input tensor"


##############################################
# TFCBlock Tests
##############################################


@pytest.mark.parametrize(
    "num_layers, in_ch, growth_rate, kernel_size",
    [(1, 3, 16, (3, 3)), (3, 8, 8, (3, 3)), (5, 4, 4, (5, 5))],
)
def test_tfc_block(
    num_layers: int, in_ch: int, growth_rate: int, kernel_size: tuple
) -> None:
    """
    GIVEN a random input tensor with shape [BATCH_SIZE, in_ch, H, W],
          where H and W are spatial dimensions (e.g., 32×32)
    WHEN the tensor is processed by the TFCBlock that implements dense 2-D convolutions with
         dense connections, using the specified num_layers, in_ch, growth_rate, and kernel_size
    THEN the output tensor should have shape [BATCH_SIZE, num_layers * growth_rate, H, W],
         and a backward pass should successfully compute gradients.
    """
    H, W = 32, 32
    x: Tensor = torch.randn(BATCH_SIZE, in_ch, H, W, requires_grad=True)
    block = TFCBlock(num_layers, in_ch, growth_rate, kernel_size)
    y: Tensor = block(x)
    expected_channels = num_layers * growth_rate
    expected_shape = (BATCH_SIZE, expected_channels, H, W)
    assert (
        y.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {y.shape}"
    loss = y.mean()
    loss.backward()
    assert x.grad is not None, "Expected gradients to flow back to the input tensor"


@pytest.mark.parametrize(
    "in_channels, growth_rate, num_tfc_layers, kernel_size, bottleneck_factor, min_units, T",
    [
        # Test with a small configuration:
        (32, 4, 3, (3, 3), 16, 16, 16),
        # Test with a larger configuration:
        (64, 8, 2, (3, 3), 32, 16, 32),
        (128, 8, 2, (3, 3), 32, 16, 32),
    ],
)
def test_tfctdf_block(
    in_channels: int,
    growth_rate: int,
    num_tfc_layers: int,
    kernel_size: tuple,
    bottleneck_factor: int,
    min_units: int,
    T: int,
) -> None:
    """
    GIVEN a random input tensor with shape [BATCH_SIZE, in_channels, F, T] where F equals
          in_channels (ensuring that the frequency dimension matches the expected 'in_features'
          for the TDF block)
    WHEN the tensor is processed by the TFCTDFBlock that combines:
         - A first TFC block with a transition,
         - A TDF block (with a 2-layer bottleneck),
         - A second TFC block with a transition, and
         - A final residual connection
    THEN the output tensor should have the same shape as the input,
         and a backward pass should successfully compute gradients.
    """
    # Here, we set the frequency dimension F equal to in_channels.
    F = in_channels
    x: Tensor = torch.randn(BATCH_SIZE, in_channels, F, T, requires_grad=True)

    block = TFCTDFBlock(
        in_channels=in_channels,
        growth_rate=growth_rate,
        num_tfc_layers=num_tfc_layers,
        kernel_size=kernel_size,
        bottleneck_factor=bottleneck_factor,
        min_units=min_units,
    )

    # Forward pass
    y: Tensor = block(x)

    # The output shape should match the input shape.
    assert y.shape == x.shape, f"Expected output shape {x.shape}, but got {y.shape}"

    # Verify that gradients flow: perform a backward pass.
    loss = y.mean()
    loss.backward()
    assert x.grad is not None, "Expected gradients to flow back to the input tensor"
