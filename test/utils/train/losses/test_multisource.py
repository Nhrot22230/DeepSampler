from test.utils_test import (
    NUM_SOURCES,
    generate_batched_tensors,
    generate_unbatched_tensors,
)

import pytest
import torch

from src.utils.train.losses import MultiSourceLoss


def test_unbatched_input_processing():
    """
    GIVEN unbatched inputs with shape [n_freq, time_steps] for each source,
    WHEN the MultiSourceLoss processes these inputs,
    THEN it should complete without error and return a non-negative scalar tensor.

    Preconditions:
        - 'outputs' and 'targets' are generated using random unbatched tensors.
    Postconditions:
        - Loss is a torch.Tensor and its value is non-negative.
    """
    outputs = generate_unbatched_tensors()
    targets = generate_unbatched_tensors()
    loss_fn = MultiSourceLoss(weights=[1.0] * NUM_SOURCES, distance="l1")
    loss = loss_fn(outputs, targets)
    assert isinstance(loss, torch.Tensor), "Loss must be a torch tensor."
    assert loss.item() >= 0, "Loss must be non-negative."


@pytest.mark.parametrize("distance", ["l1", "l2"])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_batched_input_processing(distance: str, batch_size: int):
    """
    GIVEN batched inputs with shape [batch_size, n_freq, time_steps] for each source,
    WHEN the MultiSourceLoss processes these batched inputs,
    THEN it should complete without error and return a non-negative scalar tensor.

    Preconditions:
        - 'outputs' and 'targets' are generated using random batched tensors.
    Postconditions:
        - Loss is a torch.Tensor and its value is non-negative.
    """
    outputs = generate_batched_tensors(batch_size=batch_size)
    targets = generate_batched_tensors(batch_size=batch_size)
    loss_fn = MultiSourceLoss(weights=[1.0] * NUM_SOURCES, distance=distance)
    loss = loss_fn(outputs, targets)
    assert isinstance(loss, torch.Tensor), "Loss must be a torch tensor."
    assert loss.item() >= 0, "Loss must be non-negative."


def test_loss_is_positive():
    """
    GIVEN two different sets of unbatched inputs,
    WHEN the MultiSourceLoss computes the loss,
    THEN the loss value should be positive.

    Preconditions:
        - 'outputs' and 'targets' are generated using different random tensors.
    Postconditions:
        - Loss value is greater than 0.
    """
    outputs = generate_unbatched_tensors()
    targets = generate_unbatched_tensors()
    loss_fn = MultiSourceLoss(weights=[1.0] * NUM_SOURCES, distance="l1")
    loss = loss_fn(outputs, targets)
    assert loss.item() > 0, "Loss should be positive for different inputs."


@pytest.mark.parametrize("distance", ["l1", "l2"])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_loss_is_zero_when_equal(distance: str, batch_size: int):
    """
    GIVEN identical batched inputs for outputs and targets,
    WHEN the MultiSourceLoss computes the loss,
    THEN the loss value should be zero.

    Preconditions:
        - 'outputs' are generated and 'targets' are an exact clone of 'outputs'.
    Postconditions:
        - Loss is zero within floating point tolerance.
    """
    outputs = generate_batched_tensors(batch_size=batch_size)
    targets = [output.clone() for output in outputs]
    loss_fn = MultiSourceLoss(weights=[1.0] * NUM_SOURCES, distance=distance)
    loss = loss_fn(outputs, targets)
    assert torch.isclose(
        loss, torch.tensor(0.0), atol=1e-6
    ), "Loss should be zero when outputs equal targets."
