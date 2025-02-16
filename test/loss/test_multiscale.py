import torch
import pytest
from src.utils.training.loss import MultiScaleLoss
from test.utils import generate_unbatched_tensors, generate_batched_tensors, NUM_SOURCES


def test_unbatched_input_processing():
    """
    GIVEN unbatched inputs with shape [n_freq, time_steps] for each source,
    WHEN the MultiScaleLoss processes these inputs,
    THEN it should complete without error and return a non-negative scalar tensor.

    Preconditions:
        - 'outputs' and 'targets' are generated using random unbatched tensors.
    Postconditions:
        - Loss is a torch.Tensor and its value is non-negative.
    """
    outputs = generate_unbatched_tensors()
    targets = generate_unbatched_tensors()
    loss_fn = MultiScaleLoss(weights=[1.0] * NUM_SOURCES, distance="l1")
    loss = loss_fn(outputs, targets)
    assert isinstance(loss, torch.Tensor), "Loss must be a torch tensor."
    assert loss.item() >= 0, "Loss must be non-negative."


@pytest.mark.parametrize("distance", ["l1", "l2"])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_batched_input_processing(distance: str, batch_size: int):
    """
    GIVEN batched inputs with shape [batch_size, n_freq, time_steps] for each source,
    WHEN the MultiScaleLoss processes these batched inputs,
    THEN it should complete without error and return a non-negative scalar tensor.

    Preconditions:
        - 'outputs' and 'targets' are generated using random batched tensors.
    Postconditions:
        - Loss is a torch.Tensor and its value is non-negative.
    """
    outputs = generate_batched_tensors(batch_size=batch_size)
    targets = generate_batched_tensors(batch_size=batch_size)
    loss_fn = MultiScaleLoss(weights=[1.0] * NUM_SOURCES, distance=distance)
    loss = loss_fn(outputs, targets)
    assert isinstance(loss, torch.Tensor), "Loss must be a torch tensor."
    assert loss.item() >= 0, "Loss must be non-negative."


def test_loss_is_positive():
    """
    GIVEN two different sets of unbatched inputs,
    WHEN the MultiScaleLoss computes the loss,
    THEN the loss value should be positive.

    Preconditions:
        - 'outputs' and 'targets' are generated using different random tensors.
    Postconditions:
        - Loss value is greater than 0.
    """
    outputs = generate_unbatched_tensors()
    targets = generate_unbatched_tensors()
    loss_fn = MultiScaleLoss(weights=[1.0] * NUM_SOURCES, distance="l1")
    loss = loss_fn(outputs, targets)
    assert loss.item() > 0, "Loss should be positive for different inputs."


@pytest.mark.parametrize("distance", ["l1", "l2"])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_loss_is_zero_when_equal(distance: str, batch_size: int):
    """
    GIVEN identical batched inputs for outputs and targets,
    WHEN the MultiScaleLoss computes the loss,
    THEN the loss value should be zero.

    Preconditions:
        - 'outputs' are generated and 'targets' are an exact clone of 'outputs'.
    Postconditions:
        - Loss is zero within floating point tolerance.
    """
    outputs = generate_batched_tensors(batch_size=batch_size)
    targets = [output.clone() for output in outputs]
    loss_fn = MultiScaleLoss(weights=[1.0] * NUM_SOURCES, distance=distance)
    loss = loss_fn(outputs, targets)
    assert torch.isclose(
        loss, torch.tensor(0.0), atol=1e-6
    ), "Loss should be zero when outputs equal targets."


def test_invalid_distance_metric():
    """
    GIVEN an invalid distance metric,
    WHEN the MultiScaleLoss is instantiated,
    THEN it should raise a ValueError.

    Preconditions:
        - The distance parameter is set to an invalid value.
    Postconditions:
        - A ValueError is raised upon instantiation.
    """
    with pytest.raises(ValueError):
        MultiScaleLoss(weights=[1.0] * NUM_SOURCES, distance="invalid")
