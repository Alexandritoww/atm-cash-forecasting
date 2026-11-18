import torch

from src.models.multitask_network import MultiTaskATMNetwork


def test_multitask_output_shape():
    model = MultiTaskATMNetwork(input_dim=6)

    inputs = torch.randn(8, 6)
    outputs = model(inputs)

    assert outputs.shape == (8, 4)


def test_multitask_invalid_input_dimension():
    import pytest

    with pytest.raises(ValueError):
        MultiTaskATMNetwork(input_dim=0)