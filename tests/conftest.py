import torch
import pytest


@pytest.fixture(autouse=True)
def reproducible_torch_seed():
    torch.manual_seed(0)

