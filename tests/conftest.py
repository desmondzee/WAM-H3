import pytest
import torch

from wam_h3.model.config import WAMH3Config


@pytest.fixture
def cfg():
    return WAMH3Config.tiny()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
