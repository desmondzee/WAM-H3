import pytest
import torch
from accelerate.state import AcceleratorState

from wam_h3.model.config import WAMH3Config


@pytest.fixture
def cfg():
    return WAMH3Config.tiny()


@pytest.fixture(autouse=True)
def _seed():
    AcceleratorState._reset_state(reset_partial_state=True)
    torch.manual_seed(0)
