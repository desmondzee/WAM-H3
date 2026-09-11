from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from wam_h3.model.text_encoder import collate_instructions, presentation_t2va

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(ROOT / "FL2VA/tokenizer")


def test_presentation_is_plain_tokens_without_special_tokens(tok):
    ids = presentation_t2va(tok, "pick up the black bowl")
    assert ids.dtype == torch.long and ids.ndim == 1
    assert tok.decode(ids) == "pick up the black bowl"
    assert tok.bos_token_id not in ids.tolist() and tok.eos_token_id not in ids.tolist()


def test_collate_pads_and_masks():
    embs = [torch.ones(3, 8), torch.full((5, 8), 2.0)]
    ctx, valid = collate_instructions(embs, text_len=6)
    assert ctx.shape == (2, 6, 8) and valid.shape == (2, 6)
    assert valid.tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]
    assert (ctx[0, :3] == 1).all() and (ctx[0, 3:] == 0).all()
    assert (ctx[1, :5] == 2).all()


def test_collate_rejects_overlong():
    with pytest.raises(ValueError):
        collate_instructions([torch.ones(7, 8)], text_len=6)
