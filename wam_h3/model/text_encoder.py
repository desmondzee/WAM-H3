from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel

RETAINED_LAYERS = 50


def presentation_t2va(tokenizer, prompt):
    return torch.tensor(tokenizer(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long)


def collate_instructions(embs, text_len):
    ctx = torch.zeros(len(embs), text_len, embs[0].shape[-1], dtype=embs[0].dtype)
    valid = torch.zeros(len(embs), text_len, dtype=torch.bool)
    for i, e in enumerate(embs):
        if e.shape[0] > text_len:
            raise ValueError(f"instruction has {e.shape[0]} tokens, text_len is {text_len}")
        ctx[i, text_len - e.shape[0]:] = e
        valid[i, text_len - e.shape[0]:] = True
    return ctx, valid


class TextEncoder(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.model.language_model.norm = nn.Identity()

    @staticmethod
    def load_config(root):
        cfg = Qwen3VLConfig.from_pretrained(Path(root) / "text_encoder")
        cfg.text_config.num_hidden_layers = RETAINED_LAYERS
        return cfg

    @classmethod
    def from_pretrained(cls, root, device="cpu", dtype=torch.bfloat16):
        root = Path(root)
        model = Qwen3VLModel.from_pretrained(root / "text_encoder", config=cls.load_config(root), dtype=dtype, device_map=device)
        return cls(model.eval(), AutoTokenizer.from_pretrained(root / "tokenizer"))

    @torch.no_grad()
    def encode(self, prompts):
        dev = next(self.model.parameters()).device
        out = []
        for p in prompts:
            ids = presentation_t2va(self.tokenizer, p)[None].to(dev)
            out.append(self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0].cpu())
        return out
