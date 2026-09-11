import re

from peft import LoraConfig, inject_adapter_in_model

TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear")
HEADS = ("action_in.", "proprio_in.", "final_layer.action_out.")


def apply_lora(dit, r, alpha=None, targets=TARGETS, dropout=0.0):
    dit.requires_grad_(False)
    pattern = r".*blocks\.\d+\.(" + "|".join(re.escape(t) for t in targets) + ")$"
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=r if alpha is None else alpha, lora_dropout=dropout, target_modules=pattern), dit)
    for n, p in dit.named_parameters():
        if n.startswith(HEADS):
            p.requires_grad_(True)
    return dit
