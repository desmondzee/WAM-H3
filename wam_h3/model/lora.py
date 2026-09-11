import re

from peft import LoraConfig, inject_adapter_in_model

TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear")
HEADS = ("action_in.", "proprio_in.", "final_layer.action_out.")


def apply_lora(dit, r, alpha=None, targets=TARGETS, dropout=0.0, adaln_r=None):
    dit.requires_grad_(False)
    alpha = r if alpha is None else alpha
    pattern = r".*blocks\.\d+\.(" + "|".join(re.escape(t) for t in targets) + ")$"
    rank, alphas = {}, {}
    if adaln_r and adaln_r != r:
        rank, alphas = {"adaln_proj.linear": adaln_r}, {"adaln_proj.linear": alpha * adaln_r / r}
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=pattern,
                                       rank_pattern=rank, alpha_pattern=alphas), dit)
    for n, p in dit.named_parameters():
        if n.startswith(HEADS):
            p.requires_grad_(True)
    return dit
