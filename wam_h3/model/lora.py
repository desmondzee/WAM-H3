from peft import LoraConfig, inject_adapter_in_model

TARGET = r".*blocks\.\d+\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2)|adaln_proj\.linear)$"
HEADS = ("action_in.", "proprio_in.", "final_layer.action_out.")


def apply_lora(dit, r, alpha=None):
    dit.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=r if alpha is None else alpha, target_modules=TARGET), dit)
    for n, p in dit.named_parameters():
        if n.startswith(HEADS):
            p.requires_grad_(True)
    return dit
