"""LoRA and full fine-tuning setup for the transformer."""

import torch


def setup_lora(transformer, cfg):
    """
    Apply LoRA adapters to the transformer.

    Returns the peft-wrapped model (only LoRA params are trainable).
    """
    from peft import LoraConfig, get_peft_model

    lora_rank = cfg.get("lora_rank", 128)
    lora_alpha = cfg.get("lora_alpha", 128)

    target_modules = set()
    for name, mod in transformer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            target_modules.add(name.split(".")[-1])

    attn_ffn_keywords = ["to_q", "to_k", "to_v", "to_out", "proj", "ff", "net"]
    target_modules = sorted([
        m for m in target_modules
        if any(kw in m for kw in attn_ffn_keywords)
    ])

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )

    transformer = get_peft_model(transformer, lora_config)
    return transformer


def setup_full_finetune(transformer):
    """Enable gradients on all transformer parameters for full fine-tuning."""
    transformer.requires_grad_(True)
    return transformer
