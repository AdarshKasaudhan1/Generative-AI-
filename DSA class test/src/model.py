"""GPT-2 Model loading, tokenizer configuration, and parameter customization."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def get_default_device() -> torch.device:
    """Return cuda if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_gpt2_model_and_tokenizer(
    model_name_or_path: str = "gpt2",
    device: Optional[torch.device] = None,
    freeze_early_layers: int = 0,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Load GPT-2 model and tokenizer with proper padding configurations and optional freezing/LoRA.

    Args:
        model_name_or_path: HuggingFace model identifier or local checkpoint folder path.
        device: PyTorch device to load the model on (defaults to cuda if available).
        freeze_early_layers: Number of transformer blocks (0 to 12) to freeze.
        use_lora: Whether to apply Low-Rank Adaptation (LoRA) for parameter-efficient tuning.
        lora_r: Rank of LoRA update matrices.
        lora_alpha: Alpha scaling factor for LoRA.
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        Tuple of (model, tokenizer, metadata_dict).
    """
    if device is None:
        device = get_default_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_applied = False
    if freeze_early_layers > 0 and not use_lora:
        freeze_early_layers = min(12, max(0, freeze_early_layers))
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "wte"):
                for p in model.transformer.wte.parameters():
                    p.requires_grad = False
            if hasattr(model.transformer, "wpe"):
                for p in model.transformer.wpe.parameters():
                    p.requires_grad = False
            if hasattr(model.transformer, "h"):
                for block in model.transformer.h[:freeze_early_layers]:
                    for p in block.parameters():
                        p.requires_grad = False

    elif use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft package is required for LoRA fine-tuning. Install via pip install peft.")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        lora_applied = True

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    meta = {
        "model_name_or_path": model_name_or_path,
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": round(trainable_pct, 2),
        "frozen_layers": "LoRA mode" if lora_applied else freeze_early_layers,
        "lora_enabled": lora_applied,
        "vocab_size": len(tokenizer)
    }

    return model, tokenizer, meta


def get_parameter_summary(model: nn.Module) -> Dict[str, Any]:
    """Return dictionary of total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = round((trainable / total) * 100, 2) if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "percentage": percentage
    }
