"""Text generation and side-by-side comparison engine for Base vs Fine-Tuned GPT-2."""

import time
from typing import Dict, Any, Optional, List
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    min_new_tokens: int = 10,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    do_sample: bool = True,
    seed: Optional[int] = 42,
    stop_at_eos: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Generate text completion from a prompt using GPT-2 causal language model.

    Args:
        model: Loaded GPT-2 model.
        tokenizer: Corresponding tokenizer.
        prompt: Initial prompt text.
        max_new_tokens: Maximum additional tokens to generate.
        min_new_tokens: Minimum additional tokens before allowing EOS.
        temperature: Sampling randomness (higher = more creative/diverse).
        top_k: Top-k filtering threshold.
        top_p: Nucleus sampling probability cutoff.
        repetition_penalty: Multiplicative penalty for repeated tokens.
        do_sample: Whether to sample or use greedy decoding.
        seed: Random seed for reproducibility.
        stop_at_eos: Whether to terminate generation upon encountering EOS token.
        device: Device to place tensors on.

    Returns:
        Dictionary with full_text, continuation, latency, and throughput stats.
    """
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    eos_token_id = tokenizer.eos_token_id if stop_at_eos else None

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens if stop_at_eos else 0,
            do_sample=do_sample,
            temperature=max(0.01, float(temperature)),
            top_k=max(1, int(top_k)),
            top_p=min(1.0, max(0.01, float(top_p))),
            repetition_penalty=max(1.0, float(repetition_penalty)),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_id
        )

    latency_sec = time.time() - t0
    generated_tokens_count = len(output_ids[0]) - len(input_ids[0])
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    continuation = full_text[len(prompt):].strip()

    return {
        "prompt": prompt,
        "full_text": full_text,
        "continuation": continuation,
        "tokens_generated": generated_tokens_count,
        "latency_sec": round(latency_sec, 3),
        "tokens_per_sec": round(generated_tokens_count / max(0.001, latency_sec), 1),
    }


def analyze_style_shift(base_text: str, finetuned_text: str) -> Dict[str, Any]:
    """Perform vocabulary and structural comparison between base and fine-tuned outputs."""
    base_words = set(w.lower() for w in base_text.split() if len(w) > 2)
    ft_words = set(w.lower() for w in finetuned_text.split() if len(w) > 2)

    shared_words = base_words.intersection(ft_words)
    ft_unique_words = ft_words - base_words

    has_headers = any(marker in finetuned_text for marker in (
        "[", "]", "DATE:", "DEPARTMENT:", "SUBJECT:", "SPECS:", "PRICE:"
    ))

    return {
        "base_word_count": len(base_text.split()),
        "finetuned_word_count": len(finetuned_text.split()),
        "shared_vocabulary_count": len(shared_words),
        "finetuned_unique_words": sorted(list(ft_unique_words))[:15],
        "adopted_structured_format": has_headers,
    }


def compare_models(
    base_model: PreTrainedModel,
    base_tokenizer: PreTrainedTokenizer,
    ft_model: PreTrainedModel,
    ft_tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    min_new_tokens: int = 10,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    stop_at_eos: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """Executes side-by-side inference with identical parameters on Base and Fine-Tuned GPT-2."""
    base_result = generate_text(
        base_model,
        base_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_at_eos=stop_at_eos,
        seed=seed
    )

    ft_result = generate_text(
        ft_model,
        ft_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_at_eos=stop_at_eos,
        seed=seed
    )

    style_shift = analyze_style_shift(base_result["full_text"], ft_result["full_text"])

    return {
        "prompt": prompt,
        "base": base_result,
        "finetuned": ft_result,
        "style_shift": style_shift,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        }
    }
