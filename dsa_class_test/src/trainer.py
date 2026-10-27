"""PyTorch Training loop with loss tracking, gradient clipping, evaluation, and callbacks."""

import os
import time
import math
import json
from typing import Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Calculate validation cross-entropy loss and perplexity.

    Args:
        model: PyTorch model.
        val_loader: DataLoader for validation data.
        device: Active PyTorch computing device.

    Returns:
        Tuple of (average_validation_loss, perplexity).
    """
    model.eval()
    total_val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_val_loss += loss.item()
            val_steps += 1

    avg_val_loss = total_val_loss / max(1, val_steps)
    perplexity = round(math.exp(min(avg_val_loss, 20.0)), 2)
    return avg_val_loss, perplexity


def train_gpt2(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 8e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    output_dir: str = "checkpoints/custom_gpt2",
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """Executes causal LM training on GPT-2 with real-time tracking.

    Args:
        model: HuggingFace GPT-2 model.
        tokenizer: HuggingFace tokenizer.
        train_dataset: PyTorch Dataset for training.
        val_dataset: Optional PyTorch Dataset for validation.
        epochs: Number of training epochs.
        batch_size: Batch size per training step.
        lr: Peak learning rate for AdamW.
        weight_decay: Weight decay factor for non-bias parameters.
        warmup_ratio: Fraction of total steps used for linear warmup.
        max_grad_norm: Maximum norm for gradient clipping.
        gradient_accumulation_steps: Number of batches to accumulate gradients.
        output_dir: Directory to save the final model and tokenizer.
        device: Device to train on (defaults to cuda if available, else cpu).
        progress_callback: Callback function receiving step/epoch progress dicts.

    Returns:
        Dictionary containing comprehensive training history and metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    no_decay = ["bias", "LayerNorm.weight", "ln_1.weight", "ln_2.weight", "ln_f.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history: Dict[str, Any] = {
        "step_losses": [],
        "epoch_losses": [],
        "val_losses": [],
        "perplexities": [],
        "learning_rates": [],
        "timestamps": [],
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "device": str(device),
        },
    }

    start_time = time.time()
    global_step = 0
    optimizer.zero_grad()

    print(f"\n--- Starting GPT-2 Fine-Tuning ---")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Total Steps: {total_steps}")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset) if val_dataset else 0}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_in_epoch = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            batch_loss_val = loss.item() * gradient_accumulation_steps
            epoch_loss += batch_loss_val
            step_in_epoch += 1

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                curr_lr = scheduler.get_last_lr()[0]
                history["step_losses"].append(round(batch_loss_val, 4))
                history["learning_rates"].append(curr_lr)

                if progress_callback:
                    progress_callback({
                        "event": "step",
                        "step": global_step,
                        "total_steps": total_steps,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "loss": round(batch_loss_val, 4),
                        "lr": curr_lr,
                    })

        avg_epoch_loss = epoch_loss / max(1, step_in_epoch)
        history["epoch_losses"].append(round(avg_epoch_loss, 4))

        if val_loader:
            val_loss, ppl = evaluate(model, val_loader, device)
            history["val_losses"].append(round(val_loss, 4))
            history["perplexities"].append(round(ppl, 2))
            val_str = f" | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}"
        else:
            val_loss, ppl = None, None
            history["val_losses"].append(None)
            history["perplexities"].append(None)
            val_str = ""

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_epoch_loss:.4f}{val_str} | Elapsed: {elapsed:.1f}s")

        if progress_callback:
            progress_callback({
                "event": "epoch",
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_loss": round(avg_epoch_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "perplexity": round(ppl, 2) if ppl is not None else None,
                "elapsed_sec": round(elapsed, 1),
            })

    total_training_time = round(time.time() - start_time, 2)
    history["total_training_time_sec"] = total_training_time
    history["final_train_loss"] = history["epoch_losses"][-1] if history["epoch_losses"] else None
    history["final_perplexity"] = history["perplexities"][-1] if history["perplexities"] and history["perplexities"][-1] is not None else (
        round(math.exp(min(history["final_train_loss"], 20.0)), 2) if history["final_train_loss"] else None
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Saved to '{output_dir}'.")
    print(f"Total training time: {total_training_time}s | Final Train Loss: {history['final_train_loss']} | Final PPL: {history['final_perplexity']}")

    if progress_callback:
        progress_callback({
            "event": "completed",
            "output_dir": output_dir,
            "total_training_time_sec": total_training_time,
            "final_train_loss": history["final_train_loss"],
            "final_perplexity": history["final_perplexity"],
        })

    return history
