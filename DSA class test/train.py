"""Command-line training script for custom-style GPT-2 fine-tuning."""

import os
import argparse
import sys
from src.dataset import load_dataset_from_file, prepare_datasets
from src.model import load_gpt2_model_and_tokenizer
from src.trainer import train_gpt2


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on custom style text datasets.")
    parser.add_argument("--data", type=str, default="data/campus_announcements.txt", help="Path to text dataset.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence token length.")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of transformer layers to freeze (0-12).")
    parser.add_argument("--use_lora", action="store_true", help="Apply Low-Rank Adaptation (LoRA).")
    parser.add_argument("--output_dir", type=str, default="checkpoints/campus_announcements_gpt2", help="Directory to save model checkpoint.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of samples for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    print(f"\n[Step 1/3] Loading GPT-2 Base Architecture...")
    model, tokenizer, meta = load_gpt2_model_and_tokenizer(
        model_name_or_path="gpt2",
        freeze_early_layers=args.freeze_layers,
        use_lora=args.use_lora,
    )
    print(f"  Device: {meta['device']} | Total Parameters: {meta['total_parameters']:,} | Trainable: {meta['trainable_parameters']:,} ({meta['trainable_percentage']}%)")

    print(f"\n[Step 2/3] Loading & Preprocessing Dataset: '{args.data}'...")
    raw_text = load_dataset_from_file(args.data)
    train_ds, val_ds, stats = prepare_datasets(
        raw_text,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split,
        seed=args.seed
    )
    print(f"  Total Samples: {stats['num_total_samples']} | Train: {stats['num_train_samples']} | Val: {stats['num_val_samples']}")
    print(f"  Total Tokens: {stats['total_tokens']:,} | Avg Tokens/Sample: {stats['avg_tokens_per_sample']} | Max: {stats['max_tokens_sample']}")

    print(f"\n[Step 3/3] Commencing Fine-Tuning Loop...")
    history = train_gpt2(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    print(f"\nProcess complete. Fine-tuned model checkpoint saved at '{args.output_dir}'.")


if __name__ == "__main__":
    main()
