"""Command-line text generation and side-by-side comparison script."""

import os
import argparse
from src.model import load_gpt2_model_and_tokenizer
from src.generator import generate_text, compare_models


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Generate text using base or fine-tuned GPT-2 models.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/campus_announcements_gpt2", help="Path to fine-tuned checkpoint directory.")
    parser.add_argument("--prompt", type=str, default="[CAMPUS NOTICE]\nDATE: November 15, 2026\nDEPARTMENT: Department of Artificial Intelligence\nSUBJECT:", help="Input prompt text.")
    parser.add_argument("--max_tokens", type=int, default=80, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling cutoff.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K token selection.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--no_compare", action="store_true", help="Only run generation on the fine-tuned checkpoint.")

    args = parser.parse_args()

    if args.no_compare:
        print_banner("SINGLE MODEL TEXT GENERATION")
        print(f"Loading checkpoint: '{args.checkpoint}'...")
        model, tokenizer, _ = load_gpt2_model_and_tokenizer(args.checkpoint)
        res = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed
        )
        print(f"\n[Prompt]:\n{res['prompt']}")
        print(f"\n[Generated Output]:\n{res['full_text']}")
        print(f"\n[Stats]: Generated {res['tokens_generated']} tokens in {res['latency_sec']}s ({res['tokens_per_sec']} tok/sec)")
        return

    print_banner("SIDE-BY-SIDE GENERATION COMPARISON (BASE VS FINE-TUNED)")
    print("\n[Loading Models] Loading Base GPT-2 and Fine-Tuned checkpoint...")
    base_model, base_tokenizer, _ = load_gpt2_model_and_tokenizer("gpt2")
    ft_model, ft_tokenizer, _ = load_gpt2_model_and_tokenizer(args.checkpoint)

    comp = compare_models(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        ft_model=ft_model,
        ft_tokenizer=ft_tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed
    )

    print(f"\n>>> PROMPT INPUT <<<\n{comp['prompt']}")

    print("\n" + "-" * 30 + " BASE PRETRAINED GPT-2 " + "-" * 30)
    print(comp["base"]["full_text"])
    print(f"\n[Latency: {comp['base']['latency_sec']}s | Speed: {comp['base']['tokens_per_sec']} tok/s]")

    print("\n" + "-" * 30 + " FINE-TUNED GPT-2 " + "-" * 30)
    print(comp["finetuned"]["full_text"])
    print(f"\n[Latency: {comp['finetuned']['latency_sec']}s | Speed: {comp['finetuned']['tokens_per_sec']} tok/s]")

    shift = comp["style_shift"]
    print("\n" + "=" * 30 + " STYLE SHIFT AUDIT " + "=" * 30)
    print(f"Base Word Count: {shift['base_word_count']} | Fine-Tuned Word Count: {shift['finetuned_word_count']}")
    print(f"Shared Vocabulary Count: {shift['shared_vocabulary_count']}")
    print(f"Adopted Structured Domain Format: {'YES' if shift['adopted_structured_format'] else 'NO'}")
    print(f"Key Style-Shift Unique Tokens: {', '.join(shift['finetuned_unique_words'][:10])}")


if __name__ == "__main__":
    main()
