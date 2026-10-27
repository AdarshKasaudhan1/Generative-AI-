"""Dataset preprocessing and PyTorch Dataset implementation for GPT-2 fine-tuning."""

import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


def extract_text_from_pdf(pdf_source: Any) -> str:
    """Extract text from PDF using pypdfium2."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_source)
        text_parts = []
        for page in pdf:
            text_page = page.get_textpage()
            text = text_page.get_text_range()
            if text and text.strip():
                text_parts.append(text.strip())
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def clean_and_normalize_text(text: str) -> str:
    """Clean text while preserving structural markers."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_samples(raw_text: str, delimiter: Optional[str] = None) -> List[str]:
    """Split text into distinct samples/documents based on delimiter or double newlines."""
    cleaned = clean_and_normalize_text(raw_text)
    if not cleaned:
        return []

    if delimiter and delimiter in cleaned:
        raw_parts = cleaned.split(delimiter)
        samples = [p.strip() for p in raw_parts if len(p.strip()) > 10]
        return samples

    if "[" in cleaned and "\n\n[" in cleaned:
        parts = re.split(r"\n\n(?=\[)", cleaned)
        samples = [p.strip() for p in parts if len(p.strip()) > 10]
        return samples

    parts = cleaned.split("\n\n")
    samples = [p.strip() for p in parts if len(p.strip()) > 10]
    if not samples and len(cleaned) > 10:
        samples = [cleaned]
    return samples


class CausalLMDataset(Dataset):
    """PyTorch Dataset for Causal Language Modeling with GPT-2.

    Each sample is tokenized, enclosed with <|endoftext|>, and padded/truncated to max_length.
    Labels are cloned from input_ids, with pad tokens set to -100 so CrossEntropy ignores them.
    """

    def __init__(
        self,
        samples: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        add_eos_token: bool = True
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, torch.Tensor]] = []

        eos = tokenizer.eos_token or "<|endoftext|>"
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        for sample in samples:
            text = sample.strip()
            if not text:
                continue
            if add_eos_token and not text.endswith(eos):
                text = text + eos

            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            if pad_id is not None:
                labels[labels == pad_id] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def prepare_datasets(
    text_content: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[CausalLMDataset, Optional[CausalLMDataset], Dict[str, Any]]:
    """Splits text content into train and validation datasets and returns descriptive stats."""
    samples = split_text_into_samples(text_content)
    if not samples:
        raise ValueError("No valid training samples could be extracted from input text.")

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=rng).tolist()
    shuffled_samples = [samples[i] for i in indices]

    val_size = max(1, int(len(samples) * val_split)) if val_split > 0 and len(samples) > 3 else 0
    if val_size > 0:
        train_samples = shuffled_samples[val_size:]
        val_samples = shuffled_samples[:val_size]
    else:
        train_samples = shuffled_samples
        val_samples = []

    train_dataset = CausalLMDataset(train_samples, tokenizer, max_length=max_length)
    val_dataset = CausalLMDataset(val_samples, tokenizer, max_length=max_length) if val_samples else None

    token_lengths = [len(tokenizer.encode(s)) for s in samples]
    avg_tokens = sum(token_lengths) / max(1, len(token_lengths))
    max_tokens = max(token_lengths) if token_lengths else 0
    total_tokens = sum(token_lengths)

    stats = {
        "num_total_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": round(avg_tokens, 1),
        "max_tokens_sample": max_tokens,
        "max_seq_length": max_length,
        "raw_char_count": len(text_content),
        "sample_preview": samples[:3]
    }

    return train_dataset, val_dataset, stats


def load_dataset_from_file(file_path: str) -> str:
    """Read dataset from text file with fallback encodings."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
