"""End-to-End Pipeline Automated Verification Test."""

import os
import shutil
import torch
from src.dataset import load_dataset_from_file, prepare_datasets
from src.model import load_gpt2_model_and_tokenizer
from src.trainer import train_gpt2
from src.generator import compare_models, generate_text


def test_full_pipeline():
    print("=" * 60)
    print("RUNNING AUTOMATED GPT-2 PIPELINE INTEGRATION TEST")
    print("=" * 60)

    test_dir = "checkpoints/test_pipeline_run"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("\n[Step 1] Testing Model and Tokenizer Loader...")
    model, tokenizer, meta = load_gpt2_model_and_tokenizer("gpt2")
    assert model is not None, "Failed to load model"
    assert tokenizer is not None, "Failed to load tokenizer"
    print(f"  Passed! Total params: {meta['total_parameters']:,} | Device: {meta['device']}")

    print("\n[Step 2] Testing Dataset Preprocessing & Tokenization...")
    sample_text = (
        "[CAMPUS NOTICE]\n"
        "DEPARTMENT: Examination Cell\n"
        "SUBJECT: Lab Test Schedule\n"
        "DETAILS: The Generative AI Lab test will take place on Tuesday.\n"
        "ACTION REQUIRED: Bring your laptops.\n\n"
        "[CAMPUS NOTICE]\n"
        "DEPARTMENT: Library Services\n"
        "SUBJECT: Book Returns\n"
        "DETAILS: Overdue library books must be returned by Friday afternoon.\n"
        "ACTION REQUIRED: Pay any dues online."
    )
    train_ds, val_ds, stats = prepare_datasets(sample_text, tokenizer, max_length=64, val_split=0.5, seed=42)
    assert len(train_ds) >= 1, "Train dataset should contain at least 1 sample"
    batch = train_ds[0]
    assert "input_ids" in batch and "labels" in batch, "Missing required keys in dataset item"
    print(f"  Passed! Preprocessed {stats['num_total_samples']} samples, avg tokens: {stats['avg_tokens_per_sample']}.")

    print("\n[Step 3] Testing 1-Epoch Fast Mini Training Run...")
    history = train_gpt2(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=1,
        batch_size=1,
        lr=5e-5,
        output_dir=test_dir
    )
    assert os.path.exists(os.path.join(test_dir, "model.safetensors")), "Saved model weights missing"
    assert os.path.exists(os.path.join(test_dir, "training_history.json")), "Training history missing"
    print(f"  Passed! Checkpoint saved. Final Loss: {history['final_train_loss']}")

    print("\n[Step 4] Testing Text Generation and Model Comparison...")
    ft_model, ft_tokenizer, _ = load_gpt2_model_and_tokenizer(test_dir)
    res = compare_models(
        base_model=model,
        base_tokenizer=tokenizer,
        ft_model=ft_model,
        ft_tokenizer=ft_tokenizer,
        prompt="[CAMPUS NOTICE]\nDEPARTMENT: Examination Cell\nSUBJECT:",
        max_new_tokens=20,
        seed=42
    )
    assert "base" in res and "finetuned" in res, "Comparison output missing expected keys"
    print(f"  Passed! Dual generation successful.")
    print(f"  Fine-tuned output preview:\n{res['finetuned']['full_text']}")

    # Clean up test output
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! The pipeline is fully functional.")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
