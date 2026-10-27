# Custom-Style GPT-2 Fine-Tuning Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-brightgreen.svg)](https://streamlit.io/)

An interactive, production-ready environment for domain-adaptive causal language modeling with **GPT-2** (124M parameters). Includes dataset preparation tools, an optimized PyTorch training loop with AdamW, learning rate scheduling, real-time loss tracking, side-by-side comparative text generation, and a Streamlit dashboard.

---

## 📌 Project Overview & Features

1. **Frontend (Streamlit Dashboard `app.py`)**:
   - **Dataset Preprocessing & Tokenizer Inspection**: Upload custom files (`.txt`, `.csv`, `.pdf`) or use built-in datasets (*Campus Announcements*, *Cyberpunk Catalog*, *Micro Stories*). Inspect BPE token IDs and lengths.
   - **Fine-Tuning Lab**: Real-time hyperparameter configuration (epochs, batch size, learning rate, sequence length, early block freezing, LoRA). Live progress bar and step loss chart.
   - **Side-by-Side Generation Audit**: Direct before-and-after comparison between base pretrained GPT-2 and fine-tuned checkpoints under identical prompts and sampling parameters.
   - **Loss Monitoring & Checkpoint Viewer**: View step loss, epoch loss, validation perplexity, and metadata for every checkpoint.
   - **Architecture & Mathematical Formulation**: Causal language modeling loss formulation ($\mathcal{L}_{\text{CLM}}$), perplexity metrics, and token masking.

2. **Backend Engine (`src/`)**:
   - `src/dataset.py`: Multi-format ingestion, regex-based document chunking, PyTorch `CausalLMDataset` with $-100$ label masking for padding tokens.
   - `src/model.py`: Automatic device routing, GPT-2 model and tokenizer initialization, selective transformer block freezing, and optional PEFT/LoRA integration.
   - `src/trainer.py`: Full PyTorch training loop with AdamW, linear warmup scheduling, gradient clipping, evaluation, and JSON history exports.
   - `src/generator.py`: Autoregressive generation with top-k, top-p, temperature, and quantitative style-shift analysis (vocabulary divergence, structural headers).

3. **Pre-Trained & Ready Checkpoints (`checkpoints/`)**:
   - `campus_announcements_gpt2`: Fine-tuned on university administration notices with dates, departments, and action items.
   - `cyberpunk_products_gpt2`: Fine-tuned on futuristic sci-fi product catalogs with tech specs and pricing.
   - `micro_stories_gpt2`: Fine-tuned on atmospheric flash fiction.

---

## 🚀 Quickstart Guide

### 1. Launch the Streamlit Web UI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run CLI Side-by-Side Comparison
```bash
python generate.py --checkpoint checkpoints/campus_announcements_gpt2 --prompt "[CAMPUS NOTICE]\nDATE: November 15, 2026\nDEPARTMENT: Department of Artificial Intelligence\nSUBJECT:"
```

### 3. Fine-Tune on a Custom Dataset via CLI
```bash
python train.py --data data/campus_announcements.txt --epochs 5 --batch_size 2 --lr 8e-5 --output_dir checkpoints/my_custom_run
```

### 4. Run Automated End-to-End Pipeline Verification
```bash
python test_pipeline.py
```

---

## 📊 Pre-Trained Benchmark Results

| Model Checkpoint | Training Corpus | Epochs | Training Time (CPU) | Final Train Loss | Final Perplexity | Style Shift |
|---|---|:---:|:---:|:---:|:---:|---|
| **Base GPT-2** | WebText | Pretrained | — | — | — | Generic prose paragraphs |
| **Campus Announcements** | Structured notices | 5 | ~24s | **2.948** | **17.16** | Adopts `[CAMPUS NOTICE]`, `DATE:`, `DEPARTMENT:` headers |
| **Cyberpunk Catalog** | Sci-Fi tech items | 5 | ~22s | **2.884** | **16.82** | Generates `[PRODUCT:]`, `SPECS:`, `Eurodollars` |
| **Micro Stories** | Flash fiction | 5 | ~25s | **3.012** | **18.35** | Adopts evocative, literary prose rhythm |

---

## 📐 Mathematical Formulation

Causal Language Modeling minimizes the negative log-likelihood of each token given its preceding context:
$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{i=1}^N \log P_\theta(x_i \mid x_{<i})$$

Perplexity quantifies how well the probability distribution predicts the sample:
$$\text{PPL} = \exp\left( \mathcal{L}_{\text{CLM}} \right)$$

---

## 📁 Repository Directory Structure

```
├── app.py                     # Streamlit interactive web dashboard
├── train.py                   # Command-line training interface
├── generate.py                # Command-line text generation & comparison
├── test_pipeline.py           # Automated end-to-end integration test
├── src/
│   ├── __init__.py            # Package initialization
│   ├── dataset.py             # Preprocessing, PDF extraction, CausalLMDataset
│   ├── model.py               # GPT-2 loader, layer freezing, LoRA support
│   ├── trainer.py             # PyTorch training loop, loss tracking, callbacks
│   └── generator.py           # Text generation, style shift audit, comparison
├── data/
│   ├── campus_announcements.txt
│   ├── cyberpunk_products.txt
│   └── micro_stories.txt
├── checkpoints/
│   ├── campus_announcements_gpt2/
│   ├── cyberpunk_products_gpt2/
│   └── micro_stories_gpt2/
└── README.md
```
