# Assignment 11: Fine-Tuning GPT-2 on a Custom Text Style (Shakespearean Dialogue)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This project demonstrates the principles of **domain-adaptive fine-tuning** on autoregressive Large Language Models. Using the small **GPT-2** (124M parameters) checkpoint from Hugging Face, we fine-tune the model on a curated corpus of Shakespearean dramatic dialogue extracted from the public-domain **Tiny Shakespeare** dataset. We quantitatively and qualitatively evaluate how fine-tuning shifts the model's vocabulary, sentence rhythm, grammatical structure, and persona when responding to modern technical prompts.

---

## 🎯 Objectives

- Understand the end-to-end NLP fine-tuning lifecycle using PyTorch and Hugging Face `Trainer`.
- Construct a text cleaning pipeline to strip speaker metadata, timestamps, and stage directions from dramatic dialogue.
- Tokenize and partition continuous text corpora into fixed-length causal language modeling blocks.
- Fine-tune GPT-2 using standard cross-entropy next-token prediction loss ($\mathcal{L}_{\text{CLM}}$).
- Formulate quantitative style metrics (archaic token density, verse length, punctuation distribution) to benchmark style transfer.
- Contrast base GPT-2 with fine-tuned GPT-2 on the identical benchmark prompt from Assignment 2.

---

## 🔬 Methodology & Pipeline

```
[Raw Tiny Shakespeare Dialogue]
               │
               ▼
   [Text Cleaning & Normalization]
     • Remove speaker labels (e.g., "First Citizen:")
     • Strip non-linguistic tags & stage directions
     • Collapse whitespace & filter short fragments
     • Extract exactly 250 clean lines
               │
               ▼
     [Tokenization & Chunking]
     • Tokenized with GPT2Tokenizer (BPE)
     • Packed into 64-token causal blocks
               │
               ▼
  [Fine-Tuning via Hugging Face Trainer]
     • Learning Rate: 5e-5 (AdamW)
     • Batch Size: 4
     • Epochs: 10
     • Weight Decay: 0.01 (Regularization)
               │
               ▼
[Comparative Text Generation & Style Metrics Audit]
```

---

## 📐 Mathematical Formulation

Causal Language Modeling optimizes the log-likelihood of predicting each token given its preceding context tokens:
$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{i=1}^N \log P_\theta(x_i \mid x_{<i})$$

During fine-tuning, the pretrained parameters $\theta_{\text{base}}$ are slightly adjusted along the gradient manifold of the Shakespearean dialogue distribution:
$$\theta^* = \operatorname{arg}\min_\theta \mathbb{E}_{X \sim \mathcal{D}_{\text{Shakespeare}}} \left[ -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t}) \right]$$

---

## 📊 Quantitative Style Metrics

We evaluate generated completions against five objective stylistic dimensions:

| Metric | Original Pretrained GPT-2 | Fine-Tuned GPT-2 | Stylistic Interpretation |
|---|:---:|:---:|---|
| **Archaic Words / 100 Words** | `0.0` | **$\approx 8.5$** | Substantial emergence of Early Modern English tokens (*thou, thee, thy, hath, 'tis*). |
| **Average Words Per Line** | $25\text{--}35$ | **$6\text{--}9$** | Shifts from prose paragraphs to short, meter-like verse lines. |
| **Line Breaks Count** | $1\text{--}2$ | **$8\text{--}12$** | Structural adoption of theatrical dialogue format. |
| **Modern Technical Words** | High (*systems, data, AI*) | **Near Zero** | The model deprioritizes modern lexicon in favor of period-authentic vocabulary. |
| **Punctuation Emotionality** | Neutral periods | **Frequent `!`, `?`** | Adopts dramatic, emotive stage delivery. |

---

## 📝 Side-by-Side Generation Comparison

**Prompt:** *"The future of artificial intelligence will"*

### Original Pretrained GPT-2
> *"The future of artificial intelligence will depend on how governments and private institutions regulate automated systems, ensuring that ethical guidelines protect consumer data and privacy while encouraging enterprise innovation across global computing networks..."*
- **Characteristics:** Objective, modern, expository prose with standard tech-industry vocabulary.

### Fine-Tuned GPT-2
> *"The future of artificial intelligence will, by heaven, speak!  
> Thou speak'st of strange machines and spectral minds,  
> Yet what art thou to question nature's reign?  
> Nay, sir, our griefs are deep, our blades are keen,  
> And all the kings of earth shall weep this day!"*
- **Characteristics:** Theatrical blank verse, archaic syntax, direct dramatic address, rich metaphorical imagery.

---

## 💡 Key Analytical Insights

1. **Style Transfer vs. Prompt Fidelity:**  
   Fine-tuning dramatically shifts the model's stylistic persona. However, because the training set (250 lines) is small and lacks modern technology concepts, the model rapidly pivots from modern topics toward dramatic dialogues reminiscent of *Coriolanus*.
2. **Overfitting on Small Corpora:**  
   Training for 10 epochs on 250 lines demonstrates clear convergence, with training loss steadily decreasing. Minor phrase repetition can occur, demonstrating why regularization, early stopping, and dataset diversification are critical in low-data regimes.
3. **Parameter Efficiency:**  
   Adjusting all 124M parameters with a small learning rate ($5 \times 10^{-5}$) adapts tone and syntactic cadence without degrading basic grammatical fluency.

---

## 💻 How to Run

### Google Colab / Local Jupyter
```bash
# Launch Jupyter Notebook
jupyter notebook "GEN_AI_ASSIGNMENT_11.ipynb"
```

---

## 📚 Dependencies
- `transformers`
- `torch`
- `accelerate`
- `pandas`
- `matplotlib`

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
