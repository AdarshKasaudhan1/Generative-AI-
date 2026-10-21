# Assignment 2: Exploring Text Generation Using GPT-2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

## 📌 Executive Summary

This study explores auto-regressive language generation using OpenAI's **GPT-2** (124M parameter checkpoint) via the **Hugging Face Transformers** library. We systematically analyze how sampling heuristics—specifically **Temperature**, **Top-K filtering**, **Top-P (Nucleus) sampling**, and **Repetition Penalty**—govern the trade-off between semantic coherence, creativity, and lexical diversity.

---

## 🎯 Objectives

- Gain hands-on understanding of stochastic vs. deterministic decoding strategies in autoregressive Transformer models.
- Quantitatively evaluate the effect of:
  - **Temperature ($T$):** Softmax probability flattening vs. sharpening.
  - **Top-K Sampling:** Hard truncation of the token candidate pool.
  - **Top-P (Nucleus) Sampling:** Cumulative probability mass truncation.
  - **Repetition Penalty ($\theta$):** Logit discounting for previously generated tokens.
- Generate multi-style creative narratives using controlled decoding parameter sets.

---

## 🔬 Mathematical Formulations of Decoding Strategies

### 1. Temperature Scaling
Given logit vector $u$, the probability of token $i$ is scaled by temperature $T > 0$:
$$P(x_t = i \mid x_{<t}) = \frac{\exp(u_i / T)}{\sum_j \exp(u_j / T)}$$
- **$T \to 0$:** Collapses to greedy search (argmax), deterministic, repetitive.
- **$T > 1$:** Flattens distribution, increasing entropy and randomness.

### 2. Top-K Filtering
Restricts sampling to the $K$ most likely next tokens:
$$\mathcal{V}^{(K)} = \operatorname{arg top-k}_{i \in \mathcal{V}} P(x_t = i \mid x_{<t})$$

### 3. Top-P (Nucleus) Sampling
Selects the smallest subset of tokens whose cumulative probability exceeds threshold $p \in (0, 1]$:
$$\sum_{i \in \mathcal{V}^{(p)}} P(x_t = i \mid x_{<t}) \ge p$$

### 4. Repetition Penalty
For each token $i$ already appearing in the context, its logit $u_i$ is modified:
$$u_i' = \begin{cases} u_i / \theta & \text{if } u_i > 0 \\ u_i \cdot \theta & \text{if } u_i < 0 \end{cases} \quad (\theta > 1.0)$$

---

## 📊 Systematic Experimental Results

**Base Prompt:** *"AI is changing the world by..."*

| Exp # | Parameter Altered | Value | Generated Sample | Qualitative Observation |
|:---:|---|:---:|---|---|
| **Base** | Standard Sampling | Defaults | *"...improving business processes and offering new solutions across diverse sectors."* | Baseline conversational style with moderate diversity. |
| **01** | Temperature | `0.3` | *"...improving healthcare, education, and transportation with intelligent solutions."* | Highly logical, factually conservative, and predictable syntax. |
| **02** | Top-K | `20` | *"...through smart systems that help businesses and people make better decisions."* | Focused vocabulary; prevents rare or anomalous tokens. |
| **03** | Top-P (Nucleus) | `0.60` | *"...by creating intelligent technologies that learn from data and improve daily life."* | Highly fluent, natural human cadence without awkward phrasing. |
| **04** | Repetition Penalty | `1.8` | *"...by encouraging innovation, increasing productivity, and supporting creative problem-solving."* | Eliminates tautologies and looping phrases; boosts vocabulary diversity. |
| **05** | High Temperature | `2.0` | *"...while rivers sing to digital mountains and forgotten machines paint invisible memories."* | Extremely hallucinatory; poetically rich yet semantically fractured. |

---

## 🎨 Controlled Creative Storytelling

**Narrative Scenario:**  
> *"A severe monsoon flood devastated a small village in Nepal. As the rain finally stopped, a young volunteer stepped into the muddy streets, determined to help families rebuild their lives."*

### 1. Realistic & Grounded (`Temp = 0.5`)
> *"The volunteer organized food distribution and helped children return to temporary classrooms. Although many homes were damaged, hope began to grow as neighbors worked together to rebuild their community."*
- **Outcome:** Structured, empathetic, documentary-like journalism.

### 2. Expressive & Dynamic (`Temp = 0.8`, `Top-K = 40`)
> *"Carrying medical supplies and warm blankets, the young volunteer inspired others to join the rescue effort. Every repaired bridge and rebuilt house became a symbol of courage and unity across the flood-affected village."*
- **Outcome:** Cinematic, emotionally resonant, balanced vocabulary.

### 3. Poetic & Diverse (`Temp = 1.0`, `Top-P = 0.9`)
> *"As the sun appeared after days of heavy rain, the village slowly came back to life. People planted new crops, rebuilt schools, and proved that kindness and determination could overcome even the worst natural disaster."*
- **Outcome:** Evocative storytelling with rich imagery and smooth transitions.

---

## 💡 Practical Takeaways

1. **Deterministic Tasks (Q&A, Code Generation, Summarization):** Set $T \in [0.1, 0.3]$ with $P \in [0.85, 0.95]$ or greedy search.
2. **Conversational AI & Dialogue:** Set $T \in [0.6, 0.7]$, $Top\text{-}K \in [40, 50]$, and $Repetition\ Penalty \approx 1.15$.
3. **Creative Writing & Brainstorming:** Set $T \in [0.8, 1.0]$, $Top\text{-}P \in [0.90, 0.95]$.

---

## 💻 How to Run

```bash
# Launch Jupyter Notebook
jupyter notebook "Exploring Text Generation Using GPT-2.ipynb"
```

---

## 📚 Dependencies
- `transformers`
- `torch`
- `jupyter`

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
