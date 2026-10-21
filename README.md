<div align="center">

# 🧠 Generative Artificial Intelligence
### Comprehensive Laboratory & Research Portfolio | Semester V

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Enabled-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)

<p align="center">
  <b>A production-grade, mathematically grounded implementation portfolio exploring deep generative modeling paradigms.</b><br>
  Spanning Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Neural Style Transfer, Autoregressive Large Language Models (LLMs), and Responsible AI Auditing.
</p>

---

</div>

## 📖 Executive Overview

This repository comprises an advanced academic and practical portfolio developed for the **Generative AI** curriculum (Semester V). Generative Artificial Intelligence represents a fundamental paradigm shift from conventional discriminative classification to high-dimensional probabilistic distribution learning. Rather than estimating conditional boundaries $P(Y \mid X)$, generative architectures approximate the true underlying data distribution $P(X)$ or joint distribution $P(X, Y)$, enabling the synthesis, reconstruction, and transformation of complex real-world modalities including imagery, semantic text, and stylistic representations.

This codebase encompasses **11 end-to-end laboratory assignments and research modules**, systematically exploring:
- **Foundational Paradigms:** Comparative dynamics of discriminative classifiers vs. generative density approximators.
- **Latent Variable Models:** Deterministic Autoencoders vs. Probabilistic Variational Autoencoders (VAEs), manifold regularization, and the reparameterization trick.
- **Deep Generative Computer Vision:** Convolutional GANs (DCGAN), high-resolution latent space interpolation (StyleGAN2-ADA), and empirical mitigation of training pathologies like mode collapse.
- **Neural Style Transfer & Domain Translation:** Paired/unpaired adversarial domain translation (CycleGAN) contrasted with iterative feature Gram-matrix optimization (VGG-19 NST).
- **Autoregressive Large Language Models:** Decoding strategies (Nucleus sampling, Top-K, Temperature scaling), algorithmic and representational bias auditing, and domain-adaptive fine-tuning on dramatic verse.

---

## 🏛️ Core Architectural Pillars

```
                                      ┌─────────────────────────────────────────────────┐
                                      │        Generative AI Foundations & Models       │
                                      └────────────────────────┬────────────────────────┘
                                                               │
         ┌─────────────────────────┬───────────────────────────┼───────────────────────────┬─────────────────────────┐
         ▼                         ▼                           ▼                           ▼                         ▼
┌─────────────────┐       ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐       ┌─────────────────┐
│  Pillar 1:      │       │  Pillar 2:      │         │  Pillar 3:      │         │  Pillar 4:      │       │  Pillar 5:      │
│  Discriminative │       │  Latent Spaces  │         │  Adversarial    │         │  Style Transfer │       │  Language       │
│  vs. Generative │       │  & VAEs         │         │  Vision (GANs)  │         │  & Translation  │       │  Models & NLP   │
├─────────────────┤       ├─────────────────┤         ├─────────────────┤         ├─────────────────┤       ├─────────────────┤
│ • Logistic Reg. │       │ • Vanilla AE    │         │ • DCGAN         │         │ • CycleGAN      │       │ • GPT-2 Decoding│
│ • Autoencoders  │       │ • VAE (ELBO)    │         │ • StyleGAN2-ADA │         │ • VGG-19 NST    │       │ • Bias Auditing │
│ • Joint P(X, Y) │       │ • Reparameter.  │         │ • Mode Collapse │         │ • Gram Matrices │       │ • Domain Tuning │
│ • Manifold Map  │       │ • Compression   │         │ • Label Smooth  │         │ • Cycle Loss    │       │ • Style Metrics │
└─────────────────┘       └─────────────────┘         └─────────────────┘         └─────────────────┘       └─────────────────┘
```

---

## 📑 Master Index of Assignments

| # | Module / Topic | Primary Objective | Architecture / Key Tech | Dataset | Framework | Documentation & Notebook |
|:---:|---|---|---|---|:---:|:---:|
| **01** | **Generative vs. Discriminative Models** | Compare classification boundaries against data manifold reconstruction | Logistic Regression vs. Fully-Connected Autoencoder | MNIST (Digits 0 & 1) | TensorFlow / Scikit-Learn | [📁 Docs](./Assignment%201%20Understanding%20Generative%20and%20Discriminative%20Models%20Using%20MNIST/README.md) • [📓 Notebook](./Assignment%201%20Understanding%20Generative%20and%20Discriminative%20Models%20Using%20MNIST/Assignment_1_GEN_AI.ipynb) |
| **02** | **Exploring Text Generation Using GPT-2** | Analyze decoding heuristics (Temperature, Top-K, Top-P, Repetition Penalty) | GPT-2 (124M Base) Auto-Regressive LM | Open Prompts & Nepal Narrative | Hugging Face / PyTorch | [📁 Docs](./Assignment%202%20Exploring%20Text%20Generation%20Using%20GPT-2/README.md) • [📓 Notebook](./Assignment%202%20Exploring%20Text%20Generation%20Using%20GPT-2/Exploring%20Text%20Generation%20Using%20GPT-2.ipynb) |
| **03** | **Identifying Bias in Pretrained Generative AI** | Audit occupational, gender, and social stereotyping in language completions | GPT-2 Auditing Framework | 6 Benchmark Professions | Hugging Face / PyTorch | [📁 Docs](./Assignment%203%20Identifying%20Bias%20in%20a%20Pretrained%20Generative%20AI%20Model/README.md) • [📄 Report PDF](./Assignment%203%20Identifying%20Bias%20in%20a%20Pretrained%20Generative%20AI%20Model/Identifying%20Bias%20in%20a%20Pretrained%20Generative%20AI%20Model.pdf) |
| **04** | **Training a DCGAN on Fashion-MNIST** | Synthesize apparel images using fractional-strided convolutional GANs | Deep Convolutional GAN (DCGAN) | Fashion-MNIST (10 Classes) | TensorFlow / Keras | [📁 Docs](./Assignment%204%20Training%20a%20DCGAN%20on%20the%20Fashion-MNIST%20Dataset/README.md) • [📓 Notebook](./Assignment%204%20Training%20a%20DCGAN%20on%20the%20Fashion-MNIST%20Dataset/DCGAN_Fashion_MNIST_TensorFlow_Colab%20Ass%204.ipynb) |
| **05** | **Latent Space Interpolation with StyleGAN2-ADA** | Probe latent vector trajectories and attribute disentanglement in high-res faces | StyleGAN2-ADA (NVIDIA PyTorch) | FFHQ Pretrained Weights | PyTorch | [📁 Docs](./Assignment%205%20Latent%20Space%20Interpolation%20and%20Image%20Generation%20Using%20StyleGAN2-ADA/README.md) • [📓 Notebook](./Assignment%205%20Latent%20Space%20Interpolation%20and%20Image%20Generation%20Using%20StyleGAN2-ADA/Latent_Space_Interpolation_and_Image_Generation_Using_StyleGAN2_ADAipynb.ipynb) |
| **06** | **AI-Based Image Style Transfer** | Contrast unpaired domain translation with Gram-matrix content/style optimization | CycleGAN (Monet) vs. VGG-19 NST (Van Gogh) | Landscape & The Starry Night | PyTorch / TorchVision | [📁 Docs](./Assignment%206%20—%20AI-Based%20Image%20Style%20Transfer/README.md) • [📓 Notebook](./Assignment%206%20—%20AI-Based%20Image%20Style%20Transfer/GEN_AI_ASSIGNMENT_6.ipynb) |
| **07** | **Fixing Mode Collapse in GANs via Label Smoothing** | Induce generator degeneracy and resolve it via one-sided label smoothing | DCGAN with Modified Gradient Dynamics | MNIST (Digits 0–9) | PyTorch / TorchVision | [📁 Docs](./Assignment%207%20Creating%20and%20Fixing%20Mode%20Collapse%20in%20a%20Generative%20Adversarial%20Network%20%28GAN%29%20Using%20Label%20Smoothing/README.md) • [📓 Notebook](./Assignment%207%20Creating%20and%20Fixing%20Mode%20Collapse%20in%20a%20Generative%20Adversarial%20Network%20%28GAN%29%20Using%20Label%20Smoothing/Creating_and_Fixing_Mode_Collapse_in_a_Generative_Adversarial_Network_%28GAN%29_Using_Label_Smoothing_7.ipynb) |
| **08** | **MNIST Deterministic Autoencoder vs. VAE** | Evaluate 2D latent space topology, cluster separation, and continuous generation | Dense Autoencoder vs. Variational Autoencoder | MNIST Benchmark | TensorFlow / Keras | [📁 Docs](./Assignment%208%20MNIST%20Autoencoder%20&%20VAE/README.md) • [📓 Notebook](./Assignment%208%20MNIST%20Autoencoder%20&%20VAE/Assignment_8%20GEN%20AI.ipynb) |
| **09** | **VAE for Image Compression & Denoising** | Measure lossy compression ratios and stochastic Gaussian noise suppression | Variational Autoencoder ($k \in \{2, 16, 32\}$) | MNIST Benchmark | TensorFlow / Keras | [📁 Docs](./Assignment%209%20Variational%20Autoencoder%20for%20Image%20Compression%20and%20Denoising/README.md) • [📓 Notebook](./Assignment%209%20Variational%20Autoencoder%20for%20Image%20Compression%20and%20Denoising/assgn_09_gen_ai.ipynb) |
| **10** | **VAE Compression & Denoising (Colab Suite)** | Comprehensive per-sample reconstruction MSE benchmarking and ablation | Variational Autoencoder (Colab Edition) | MNIST Benchmark | TensorFlow / Keras | [📁 Docs](./Assignment%2010%20—%20VAE%20for%20Compression%20and%20Denoising/README.md) • [📓 Notebook](./Assignment%2010%20—%20VAE%20for%20Compression%20and%20Denoising/GEN_AI_ASSIGNMENT_10.ipynb) |
| **11** | **Fine-Tuning GPT-2 on Custom Text Style** | Adapt language model voice, rhythm, and syntax toward Early Modern English | GPT-2 Fine-Tuned via Hugging Face Trainer | Tiny Shakespeare (*Coriolanus*) | Transformers / PyTorch | [📁 Docs](./Assignment%2011%20Fine-Tuning%20GPT-2%20on%20a%20Custom%20Text%20Style/README.md) • [📓 Notebook](./Assignment%2011%20Fine-Tuning%20GPT-2%20on%20a%20Custom%20Text%20Style/GEN_AI_ASSIGNMENT_11.ipynb) |

---

## 🔍 In-Depth Module Walkthrough

### 1. Generative vs. Discriminative Foundations
- **Core Concept:** Discriminative algorithms seek to model $P(Y \mid X)$ by establishing decision boundaries that separate classes. Generative algorithms model the joint probability $P(X, Y)$ or data prior $P(X)$, enabling unsupervised pattern reconstruction.
- **Implementation:** Compares Logistic Regression against a 4-layer Autoencoder with a 32-dimensional bottleneck on MNIST digits 0 and 1.
- **Key Outcome:** Logistic regression achieves $>99.8\%$ accuracy by isolating separating hyperplanes; the Autoencoder achieves $95.9\%$ dimensionality reduction while preserving topological character geometry.

### 2. Autoregressive Text Decoding Strategies
- **Core Concept:** Explores autoregressive sampling heuristics $P(w_t \mid w_{<t})$ in GPT-2 (124M).
- **Parameters Evaluated:** Temperature $T \in [0.3, 2.0]$, Top-K ($K = 20$), Top-P Nucleus ($p = 0.60$), and Repetition Penalty ($\theta = 1.8$).
- **Key Outcome:** Temperature controls the entropy of the softmax logit distribution ($T \to 0$ yields conservative, repetitive text; $T \ge 2.0$ causes catastrophic hallucinations). Top-P dynamic truncation generates the most natural, human-like cadence.

### 3. Auditing Algorithmic Bias in Foundation Models
- **Core Concept:** Audits systematic representational bias inherited by large language models from historical internet training corpora.
- **Methodology:** Evaluates GPT-2 generations primed with neutral prompts across 6 societal roles (Doctor, Nurse, Executive, Scientist, Police Officer, Criminal).
- **Key Outcome:** Uncovers statistically significant gender and tone associations (e.g., Doctors and Executives heavily skewed toward male pronouns and authoritative language; Nurses skewed toward female pronouns and nurturing language). Outlines formal CDA and alignment mitigation strategies.

### 4. Deep Convolutional Generative Adversarial Networks (DCGAN)
- **Core Concept:** Radford et al.'s DCGAN replaces fully-connected layers with spatial transposed convolutions and strided convolutions.
- **Implementation:** Synthesizes $28 \times 28 \times 1$ apparel articles from $z \sim \mathcal{N}(0, I_{100})$ using custom `tf.GradientTape` adversarial training loops on Fashion-MNIST.
- **Key Outcome:** Demonstrates progressive training evolution over 30–40 epochs—transitioning from amorphous noise (epochs 1–5) to recognizable footwear and bags (epochs 10–15) to textured garments with fine hems and soles (epochs 20–30).

### 5. High-Resolution Latent Space Interpolation (StyleGAN2-ADA)
- **Core Concept:** Analyzes continuous latent space topology in NVIDIA's StyleGAN2-ADA architecture.
- **Methodology:** Applies Linear Interpolation (LERP) and Spherical Linear Interpolation (SLERP) across the mapping network's latent vector spaces to generate morphing photorealistic facial transitions.
- **Key Outcome:** Validates manifold continuity: intermediate vectors generate photorealistic, smooth intermediate faces without morphing anomalies. Probes the entanglement of the raw $\mathcal{Z}$-space versus the intermediate $\mathcal{W}$-space.

### 6. Neural Style Transfer & Domain Translation
- **Core Concept:** Directly contrasts feedforward generative domain translation (CycleGAN) against test-time optimization (VGG-19 NST).
- **Implementation:** Translates a landscape photograph into Claude Monet's impressionist domain via cycle-consistent adversarial mapping, and separately stylizes it using Vincent van Gogh's *The Starry Night* via multi-layer Gram matrix loss optimization.
- **Key Outcome:** CycleGAN performs instant feedforward domain adaptation ($\approx 50\text{ ms}$), altering overall color temperature and painterly ambiance; VGG-19 NST accurately fuses high-frequency brush textures while strictly preserving edge layouts via explicit content loss penalties.

### 7. Inducing and Fixing GAN Mode Collapse
- **Core Concept:** Mode collapse occurs when the generator collapses to producing an extremely narrow subset of samples that reliably fool the discriminator.
- **Methodology:** Intentionally destabilizes a PyTorch DCGAN by stripping Batch Normalization, constricting capacity, and drastically increasing the learning rate. Then remediates training by applying **One-Sided Label Smoothing** ($y_{\text{real}} = 0.9$).
- **Key Outcome:** Label smoothing eliminates discriminator logit saturation, preventing vanishing gradients and restoring full multi-digit diversity ($0\text{--}9$) across fixed latent probe vectors.

### 8. Deterministic vs. Variational Autoencoders (VAEs)
- **Core Concept:** Contrasts unregularized deterministic feature compression with probabilistic Gaussian latent conditioning.
- **Implementation:** Compares a standard autoencoder against a VAE using the **Reparameterization Trick** ($z = \mu + \sigma \odot \epsilon$) and the Evidence Lower Bound (ELBO) objective.
- **Key Outcome:** The standard autoencoder's 2D latent space exhibits non-convex clusters and vast unmapped voids where sampling yields gibberish. The VAE produces a continuous, centered Gaussian manifold allowing seamless 2D grid digit synthesis.

### 9 & 10. VAEs for Lossy Compression & Stochastic Denoising
- **Core Concept:** Formulates the dual application of VAE bottlenecks for massive bandwidth reduction and implicit manifold projection denoising.
- **Methodology:** Explores latent bottleneck dimensions $k \in \{2, 16, 32\}$ on MNIST images corrupted with additive Gaussian noise ($\sigma \in [0.3, 0.5]$).
- **Key Outcome:** Latent dimension $k = 2$ achieves a **$392 : 1$** compression ratio ($99.74\%$ space reduction); $k = 16$ ($49 : 1$) provides the optimal Pareto frontier balancing noise suppression with sharp stroke reconstruction.

### 11. Domain-Adaptive LLM Fine-Tuning
- **Core Concept:** Adapts the syntactic, lexical, and tonal persona of a pretrained autoregressive transformer (GPT-2, 124M) using domain-specific text.
- **Implementation:** Curates and normalizes 250 dialogue lines from Shakespeare's *Coriolanus* (Tiny Shakespeare). Fine-tunes the model using the Hugging Face `Trainer` API with causal language modeling loss.
- **Key Outcome:** When prompted with modern technical phrases (*"The future of artificial intelligence will"*), the base model generates modern expository prose, whereas the fine-tuned model shifts into poetic theatrical blank verse with high archaic token density (*thou, thee, thy, 'tis*).

---

## 📂 Repository File Structure

```
GEN AI/
├── .gitignore                                                                                        # Git exclusion rules
├── LICENSE                                                                                           # MIT Open Source License
├── README.md                                                                                         # Master portfolio documentation
├── requirements.txt                                                                                 # Unified Python environment dependencies
│
├── Assignment 1 Understanding Generative and Discriminative Models Using MNIST/
│   ├── Assignment_1_GEN_AI.ipynb                                                                     # Logistic Regression vs. Autoencoder notebook
│   └── README.md                                                                                     # Theoretical analysis and execution guide
│
├── Assignment 2 Exploring Text Generation Using GPT-2/
│   ├── Exploring Text Generation Using GPT-2.ipynb                                                   # Decoding strategies & custom narratives notebook
│   └── README.md                                                                                     # Decoding parameters comparative guide
│
├── Assignment 3 Identifying Bias in a Pretrained Generative AI Model/
│   ├── Identifying Bias in a Pretrained Generative AI Model.pdf                                      # Comprehensive research evaluation report
│   └── README.md                                                                                     # Bias auditing framework & ethical documentation
│
├── Assignment 4 Training a DCGAN on the Fashion-MNIST Dataset/
│   ├── DCGAN_Fashion_MNIST_TensorFlow_Colab Ass 4.ipynb                                              # Convolutional GAN implementation notebook
│   └── README.md                                                                                     # DCGAN architecture & epoch dynamics guide
│
├── Assignment 5 Latent Space Interpolation and Image Generation Using StyleGAN2-ADA/
│   ├── Latent_Space_Interpolation_and_Image_Generation_Using_StyleGAN2_ADAipynb.ipynb                # High-res latent vector manipulation notebook
│   └── README.md                                                                                     # StyleGAN2-ADA mechanics & interpolation guide
│
├── Assignment 6 — AI-Based Image Style Transfer/
│   ├── GEN_AI_ASSIGNMENT_6.ipynb                                                                     # CycleGAN vs. VGG-19 Style Transfer notebook
│   └── README.md                                                                                     # Domain translation vs. Gram matrix guide
│
├── Assignment 7 Creating and Fixing Mode Collapse in a Generative Adversarial Network (GAN) Using Label Smoothing/
│   ├── Creating_and_Fixing_Mode_Collapse_in_a_Generative_Adversarial_Network_(GAN)_Using_Label_Smoothing_7.ipynb # Mode collapse experiment notebook
│   └── README.md                                                                                     # GAN stability & label smoothing guide
│
├── Assignment 8 MNIST Autoencoder & VAE/
│   ├── Assignment_8 GEN AI.ipynb                                                                     # Deterministic AE vs. VAE manifold notebook
│   └── README.md                                                                                     # 2D latent topology & ELBO formulation guide
│
├── Assignment 9 Variational Autoencoder for Image Compression and Denoising/
│   ├── assgn_09_gen_ai.ipynb                                                                         # Latent dimension ablation notebook
│   └── README.md                                                                                     # Compression ratios & denoising audit guide
│
├── Assignment 10 — VAE for Compression and Denoising/
│   ├── GEN_AI_ASSIGNMENT_10.ipynb                                                                    # End-to-end Colab VAE benchmarking notebook
│   └── README.md                                                                                     # Per-sample MSE & ablation study documentation
│
└── Assignment 11 Fine-Tuning GPT-2 on a Custom Text Style/
    ├── GEN_AI_ASSIGNMENT_11.ipynb                                                                    # Shakespearean GPT-2 fine-tuning notebook
    └── README.md                                                                                     # Style quantification & generation audit guide
```

---

## ⚡ Environment Setup & Installation

### 1. Prerequisites
- **Python:** 3.8 to 3.11 recommended
- **Hardware Acceleration:** NVIDIA GPU with CUDA 11.8+ is strongly recommended for Assignments 4, 5, 6, 7, and 11. Alternatively, Google Colab provides free T4 GPU access.

### 2. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/AdarshKasaudhan1/Generative-AI-.git
cd "Generative-AI-"

# Create an isolated Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Upgrade package installer
pip install --upgrade pip

# Install complete project dependencies
pip install -r requirements.txt
```

### 3. Launching Interactive Notebooks

```bash
# Launch Jupyter Notebook server
jupyter notebook

# Or launch JupyterLab
jupyter lab
```

---

## 🤝 Responsible AI & Ethical Considerations

Generative AI systems possess immense synthetic capabilities but carry significant societal responsibilities. The experiments in this portfolio adhere to ethical guidelines:
- **Bias Transparency:** Assignment 3 explicitly isolates and quantifies occupational stereotypes rather than concealing model flaws, encouraging accountability.
- **Safety in Domain Adaptation:** Assignment 11 fine-tunes strictly on public-domain historical literature, avoiding copyright infringements or uncurated toxicity.
- **Adversarial Integrity:** Exploration of mode collapse and training instabilities (Assignment 7) is framed through the lens of model robustness and reproducibility.

---

## 📄 License

This repository is distributed under the terms of the **MIT License**. Refer to the [`LICENSE`](./LICENSE) file for complete terms and permissions.

---

## 👤 Author & Academic Attribution

**Adarsh Kumar Gupta Kasaudhan**  
- **Email:** [kadarshbca24@ced.alliance.edu.in](mailto:kadarshbca24@ced.alliance.edu.in)  
- **GitHub:** [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)  
- **Repository:** [Generative-AI-](https://github.com/AdarshKasaudhan1/Generative-AI-)  

---

<div align="center">
  <i>Developed with academic rigor, mathematical discipline, and a passion for deep generative modeling.</i>
</div>
