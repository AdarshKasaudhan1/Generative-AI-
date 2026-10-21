# Assignment 1: Understanding Generative vs. Discriminative Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E.svg)](https://scikit-learn.org/)

## 📌 Executive Summary

This project establishes the fundamental theoretical and practical distinction between **Discriminative Models** and **Generative Models** in Artificial Intelligence. Using a subset of handwritten digits (0 and 1) from the benchmark **MNIST dataset**, we implement and contrast:

1. **Discriminative Paradigm:** A **Logistic Regression** classifier that models the conditional probability $P(Y \mid X)$ to determine decision boundaries.
2. **Generative Paradigm:** A **Deep Autoencoder** neural network that models the data distribution and latent representation to reconstruct and generate digit images $P(X)$.

---

## 🎯 Objectives

- Comprehend the mathematical divergence between modeling posterior class boundaries vs. underlying data manifolds.
- Implement an interpretable discriminative classifier on raw pixel features ($28 \times 28 = 784$ dimensions).
- Architect an encoder-decoder neural network to compress and reconstruct image patterns.
- Quantitatively evaluate classification accuracy versus reconstruction loss.
- Compare sample efficiency, feature representations, and use-case applicability.

---

## 🔬 Mathematical Formulations

### 1. Discriminative Model: Logistic Regression
- **Probability Function:** 
  $$P(Y = 1 \mid X) = \sigma(W^T X + b) = \frac{1}{1 + e^{-(W^T X + b)}}$$
- **Loss Function:** Binary Cross-Entropy Loss
  $$\mathcal{L}_{\text{disc}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
- **Focus:** Finds a hyperplane separating digit `0` from digit `1`.

### 2. Generative Model: Autoencoder
- **Encoder Function:** $z = f_\phi(X) = \text{ReLU}(W_e X + b_e)$
- **Decoder Function:** $\hat{X} = g_\theta(z) = \sigma(W_d z + b_d)$
- **Reconstruction Loss:** Pixel-wise Binary Cross-Entropy
  $$\mathcal{L}_{\text{gen}}(\phi, \theta) = -\frac{1}{M} \sum_{j=1}^M \left[ x_j \log \hat{x}_j + (1 - x_j) \log(1 - \hat{x}_j) \right]$$
- **Focus:** Learns the underlying data manifold to faithfully reconstruct digit patterns from a compressed 32-dimensional bottleneck.

---

## 🛠️ Architecture & Workflow

```
[Input: 28x28 Image (784-d)]
          │
          ├──► [Discriminative: Logistic Regression] ──► Output: Class Label (0 or 1)
          │
          └──► [Generative: Autoencoder]
                    │
                    ├── Dense(128, ReLU)
                    ├── Bottleneck: Dense(32, ReLU) [Compressed Latent Space]
                    ├── Dense(128, ReLU)
                    └── Output: Dense(784, Sigmoid) ──► Reconstructed Image (28x28)
```

---

## 📊 Key Findings & Comparative Analysis

| Feature Dimension | Discriminative Model (Logistic Regression) | Generative Model (Autoencoder) |
|---|---|---|
| **Primary Goal** | Separate classes ($P(Y \mid X)$) | Reconstruct & represent data ($P(X)$) |
| **Input Data** | Images + Ground-truth labels | Raw images (Unsupervised / Self-supervised) |
| **Output** | Discrete classification probability ($0$ or $1$) | Continuous pixel intensity matrix ($28 \times 28$) |
| **Feature Representation** | Discards variance irrelevant to class boundaries | Retains structural, spatial, and topological features |
| **Generation Capability** | Cannot synthesize new data instances | Synthesizes and reconstructs novel variations |
| **Primary Evaluation** | Accuracy, Precision, Recall, F1-Score | Mean Squared Error (MSE), Reconstruction Cross-Entropy |

### Experimental Observations
- **Discriminative Classifier:** Achieves near-perfect test accuracy ($\approx 99.8\%$) within seconds, demonstrating high discriminative power for simple topological differences (circles vs. vertical strokes).
- **Generative Autoencoder:** Successfully captures stroke contours, ink density, and digit geometry, producing crisp reconstructions from just 32 latent parameters (a $95.9\%$ dimensionality reduction).

---

## 💻 How to Run

### Option A: Google Colab
Upload `Assignment_1_GEN_AI.ipynb` to [Google Colab](https://colab.research.google.com/) and run all cells sequentially.

### Option B: Local Jupyter Notebook
```bash
# 1. Activate your virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 2. Launch Jupyter Notebook
jupyter notebook "Assignment_1_GEN_AI.ipynb"
```

---

## 📚 Dependencies
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow` / `keras`

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
