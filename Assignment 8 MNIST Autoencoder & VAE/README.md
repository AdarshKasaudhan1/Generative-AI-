# Assignment 8: Deterministic Autoencoders vs. Variational Autoencoders (VAEs) on MNIST

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This study establishes a rigorous technical and empirical comparison between **Deterministic Autoencoders (AE)** and **Variational Autoencoders (VAE)** trained on the **MNIST** handwritten digit dataset. We analyze how introducing probabilistic prior conditioning and the **Reparameterization Trick** transforms a discrete, discontinuous latent representation into a continuous, generative 2D manifold capable of synthesizing novel digits through smooth spatial interpolation.

---

## 🎯 Objectives

- Architect both deterministic and probabilistic encoder-decoder models in TensorFlow/Keras.
- Understand the mathematical necessity of the **Reparameterization Trick** for enabling end-to-end backpropagation through stochastic sampling nodes.
- Derive and implement the Evidence Lower Bound (**ELBO**) loss: balancing pixel-wise Reconstruction Loss against Kullback-Leibler (**KL**) Divergence.
- Project high-dimensional digit manifolds ($28 \times 28 = 784$ dimensions) into an interpretable **2D Latent Space**.
- Generate synthetic handwritten digits across a uniform 2D grid ($z_1, z_2 \in [-3, 3]$) to inspect generative continuity.

---

## 🔬 Mathematical Formulations

### 1. Deterministic Autoencoder (AE)
- **Encoder:** Maps input $x$ to a single deterministic point $z = f_\phi(x) \in \mathbb{R}^2$.
- **Decoder:** Reconstructs image $\hat{x} = g_\theta(z)$.
- **Loss:** Mean Squared Error (MSE) or Binary Cross-Entropy (BCE) without any distributional constraint on $z$.
- **Limitation:** Latent space contains vast "holes" and discontinuities; sampling from arbitrary coordinates yields nonsensical, blurred artifacts.

### 2. Variational Autoencoder (VAE)
- **Encoder:** Outputs parameters of a Gaussian distribution: mean vector $\mu \in \mathbb{R}^2$ and log-variance vector $\log(\sigma^2) \in \mathbb{R}^2$.
- **The Reparameterization Trick:** Enables gradient backpropagation by isolating stochasticity into an external parameter-free noise variable $\epsilon$:
  $$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
- **Evidence Lower Bound (ELBO) Objective:**
  $$\mathcal{L}_{\text{VAE}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))$$
- **Reconstruction Loss:** Pixel-level Binary Cross-Entropy:
  $$\mathcal{L}_{\text{recon}} = -\sum_{j=1}^{784} \left[ x_j \log \hat{x}_j + (1 - x_j) \log(1 - \hat{x}_j) \right]$$
- **Closed-Form KL Divergence:** Regularizes $q_\phi(z \mid x)$ toward a standard Gaussian prior $\mathcal{N}(0, I)$:
  $$D_{\text{KL}} = -\frac{1}{2} \sum_{k=1}^2 \left[ 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right]$$

---

## 🛠️ Model Architectures

```
[Input: 28x28 (784-d)]
          │
          ├──► [Part A: Simple Autoencoder]
          │         Dense(128, ReLU) ──► Dense(64, ReLU) ──► Bottleneck: Dense(2)
          │         Dense(64, ReLU)  ──► Dense(128, ReLU) ──► Dense(784, Sigmoid) ──► Output
          │
          └──► [Part B: Variational Autoencoder]
                    Dense(256, ReLU)
                          │
                    ┌─────┴─────┐
                    ▼           ▼
               μ: Dense(2)   log(σ²): Dense(2)
                    └─────┬─────┘
                          ▼
            Reparameterization Trick: z = μ + σ ⊙ ε
                          │
                    Dense(256, ReLU) ──► Dense(784, Sigmoid) ──► Output
```

---

## 📊 Comparative Analysis & Observations

| Feature / Metric | Part A: Simple Autoencoder | Part B: Variational Autoencoder (VAE) |
|---|---|---|
| **Latent Geometry** | Discrete, irregular clusters with unmapped voids | Dense, continuous manifold centered at $(0,0)$ |
| **Generative Capability** | Poor (Generates unrealistic noise outside training points) | Excellent (Smooth interpolation across all digits) |
| **Loss Function** | Reconstruction loss only | Dual objective: Reconstruction + KL-Divergence |
| **Regularization** | None (Prone to overfitting) | Explicit Gaussian prior enforces continuous geometry |
| **Clustering Quality** | Disjoint clusters with arbitrary coordinate spread | Natural clustering with semantic proximity between similar digits |

### Research Questions Answered:
1. **Do similar digits form clusters in the 2D latent space?**  
   *Yes.* Visually similar digits (such as 3, 5, and 8, or 4, 7, and 9) cluster adjacent to one another because their topological pixel structures share substantial spatial overlap.
2. **Are the clusters cleanly separated?**  
   *Partially.* While distinct digit classes form distinct centroids, natural boundary overlap occurs between digits with ambiguous contours (e.g., a handwritten `3` transitioning into an `8`). The VAE smoothly interpolates through these transitions rather than producing distorted artifacts.

---

## 💻 How to Run

```bash
# Launch Jupyter Notebook
jupyter notebook "Assignment_8 GEN AI.ipynb"
```

---

## 📚 References
> Kingma, D. P., & Welling, M. (2013). **Auto-Encoding Variational Bayes.** *arXiv preprint arXiv:1312.6114*.

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
