# Assignment 10: Variational Autoencoder for Compression & Denoising (Colab Edition)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This project implements an end-to-end, reproducible **Google Colab pipeline** evaluating **Variational Autoencoders (VAEs)** across two core machine learning tasks: **Lossy Image Compression** and **Stochastic Denoising**. Using the benchmark MNIST dataset ($28 \times 28 = 784$ pixels), the implementation provides complete numerical evaluation tables, Mean Squared Error (MSE) metrics across individual test instances, and systematic latent dimensionality experiments ($k \in \{2, 16, 32\}$).

---

## 🎯 Objectives

- Implement an end-to-end VAE architecture featuring custom Keras sampling layers and the reparameterization trick.
- Formulate and optimize the combined ELBO loss function ($\text{Reconstruction Loss} + \text{KL Divergence}$).
- Quantify empirical image compression ratios achieved by mapping $784$-dimensional pixel spaces into $2$, $16$, and $32$-dimensional latent spaces.
- Test the robustness of learned image priors against zero-mean additive Gaussian noise.
- Quantitatively measure per-image reconstruction errors (MSE) before and after denoising.

---

## 🔬 Architecture & Mathematical Foundations

### 1. The Encoder Network ($q_\phi(z \mid x)$)
Maps the $784$-dimensional input image to latent distribution parameters:
- **Input:** $x \in [0, 1]^{784}$
- **Intermediate Layer:** `Dense(256, activation='relu')`
- **Latent Parameters:** 
  - Mean vector: $\mu = \text{Dense}(k)$
  - Log-variance vector: $\log(\sigma^2) = \text{Dense}(k)$

### 2. Reparameterization Trick
To allow backpropagation through stochastic sampling:
$$z = \mu + \exp\left(\frac{1}{2} \log(\sigma^2)\right) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_k)$$

### 3. The Decoder Network ($p_\theta(x \mid z)$)
Reconstructs the image from latent coordinates:
- **Input:** $z \in \mathbb{R}^k$
- **Intermediate Layer:** `Dense(256, activation='relu')`
- **Output Layer:** $\hat{x} = \text{Dense}(784, activation='sigmoid')$

### 4. Loss Function (ELBO)
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(x, \hat{x}) + D_{\text{KL}}\left(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I)\right)$$
$$D_{\text{KL}} = -\frac{1}{2} \sum_{i=1}^k \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)$$

---

## 📊 Experimental Results

### 1. Compression Ratio Evaluation
$$\text{Compression Ratio} = \frac{\text{Input Dimension}}{\text{Latent Dimension}} = \frac{784}{k}$$

| Latent Dimension ($k$) | Theoretical Compression Ratio | Percentage Space Saved |
|:---:|:---:|:---:|
| **2** | **392 : 1** | **99.74%** |
| **16** | **49 : 1** | **97.96%** |
| **32** | **24.5 : 1** | **95.92%** |

### 2. Reconstruction & Denoising Dynamics
- **Clean Reconstruction:** The model accurately preserves overall stroke shapes, topological loops, and structural symmetry across all digits.
- **Denoising with Gaussian Corruption:** Adding Gaussian noise ($\mu=0, \sigma=0.3$) obscures subtle strokes. The VAE successfully suppresses the high-frequency pixel deviations because the decoder restricts outputs to its learned continuous digit manifold.

### 3. Dimensionality Trade-Off Analysis
- **$k = 2$:** Maximizes compression ($392:1$), but sacrifices fine stroke definition (slightly blurrier edges).
- **$k = 16$:** Optimal trade-off between compact representation ($49:1$) and sharp reconstruction.
- **$k = 32$:** Delivers the highest fidelity reconstruction with minimal structural distortion.

---

## 💻 How to Run

### Google Colab
1. Upload [`GEN_AI_ASSIGNMENT_10.ipynb`](./GEN_AI_ASSIGNMENT_10.ipynb) to Google Colab.
2. Run all cells sequentially from top to bottom.
3. Observe the generated comparison plots displaying Clean, Noisy, and Denoised digit arrays.

### Local Execution
```bash
jupyter notebook "GEN_AI_ASSIGNMENT_10.ipynb"
```

---

## 📚 Dependencies
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
