# Assignment 9: Variational Autoencoder for Image Compression and Denoising

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This project investigates the dual utility of **Variational Autoencoders (VAEs)** as lossy image compression engines and statistical denoising filters. Implemented in **TensorFlow/Keras** on the **MNIST** benchmark, we systematically evaluate how the capacity of the stochastic bottleneck ($k \in \{2, 16, 32\}$) affects the trade-off between **Compression Ratio**, **Reconstruction Mean Squared Error (MSE)**, and **Gaussian Noise Filtering Capability**.

---

## 🎯 Objectives

- Architect a modular Variational Autoencoder with reparameterization layers for arbitrary latent bottleneck widths.
- Evaluate theoretical and empirical compression ratios achieved by projecting $784$-dimensional images into compact latent embeddings.
- Synthesize corrupted test benchmarks by injecting zero-mean additive Gaussian noise ($\epsilon \sim \mathcal{N}(0, \sigma^2)$).
- Test the implicit denoising capacity of VAE priors when conditioned on noisy observations.
- Conduct an ablation study comparing latent dimensions $k = 2, 16, 32$ across fidelity, compression, and denoising metrics.

---

## 🔬 Mathematical Principles & Formulations

### 1. Compression Ratio Metric
For an uncompressed input tensor $X \in \mathbb{R}^{28 \times 28}$ and latent vector $z \in \mathbb{R}^k$:
$$\text{Theoretical Compression Ratio} = \frac{\text{Original Dimension}}{\text{Latent Dimension}} = \frac{784}{k}$$
$$\text{Data Reduction Percentage} = \left(1 - \frac{k}{784}\right) \times 100\%$$

### 2. Denoising Mechanics
Given clean ground-truth image $x \in [0, 1]^{784}$, we generate corrupted input $\tilde{x}$:
$$\tilde{x} = \operatorname{clip}(x + \eta, 0, 1), \quad \eta \sim \mathcal{N}(0, \sigma^2)$$

Because the decoder $p_\theta(x \mid z)$ has learned to project only plausible points from the clean digit manifold $\mathcal{M}$, passing $\tilde{x}$ through the encoder and sampling $z$ acts as an implicit projection back onto $\mathcal{M}$, filtering out high-frequency uncorrelated Gaussian noise.

---

## 📊 Systematic Latent Dimension Ablation

| Latent Dimension ($k$) | Theoretical Compression Ratio | Space Reduction (%) | Reconstruction Fidelity (MSE) | Denoising Behavior & Quality |
|:---:|:---:|:---:|:---:|---|
| **$k = 2$** | **392 : 1** | **99.74%** | Moderate ($\approx 0.045$) | Aggressive smoothing; removes all noise but softens fine strokes. |
| **$k = 16$** | **49 : 1** | **97.96%** | High ($\approx 0.018$) | **Optimal sweet spot:** High stroke clarity with robust noise suppression. |
| **$k = 32$** | **24.5 : 1** | **95.92%** | Superior ($\approx 0.011$) | Crisp reconstructions; occasionally passes residual high-frequency noise. |

---

## 🛠️ Experimental Workflow

```
[Clean Image x (784-d)] ───────────┐
                                   ▼
[Gaussian Noise η] ──► [Corrupted Image x̃]
                                   │
                                   ▼
                        [VAE Encoder: q_φ(z|x̃)]
                                   │
                            [Sample z ∈ ℝᵏ]
                                   │
                        [VAE Decoder: p_θ(x|z)]
                                   │
                                   ▼
                        [Denoised Output x̂]
                                   │
      ┌────────────────────────────┴────────────────────────────┐
      ▼                                                         ▼
[Reconstruction MSE vs. x]                            [Visual Quality Audit]
```

---

## 💡 Practical Insights & Trade-Offs

1. **Extreme Compression ($k = 2$):** Achieves an astounding $392:1$ compression ratio, ideal for low-bandwidth edge transmission where only basic digit semantics are needed.
2. **Balanced Performance ($k = 16$):** Provides the best trade-off for practical image restoration, suppressing noise while retaining loop geometries (such as the loops in `6`, `8`, and `0`).
3. **Implicit vs. Explicit Denoising:** While standard VAEs successfully attenuate noise by manifold projection, training explicitly as a Denoising VAE (DVAE) where $\tilde{x}$ is presented during training yields even sharper boundary restoration.

---

## 💻 How to Run

```bash
# Launch Jupyter Notebook
jupyter notebook "assgn_09_gen_ai.ipynb"
```

---

## 📚 Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` / `keras`

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
