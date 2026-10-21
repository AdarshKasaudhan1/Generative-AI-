# Assignment 4: Training a Deep Convolutional GAN (DCGAN) on Fashion-MNIST

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This project implements and trains a **Deep Convolutional Generative Adversarial Network (DCGAN)** on the **Fashion-MNIST** apparel dataset using **TensorFlow 2.x and Keras**. Moving beyond primitive dense networks, this implementation uses strided transposed convolutions, spatial batch normalization, and LeakyReLU activations to learn the non-linear manifold of clothing articles (shoes, handbags, t-shirts, coats) directly from pure Gaussian noise.

---

## 🎯 Objectives

- Implement the foundational DCGAN principles formulated by Radford et al. (2015).
- Construct a transposed-convolutional Generator network that projects a 100-dimensional latent noise vector $z \sim \mathcal{N}(0, I)$ into a $28 \times 28 \times 1$ image tensor.
- Construct a convolutional Discriminator network that classifies images as real ($1$) or synthetic ($0$).
- Implement custom adversarial training loops using `tf.GradientTape` and Adam optimizers.
- Track loss equilibrium, observe the progressive emergence of apparel structures across epochs, and prevent common failure modes like mode collapse.

---

## 🔬 DCGAN Architecture Specifications

### 1. The Generator Network ($G$)
The Generator transforms latent noise into a high-dimensional image using fractional-strided convolutions:
- **Input:** Latent vector $z \in \mathbb{R}^{100}$
- **Layer 1:** `Dense(7 * 7 * 256, use_bias=False)` $\to$ `BatchNormalization()` $\to$ `LeakyReLU(alpha=0.2)`
- **Reshape:** Reshape to tensor $(7, 7, 256)$
- **Layer 2:** `Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same')` $\to$ `BatchNorm` $\to$ `LeakyReLU`
- **Layer 3:** `Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same')` $\to$ `BatchNorm` $\to$ `LeakyReLU` (Upsamples to $14 \times 14 \times 64$)
- **Output Layer:** `Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh')` (Upsamples to $28 \times 28 \times 1$)

### 2. The Discriminator Network ($D$)
The Discriminator acts as a binary classifier using strided convolutions:
- **Input:** Grayscale image tensor $x \in [-1, 1]^{28 \times 28 \times 1}$
- **Layer 1:** `Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')` $\to$ `LeakyReLU(0.2)` $\to$ `Dropout(0.3)` (Downsamples to $14 \times 14 \times 64$)
- **Layer 2:** `Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')` $\to$ `LeakyReLU(0.2)` $\to$ `Dropout(0.3)` (Downsamples to $7 \times 7 \times 128$)
- **Classifier:** `Flatten()` $\to$ `Dense(1)` (Linear logits)

---

## 📐 Mathematical Formulation

The minimax objective function optimized during training is:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- **Discriminator Loss:**
  $$\mathcal{L}_D = -\frac{1}{m} \sum_{i=1}^m \left[ \log D(x^{(i)}) + \log(1 - D(G(z^{(i)}))) \right]$$
- **Generator Loss (Non-saturating heuristic):**
  $$\mathcal{L}_G = -\frac{1}{m} \sum_{i=1}^m \log D(G(z^{(i)}))$$

---

## ⚙️ Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Latent Vector Dimension ($z$) | `100` | Standard continuous latent manifold capacity |
| Batch Size | `256` | Balances gradient stability and memory footprint |
| Generator Optimizer | Adam (`lr=1e-4`, $\beta_1=0.5$) | Suppresses oscillatory momentum in minimax dynamics |
| Discriminator Optimizer | Adam (`lr=1e-4`, $\beta_1=0.5$) | Matches Generator learning rate for equilibrium |
| Activation Functions | LeakyReLU ($\alpha=0.2$) | Prevents "dying ReLU" neuron deactivation |
| Normalization | Batch Normalization | Stabilizes deep gradient flow across transposed convs |
| Image Scaling Range | $[-1, 1]$ | Aligns pixel intensity with Generator `tanh` range |
| Training Duration | `30 - 40` Epochs | Sufficient for sharp contour convergence |

---

## 📈 Training Dynamics & Visual Progression

```
[Epoch 01 - 05] ──► Pure Gaussian noise & diffuse grey blobs
[Epoch 06 - 12] ──► Coarse silhouettes of sandals, shoes, and shirts emerge
[Epoch 13 - 22] ──► Distinct categories separate; handles on handbags, soles on boots
[Epoch 23 - 35] ──► Sharp textures, symmetry in pullovers, crisp high-frequency boundaries
```

### Empirical Findings
- **Training Stability:** Both $\mathcal{L}_D$ and $\mathcal{L}_G$ stabilized around balanced values, demonstrating that the discriminator did not overpower the generator early in training.
- **Diversity:** The generator demonstrated strong intra-class variance across multiple apparel categories without experiencing catastrophic mode collapse.

---

## 💻 How to Run

### Google Colab (Recommended for GPU)
1. Open [`DCGAN_Fashion_MNIST_TensorFlow_Colab Ass 4.ipynb`](./DCGAN_Fashion_MNIST_TensorFlow_Colab%20Ass%204.ipynb).
2. Set runtime type to **GPU** (`Runtime > Change runtime type > T4 GPU`).
3. Run all cells sequentially.

### Local Execution
```bash
jupyter notebook "DCGAN_Fashion_MNIST_TensorFlow_Colab Ass 4.ipynb"
```

---

## 📚 References
> Radford, A., Metz, L., & Chintala, S. (2015). **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** *arXiv preprint arXiv:1511.06434*.

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
