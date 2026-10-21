# Assignment 5: Latent Space Interpolation & Image Synthesis Using StyleGAN2-ADA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-StyleGAN2--ADA-76B900.svg)](https://github.com/NVlabs/stylegan2-ada-pytorch)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This project conducts advanced latent space exploration using NVIDIA's state-of-the-art **StyleGAN2-ADA (Adaptive Discriminator Augmentation)** architecture. By probing the continuous latent geometry of a high-resolution generative model trained on human faces (FFHQ), we investigate:
1. **Latent Space Interpolation:** Smooth trajectory traversal between disparate latent vectors $z_1, z_2 \in \mathcal{Z}$.
2. **Latent Dimension Manipulation:** Targeted perturbation along individual latent coordinate axes to isolate semantic and perceptual attribute shifts.

---

## 🎯 Objectives

- Understand the architectural mechanics of StyleGAN2-ADA, specifically the mapping network $f: \mathcal{Z} \to \mathcal{W}$ and modulation of convolution weights.
- Perform continuous vector interpolation in latent space to synthesize morphing photorealistic facial transitions.
- Evaluate perceptual smoothness and linearity across intermediate generated instances.
- Perturb specific latent dimensions to analyze attribute entanglement vs. disentanglement.

---

## 🔬 Mathematical Formulations

### 1. StyleGAN2 Latent Spaces
- **Initial Latent Space ($\mathcal{Z}$):** $z \sim \mathcal{N}(0, I)$, where $z \in \mathbb{R}^{512}$.
- **Disentangled Latent Space ($\mathcal{W}$):** 8-layer MLP mapping network $w = f(z) \in \mathbb{R}^{512}$.
- **Synthesis Network ($G$):** Modulates convolution weight tensors using learned affine transformations of $w$.

### 2. Linear Interpolation (LERP) & Spherical Linear Interpolation (SLERP)
To generate a seamless sequence of intermediate frames between latent vectors $z_A$ and $z_B$ over steps $\alpha \in [0, 1]$:
$$z(\alpha) = (1 - \alpha) z_A + \alpha z_B$$

For high-dimensional spherical Gaussian vectors, **SLERP** preserves latent norm:
$$z_{\text{slerp}}(\alpha) = \frac{\sin((1 - \alpha)\Omega)}{\sin \Omega} z_A + \frac{\sin(\alpha \Omega)}{\sin \Omega} z_B, \quad \text{where } \Omega = \arccos\left(\frac{z_A \cdot z_B}{\|z_A\| \|z_B\|}\right)$$

### 3. Coordinate-wise Latent Perturbation
To assess the sensitivity of an individual latent coordinate $k$:
$$z' = z + \delta \cdot e_k, \quad \delta \in [-\delta_{\max}, \delta_{\max}]$$
where $e_k$ is the one-hot basis vector along the $k$-th dimension.

---

## 🛠️ Implementation Workflow

```
[Latent Vector A: z_A] ──────┐
                             ├──► [Linear Interpolation: z(α)] ──► [StyleGAN2-ADA Generator] ──► [Continuous Morphing Sequence]
[Latent Vector B: z_B] ──────┘
                                                                           ▲
[Base Vector z] ──► [Dimension Perturbation: z + δ · e_k] ─────────────────┘ ──► [Attribute Shift Analysis]
```

---

## 📊 Key Observations & Results

1. **Perceptual Smoothness:**  
   Interpolation in latent space yields continuous, photorealistic transformations. Facial structure, skin tone, lighting conditions, and hair morphology transition gradually without unnatural visual artifacts, confirming that the generator has learned a well-behaved continuous manifold.

2. **Entangled vs. Disentangled Representations:**  
   Perturbing a single dimension in $\mathcal{Z}$-space typically induces correlated changes (e.g., modifying age might subtly alter lighting or hair color), demonstrating the natural entanglement of raw Gaussian space $\mathcal{Z}$. This highlights why NVIDIA introduced the intermediate $\mathcal{W}$-space to linearize factor variation.

3. **High-Fidelity Synthesis:**  
   StyleGAN2-ADA demonstrates remarkable stability and detail, synthesizing realistic high-frequency facial textures such as individual hair strands, micro-expressions, and eye reflections.

---

## 💻 How to Run

### Google Colab (Recommended)
1. Open [`Latent_Space_Interpolation_and_Image_Generation_Using_StyleGAN2_ADAipynb.ipynb`](./Latent_Space_Interpolation_and_Image_Generation_Using_StyleGAN2_ADAipynb.ipynb).
2. Select a GPU runtime (`Runtime > Change runtime type > GPU`).
3. Execute the setup cells to load the official PyTorch StyleGAN2-ADA checkpoint.
4. Run the interpolation and perturbation experiments to view and save the generated image sequences.

---

## 📚 References
> Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020). **Training Generative Adversarial Networks with Limited Data.** *Advances in Neural Information Processing Systems (NeurIPS 2020)*.

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
