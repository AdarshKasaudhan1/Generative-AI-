# Assignment 6: AI-Based Image Style Transfer (CycleGAN vs. Neural Style Transfer)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-Latest-red.svg)](https://pytorch.org/vision/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

This study implements and rigorously evaluates two foundational deep learning approaches for artistic image synthesis and appearance transformation applied to the identical source landscape photograph:

1. **Unpaired Image-to-Image Domain Translation (CycleGAN):** Employs an adversarial cycle-consistent architecture to map real-world photographs into the impressionistic painting domain of **Claude Monet**.
2. **Optimization-Based Neural Style Transfer (VGG-19 NST):** Leverages deep convolutional feature representations from Gatys et al. to fuse the spatial semantics of the landscape with the vibrant textures and brushstrokes of Vincent van Gogh's ***The Starry Night***.

---

## 🎯 Objectives

- Comprehend the mathematical divergence between parametric domain mapping (CycleGAN) and instance-level gradient-based feature optimization (NST).
- Implement a pretrained CycleGAN generator in PyTorch for high-fidelity photo-to-artwork translation.
- Construct a modular VGG-19 feature extraction pipeline to compute Gram matrices across multi-scale convolutional layers.
- Formulate and optimize the joint Content-Style loss surface using the Adam optimizer.
- Perform a critical side-by-side analysis contrasting content fidelity, structural integrity, and artistic abstraction.

---

## 🔬 Mathematical Formulations

### 1. CycleGAN: Unpaired Domain Translation
CycleGAN learns mappings between domains $X$ (Photographs) and $Y$ (Monet Paintings) without paired training examples:
- **Adversarial Loss:** Ensures generated images $G(x)$ match the target empirical distribution:
  $$\mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) = \mathbb{E}_{y}[\log D_Y(y)] + \mathbb{E}_{x}[\log(1 - D_Y(G(x)))]$$
- **Cycle-Consistency Loss:** Guarantees invertible bi-directional translation ($F(G(x)) \approx x$ and $G(F(y)) \approx y$):
  $$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y}[\|G(F(y)) - y\|_1]$$
- **Total Objective:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}} + \lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}}$

### 2. Gatys Neural Style Transfer (VGG-19)
- **Content Loss:** Evaluated at deeper layer $l = \text{conv4\_2}$ where high-level semantic layout is encoded:
  $$\mathcal{L}_{\text{content}}(p, x, l) = \frac{1}{2} \sum_{i, j} \left( F_{ij}^l(x) - P_{ij}^l(p) \right)^2$$
- **Style Loss:** Captures stationary feature statistics using **Gram Matrices** $G^l \in \mathbb{R}^{N_l \times N_l}$:
  $$G_{ij}^l = \sum_{k=1}^{M_l} F_{ik}^l F_{jk}^l$$
  $$\mathcal{L}_{\text{style}}(a, x) = \sum_{l \in L} w_l \frac{1}{4 N_l^2 M_l^2} \sum_{i, j} \left( G_{ij}^l(x) - A_{ij}^l(a) \right)^2$$
- **Total Joint Loss:** Optimized directly with respect to input pixel values:
  $$\mathcal{L}_{\text{NST}}(p, a, x) = \alpha \mathcal{L}_{\text{content}}(p, x) + \beta \mathcal{L}_{\text{style}}(a, x)$$

---

## 🛠️ Comparative System Architecture

```
               ┌────────────────────────────────────────────────────────┐
               │         Input Content: Landscape Photograph            │
               └──────────────────────────┬─────────────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│     Method 1: CycleGAN        │                       │ Method 2: VGG-19 Neural Style │
├───────────────────────────────┤                       ├───────────────────────────────┤
│ • Pretrained Generator G      │                       │ • Pretrained VGG-19 Backbone  │
│ • Domain-level Mapping        │                       │ • Target: Van Gogh Starry Night│
│ • Feedforward inference       │                       │ • Multi-scale Gram Matrices   │
│ • Cycle-consistent constraint │                       │ • Iterative Adam optimization │
└──────────────┬────────────────┘                       └──────────────┬────────────────┘
               ▼                                                       ▼
  [Monet-Domain Landscape]                                [Starry Night-Stylized Photo]
```

---

## 📊 Empirical Comparison & Discussion

| Evaluation Metric | CycleGAN (Monet Generator) | VGG-19 Neural Style Transfer (Van Gogh) |
|---|---|---|
| **Mechanism** | Feedforward generative neural network | Test-time gradient descent on image pixels |
| **Inference Speed** | **Real-time** ($\approx 50\text{ ms}$ on GPU) | **Iterative** ($30\text{--}60\text{ seconds}$ on GPU) |
| **Input Requirements** | Single content image (Model trained offline) | Content image + Explicit style image reference |
| **Content Preservation** | Retains general scene layout; alters color palette and lighting globally | High geometric fidelity; content loss strictly preserves edge boundaries |
| **Style Fidelity** | Captures generalized brush texture and impressionistic lighting | Faithfully transfers vivid swirls, impasto textures, and localized color palettes |
| **Flexibility** | Fixed to the trained target domain ($X \to Y$) | Arbitrary style: Can transfer any painting instantly |

### Key Experimental Insights
- **Content Preservation:** VGG-19 NST strictly constrains structural drift through explicit MSE penalty on deeper feature maps. In contrast, CycleGAN transforms color schemes across sky, vegetation, and water based on global domain statistics.
- **Visual Realism vs. Artistry:** CycleGAN outputs resemble plausible paintings painted in Monet's plein-air style. VGG-19 produces dramatic artistic hybrids that fuse recognizable brushstrokes with photographic structures.

---

## 💻 How to Run

### Google Colab (Recommended)
1. Open [`GEN_AI_ASSIGNMENT_6.ipynb`](./GEN_AI_ASSIGNMENT_6.ipynb).
2. Ensure GPU acceleration is active (`Runtime > Change runtime type > GPU`).
3. Execute setup cells to clone necessary CycleGAN submodules and download pretrained checkpoints.
4. Execute both pipeline sections and inspect the final 3-panel comparison figure.

---

## 📚 References
1. Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.** *IEEE ICCV 2017*.
2. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). **Image Style Transfer Using Convolutional Neural Networks.** *IEEE CVPR 2016*.

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
