# Assignment 7: Inducing and Mitigating Mode Collapse in GANs via Label Smoothing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-Latest-red.svg)](https://pytorch.org/vision/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 Executive Summary

Mode collapse is one of the most notoriously pervasive failure modes in Generative Adversarial Networks (GANs), occurring when the generator maps diverse latent noise vectors to a narrow, homogenous subset of output modes. This laboratory study investigates the mechanics of mode collapse by:
1. **Intentionally Inducing Failure:** Destabilizing a PyTorch **DCGAN** trained on **MNIST** by stripping normalization layers, constricting network capacity, and inflating the learning rate.
2. **Systematically Resolving Collapse:** Introducing **One-Sided Label Smoothing** alongside stabilized gradient conditioning, restoring full modal diversity across all handwritten digit classes ($0\text{--}9$).

---

## 🎯 Objectives

- Gain deep mechanistic intuition regarding the delicate Nash equilibrium between Generator ($G$) and Discriminator ($D$).
- Empirically trigger mode collapse by intentionally degrading gradient stability.
- Formulate the mathematical rationale behind **One-Sided Label Smoothing** and its effect on discriminator logit saturation.
- Benchmark and visually contrast sample diversity using identical, fixed latent probe vectors $z_{\text{fixed}} \sim \mathcal{N}(0, I)$.

---

## 🔬 Theoretical Analysis & Formulations

### 1. The Anatomy of Mode Collapse
In standard minimax GAN optimization:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

When the discriminator becomes overly confident or the generator optimization trajectory degenerates:
$$\nabla_{\theta_g} \mathbb{E}_{z}[\log(1 - D(G(z)))] \to 0 \quad \text{or explodes.}$$
The generator discovers an isolated output sample $x^*$ that reliably fools $D$ and concentrates all its probability mass on $x^*$, ignoring the remainder of the empirical data manifold ($p_g(x) \ll p_{\text{data}}(x)$).

### 2. One-Sided Label Smoothing
Standard binary targets assign hard labels $y_{\text{real}} = 1.0$ and $y_{\text{fake}} = 0.0$. When $D$ achieves high accuracy, its optimal strategy drives output logits to infinity, leading to vanishing gradients for $G$.

By applying **One-Sided Label Smoothing**:
$$y_{\text{real}} = 1 - \alpha \quad (\text{e.g., } \alpha = 0.1 \implies y_{\text{real}} = 0.9)$$
$$y_{\text{fake}} = 0.0$$

> **Critical Rule:** Label smoothing must only be applied to *positive (real)* targets. Smoothing negative (fake) labels to $\beta > 0$ introduces pathological behavior where $D$ is incentivized to assign positive probability to generator samples, reinforcing incorrect generator modes.

---

## 🛠️ Controlled Experimental Setup

| Parameter / Component | Part A: Deliberately Unstable Model | Part B: Stabilized Improved Model |
|---|---|---|
| **Generator Normalization** | ❌ **Removed** (No Batch Normalization) | ✅ **Batch Normalization** restored |
| **Generator Capacity** | Constricted feature channels | Full multi-layer convolutional capacity |
| **Learning Rate** | Inflated ($> 10\times$ optimal) | Optimized ($2 \times 10^{-4}$) |
| **Real Target Label ($y_{\text{real}}$)** | `1.0` (Hard target) | `0.9` (One-sided smoothed target) |
| **Fake Target Label ($y_{\text{fake}}$)** | `0.0` | `0.0` |
| **Latent Vector ($z$)** | $100\text{-d}$ Gaussian noise | Identical fixed $100\text{-d}$ Gaussian noise |
| **Training Epochs** | `15` Epochs | `15` Epochs |

---

## 📊 Empirical Observations & Visual Results

```
[Part A: Destabilized Generator]
   ├── Epochs 1-5: High loss oscillations; extreme discriminator dominance.
   ├── Epochs 6-15: Severe mode collapse occurs.
   └── Output: Identical probe vectors map almost exclusively to a single, degraded digit mode.

[Part B: Improved Generator with Label Smoothing]
   ├── Balanced adversarial loss trajectories; non-zero informative gradients.
   └── Output: Rich sample diversity covering all digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
```

### Key Analytical Takeaways
- **Discriminator Softening:** Changing $y_{\text{real}}$ from $1.0$ to $0.9$ prevents the discriminator from reaching infinite logits, ensuring smooth, bounded backpropagation gradients to $G$.
- **Batch Normalization Impact:** BatchNorm prevents internal covariate shift within the transposed convolutional layers of the generator, keeping gradient variance bounded across training iterations.

---

## 💻 How to Run

### Google Colab / Local Jupyter
1. Launch [`Creating_and_Fixing_Mode_Collapse_in_a_Generative_Adversarial_Network_(GAN)_Using_Label_Smoothing_7.ipynb`](./Creating_and_Fixing_Mode_Collapse_in_a_Generative_Adversarial_Network_(GAN)_Using_Label_Smoothing_7.ipynb).
2. Execute Part A to inspect the mode-collapsed image grid.
3. Execute Part B to train the stabilized model with label smoothing.
4. Review the comparative side-by-side grid using the identical latent seed.

---

## 📚 References
> Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). **Improved Techniques for Training GANs.** *Advances in Neural Information Processing Systems (NeurIPS 2016)*.

---

## 👤 Author
**Adarsh Kumar Gupta Kasaudhan**  
GitHub: [@AdarshKasaudhan1](https://github.com/AdarshKasaudhan1)
