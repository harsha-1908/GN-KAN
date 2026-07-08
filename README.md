# Grouped Nonlinear Kolmogorov–Arnold Network (GN-KAN)

Official PyTorch implementation of **Grouped Nonlinear Kolmogorov–Arnold Networks (GN-KAN)**, a structured feed-forward architecture that enhances nonlinear representation learning through **channel-wise specialization, spatial refinement, and adaptive gating**.

GN-KAN is designed as an efficient alternative to conventional MLP- and KAN-based feed-forward networks and can be used in convolutional architectures as well as Vision Transformers.

---

## Overview

GN-KAN consists of four sequential operations:

1. **Channel Expansion**
   - Expands the input representation using a point-wise ($1\times1$) projection.

2. **Grouped Scaling ($\mathcal{S}$)**
   - Partitions expanded channels into groups and applies learnable group-specific scaling.

3. **Spatial Refinement ($\mathcal{D}$)**
   - Applies depthwise convolution to capture local spatial interactions.

4. **Adaptive Gating ($\mathcal{G}$)**
   - Learns data-dependent feature selection through a sigmoid gating mechanism before channel projection.

The resulting architecture provides structured nonlinear processing while remaining fully compatible with standard PyTorch operators.

---

## Repository Structure

```
GN-KAN/
│
├── experiments/
│   ├── exp01_mnist_ablation_100e.py
│   ├── exp02_fashionmnist_ablation_100e.py
│   ├── exp03_cifar10_ablation_100e.py
│   ├── exp04_braintumor_ablation_100e.py
│   ├── exp05_fashionmnist_ablation_100e.py
│   ├── exp06_cifar100_ablation_100e.py
│   ├── exp07_robustness_ablation.py
│   ├── exp12_data_efficiency_braintumor.py
│   ├── exp15_scaling_study_braintumor.py
│   └── exp16_efficiency_ablation.py
│
├── models/
│   ├── gnkan.py
│   ├── efficientkan.py
│   ├── mlp.py
│   └── sd.py
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/harsha-1908/GN-KAN.git
cd GN-KAN
```

Install dependencies

```bash
pip install torch torchvision pandas
```

---

## Running Experiments

Examples:

### MNIST

```bash
python experiments/exp01_mnist_ablation_100e.py
```

### CIFAR-10

```bash
python experiments/exp03_cifar10_ablation_100e.py
```

### CIFAR-100

```bash
python experiments/exp06_cifar100_ablation_100e.py
```

### Brain Tumor MRI

```bash
python experiments/exp04_braintumor_ablation_100e.py
```

### Data Efficiency

```bash
python experiments/exp12_data_efficiency_braintumor.py
```

### Model Scaling

```bash
python experiments/exp15_scaling_study_braintumor.py
```

### Robustness

```bash
python experiments/exp07_robustness_ablation.py
```

---

## GN-KAN Block

```python
import torch

from models.gnkan import GNKAN

model = GNKAN(
    channels=64,
    hidden_channels=256,
    groups=8
)

x = torch.randn(8,64,32,32)

y = model(x)

print(y.shape)
```

Output

```
torch.Size([8, 64, 32, 32])
```

---

## Experimental Evaluation

The paper evaluates GN-KAN against:

- Multilayer Perceptron (MLP)
- EfficientKAN

Experiments include

- Component ablation
- Data efficiency
- Robustness
- Model scaling
- Computational efficiency
- Statistical significance analysis

Benchmarks

- MNIST
- FashionMNIST
- FashionMNIST-V2
- CIFAR-10
- CIFAR-100
- Brain Tumor MRI (BRISC)

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{thilak2026gnkan,
  title={Grouped Nonlinear Kolmogorov--Arnold Networks for Structured Feed-Forward Learning},
  author={Thilak, Gurram Harshamanya and Jha, Rajib Kumar},
  journal={Under Review},
  year={2026}
}
```
# ⚠️ Disclaimer

This repository is released strictly for **academic and research purposes**.

---

## License
# Copy Right

Copyright (c) 2026  
Gurram Harshamanya Thilak [All Rights Reserved]


