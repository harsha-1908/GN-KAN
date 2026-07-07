# Grouped Non-linear Kolmogorov-Arnold Network (GN-KAN)
A pure PyTorch official implementation of the **Grouped Non-linear Kolmogorov-Arnold Network (GN-KAN)** architecture designed as an efficient, spatially aware, and highly expressive replacement for traditional Feed-Forward Networks (FFNs/MLPs) inside Vision Transformers and convolutional pipelines.

Traditional KAN implementations rely on heavy, memory-bound B-spline calculations with irregular memory indexing that bottlenecks modern accelerator hardware. GN-KAN factorizes this mathematical expressivity using standard, dense PyTorch operators—achieving the functional benefits of continuous grid spaces within standard asymptotic computational envelopes.

---

## 🚀 Key Architectural Features

GN-KAN factorizes the feed-forward projection step into four orthogonal dimensions of expressivity:

1. **Pointwise Expansion Mapping:** A dense $1\times1$ convolution expands the representation capacity and performs cross-channel feature interaction.
2. **Channel-wise Grouped Scaling Modulation ($\mathcal{S}$):** Instead of treating the expanded hidden dimensions as a massive, uniform pool of features, the tensor is factorized into isolated channel subspaces where learnable weights adaptively scale information flow.
3. **Localized Spatial Message Propagation ($\mathcal{D}$):** Integrates group-isolated depthwise spatial interaction right inside the FFN to capture localized spatial relationships without expanding the parameter count.
4. **Data-Dependent Adaptive Feature Gating ($\mathcal{G}$):** Modulates the refined spatial features using a parallel contextual gating network via a Sigmoid mask.

---

## 📂 Repository Structure

* `gnkan.py`: Defines the standalone `GNKAN` pure-PyTorch FFN block processing inputs of shape `(B, C, H, W)`.
* `sd.py`: One of the ablation modals.

---

## 📊 Algorithmic Complexity Profile

| Architecture | Asymptotic FLOPs Complexity | Parameter Footprint | Primary Hardware Profile |
| :--- | :--- | :--- | :--- |
| **Standard MLP** | $\mathcal{O}(2 \cdot N \cdot C \cdot C_{hid})$ | $2 \cdot C \cdot C_{hid}$ | Uniform capacity / Compute-bound |
| **Spline-based KAN** | $\mathcal{O}(N \cdot C \cdot C_{hid} \cdot G)$ | $C \cdot C_{hid} \cdot G$ | Severe memory bound / Indexing bottlenecks |
| **Our GN-KAN Framework**| $\mathcal{O}(2 \cdot N \cdot C \cdot C_{hid} + N \cdot C_{hid} \cdot k^2)$ | $2 \cdot C \cdot C_{hid} + C_{hid} \cdot b + C_{hid} \cdot k^2$ | Maximum GPU Tensor-Core efficiency |

---

## 🛠️ Usage Example

```python
import torch
from gnkan import GNKAN

# Instantiate the standalone GN-KAN Block
# Channels: 64, Hidden Channels (Internal Expansion): 256, Subspace Groups: 8
gn_kan_block = GNKAN(
    channels=64, 
    hidden_channels=256, 
    groups=8, 
    spatial_kernel=3, 
    use_gating=True
)

# Input tensor shape: (Batch Size, Channels, Height, Width)
input_tensor = torch.randn(16, 64, 32, 32)
output_tensor = gn_kan_block(input_tensor)

print(f"Input Shape:  {input_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}") # Preserves dimensions: (16, 64, 32, 32)
