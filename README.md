# Neural-Architectures-for-Image-Classification
# Counting Connected Components with CNNs & MLPs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]https://colab.research.google.com/drive/1PNNEF7W6Iszb6tj9_a5N2QOQqdl889G5?usp=sharing)

## Overview
Synthetic dataset of 32×32 binary images (1–3 shapes). Compared MLP, 2‑layer CNN, and 3‑layer CNN with batch norm. Best model: 92.9% test accuracy. Under a distribution shift (scaling, squiggles, noise, occlusions), the deep CNN dropped to 60%. Regularisation (dropout 0.6 + weight decay) recovered 69.3% shifted accuracy.

## Results (original test set)
| Model | Acc | Per‑class (1/2/3) |
|-------|-----|-------------------|
| MLP   | 72.7% | 86/65/66 |
| Baseline CNN | 91.8% | 98/91/86 |
| Enhanced CNN | 92.9% | 97/90/92 |

## Distribution Shift Performance
- Baseline CNN: 64.7% (94/60/40)
- Enhanced CNN: 60.0% (60/38/82)
- **Regularised Enhanced CNN**: **69.3%** (76/58/74)

## Running the notebook
Execute all cells. The dataset and shifted test set are generated inside the notebook (seeded for reproducibility).

## Dependencies
- Python 3.10+
- PyTorch, NumPy, SciPy, scikit‑learn, matplotlib, scikit‑image
