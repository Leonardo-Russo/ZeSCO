# ZeSCO: Zero-Shot Cross-View Orientation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

ZeSCO (Zero-Shot Cross-View Orientation) is a novel zero-shot method for estimating camera orientation by matching ground-level panoramic images with aerial/satellite images. The method leverages pretrained vision transformers (DINOv2, DINOv3, CLIP) to align cross-view perspectives without requiring any task-specific training.

## 🎯 Overview

Cross-view geo-localization is challenging due to the dramatic viewpoint differences between ground and aerial imagery. ZeSCO addresses orientation estimation by:

1. **Extracting semantic features** from both ground and aerial images using pretrained vision transformers
2. **Applying intelligent filtering** to remove sky regions and weight features by depth
3. **Matching directional patterns** between vertical slices of ground panoramas and radial directions in aerial images
4. **Finding optimal alignment** through exhaustive search over possible orientations

### Key Features

- 🔥 **Zero-shot approach** - No training required, works out-of-the-box
- 🌍 **Multiple backbone support** - DINOv2, DINOv3 (ground/satellite variants), CLIP
- 🎯 **Depth-aware matching** - Incorporates monocular depth estimation for better feature weighting
- ☁️ **Sky filtering** - Automatically removes sky regions that don't contribute to alignment
- 📊 **Multi-dataset support** - Works with CVUSA, CV-Cities, CV-Global datasets
- 🔬 **Comprehensive evaluation** - Generates detailed visualizations and error statistics

## 🏗️ Architecture

### Method Pipeline

```
Ground Image (Panorama)                Aerial Image (Satellite)
          ↓                                       ↓
       Backbone                                Backbone
       (DINOv3)                                (DINOv3)
          ↓                                       ↓
     Patch Tokens                            Patch Tokens
    (Grid: 14×14)                           (Grid: 14×14)
          ↓                                       │
     Sky Filter                                   │
   Depth Estimation                               │
          ↓                                       ↓
   Vertical Averaging                     Radial Averaging
    (Column-wise)                            (Ray-wise)
          ↓                                       ↓
    Depth-weighted                         Distance-weighted
     Aggregation                             Aggregation
          │                                       │
          └──────────→ Cosine Matching ←──────────┘
                              ↓
                      Best Orientation
```


## 🚀 Usage

```bash
python apply_method.py \
    --name experiment_name \
    --backbone dinov3_crossview \
    --dataset cvglobal \
    --debug False \
    --save_mode separate
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--name` | `-n` | `untitled` | Experiment name for saving results |
| `--backbone` | `-b` | `dinov3` | Vision model backbone (`dinov2`, `dinov3`, `dinov3_crossview`, `clip`, `resnet50`) |
| `--loss` | `-l` | `cosine_similarity` | Loss function (`cosine_similarity`, `cosine_similarity_custom`) |
| `--dataset` | `-d` | `cvglobal` | Dataset to use (`cvusa`, `cvglobal`, `CITIES`, `GLOBAL`) |
| `--debug` | `-db` | `False` | Enable debug visualizations |
| `--save_mode` | `-m` | `separate` | Figure saving mode (`combined`, `separate`, `both`) |

### Dataset Preparation

```
CVUSA/
├── streetview/
│   └── panos/
│       ├── 000000.jpg
│       ├── 000001.jpg
│       └── ...
└── bingmap/
    └── 19/
        ├── 000000.jpg
        ├── 000001.jpg
        └── ...
```

## 🧪 Evaluation Metrics

- **Mean Delta Yaw Error**: Average absolute difference between predicted and ground-truth orientation
- **Standard Deviation**: Measure of prediction consistency
- **Median Delta Yaw Error**: Robust central tendency measure
- **Confidence Score**: Z-score based on distance distribution

## 🔬 Technical Details

### Feature Dimensions

| Backbone | Patch Size | Grid Size (224×224) | Feature Dim |
|----------|-----------|---------------------|-------------|
| DINOv2   | 14        | 16×16               | 768         |
| DINOv3   | 16        | 14×14               | 1024        |
| CLIP     | 16        | 14×14               | 768         |

### Depth Weighting Schemes (for Ground Images)

**Foreground Weights**:

$$
w_{\text{fore}}(d) = d
$$

**Middleground Weights**:

$$
w_{\text{mid}}(d) = \begin{cases} 
\frac{1}{\tau} d & \text{if } d \leq 0.5 \\\\
\frac{1-d}{d} & \text{otherwise}
\end{cases}
$$

**Background Weights**:
$$
w_{\text{back}}(d) = 1 - d
$$

Where $d$ is the normalized depth value and $\tau$ is a threshold parameter.

### Similarity Metric

Cosine similarity between aggregated feature vectors:

$$\text{loss} = 1 - \frac{1}{N} \sum_{i=1}^{N} \frac{v_i^{\text{ground}} \cdot v_i^{\text{aerial}}}{\|v_i^{\text{ground}}\| \|v_i^{\text{aerial}}\|}$$

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{zesco2025,
  author = {Leonardo Russo},
  title = {ZeSCO: Zero-Shot Cross-View Orientation Estimation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Leonardo-Russo/ZeSCO}}
}
```

## 🙏 Acknowledgments

This project builds upon several excellent works:

- **DINOv2/DINOv3**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **CLIP**: [openai/CLIP](https://github.com/openai/CLIP)
- **Depth Anything**: [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- **Sky Removal**: [OpenDroneMap/SkyRemoval](https://github.com/OpenDroneMap/SkyRemoval)

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [leonardo.rxsso@gmail.com](mailto:leonardo.rxsso@gmail.com).

---

**Repository:** [Leonardo-Russo/CroDINO](https://github.com/Leonardo-Russo/CroDINO)