# ZeSCO: Zero-Shot Cross-View Orientation Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ZeSCO (Zero-Shot Cross-View Orientation) is a novel zero-shot method for estimating camera orientation by matching ground-level panoramic images with aerial/satellite images. The method leverages pretrained vision transformers (DINOv2, DINOv3, CLIP) to align cross-view perspectives without requiring any task-specific training.

## 🎯 Overview

Cross-view geo-localization is challenging due to the dramatic viewpoint differences between ground and aerial imagery. ZeSCO addresses orientation estimation by:

1. **Extracting semantic features** from both ground and aerial images using pretrained vision transformers
2. **Applying intelligent filtering** to remove sky regions and weight features by depth
3. **Matching directional patterns** between vertical slices of ground panoramas and radial directions in aerial images
4. **Finding optimal alignment** through exhaustive search over possible orientations

### Key Features

- 🔥 **Zero-shot approach** - No training required, works out-of-the-box
- 🌍 **Multiple backbone support** - DINOv2, DINOv3 (ground/satellite variants), CLIP, ResNet50
- 🎯 **Depth-aware matching** - Incorporates monocular depth estimation for better feature weighting
- ☁️ **Sky filtering** - Automatically removes sky regions that don't contribute to alignment
- 📊 **Multi-dataset support** - Works with CVUSA, CV-Cities, CV-Global datasets
- 🔬 **Comprehensive evaluation** - Generates detailed visualizations and error statistics

## 🏗️ Architecture

### Method Pipeline

```
Ground Image (Panorama)          Aerial Image (Satellite)
       ↓                                  ↓
   Backbone                           Backbone
   (DINOv3)                          (DINOv3)
       ↓                                  ↓
   Patch Tokens                      Patch Tokens
   (Grid: 16×16)                     (Grid: 16×16)
       ↓                                  ↓
   Sky Filter ←─────────────────────────┘
   Depth Estimation
       ↓
   Vertical Averaging                Radial Averaging
   (Column-wise)                     (Ray-wise)
       ↓                                  ↓
   Depth-weighted                    Distance-weighted
   Aggregation                       Aggregation
       ↓                                  ↓
       └──────────→ Cosine Matching ←────┘
                         ↓
                  Best Orientation
```


## 🚀 Usage

### Basic Example

```bash
python apply_method.py \
    --name experiment_name \
    --backbone dinov3_crossview \
    --dataset cvglobal \
    --debug False \
    --save_mode separate
```

### Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--name` | `-n` | `untitled` | Experiment name for saving results |
| `--backbone` | `-b` | `dinov3` | Vision model backbone (`dinov2`, `dinov3`, `dinov3_crossview`, `clip`, `resnet50`) |
| `--loss` | `-l` | `cosine_similarity` | Loss function (`cosine_similarity`, `cosine_similarity_custom`) |
| `--dataset` | `-d` | `cvglobal` | Dataset to use (`cvusa`, `cvglobal`, `CITIES`, `GLOBAL`) |
| `--debug` | `-db` | `False` | Enable debug visualizations |
| `--save_mode` | `-m` | `separate` | Figure saving mode (`combined`, `separate`, `both`) |

### Dataset Preparation

#### CVUSA (Cross-View USA)

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

#### CV-Global

```
CVGlobal/
├── streetview/
│   └── panos/
│       ├── 00000.jpg
│       └── ...
└── bingmap/
    └── 19/
        ├── 00000.jpg
        └── ...
```

## 📊 Output

Results are saved to `results/<experiment_name>/`:

### Generated Files

- **`sample_*_combined.png`** - 2×2 grid showing:
  - Top-left: Ground panorama with original orientation
  - Top-right: Aerial image with predicted and ground-truth orientations
  - Bottom-left: Distance vs. orientation plot
  - Bottom-right: Aerial image with color-coded distance rays

- **`sample_*_ground.png`** - Ground image only
- **`sample_*_aerial.png`** - Aerial image with orientation markers
- **`sample_*_distance_plot.png`** - Distance curve
- **`sample_*_aerial_distances.png`** - Colored distance visualization

- **`delta_yaws_hist.png`** - Histogram of orientation errors
- **`delta_yaws.pkl`** - Pickled array of all delta yaw values
- **`info.txt`** - Statistical summary:

```
Delta Yaw Error Statistics
==============================

Total Samples: 1000

Error Metrics:
---------------
Mean Delta Yaw Error:       12.3456°
Standard Deviation:         8.9012°
Median Delta Yaw Error:     9.8765°
Minimum Delta Yaw Error:    0.1234°
Maximum Delta Yaw Error:    89.5678°
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
| ResNet50 | 32        | 7×7                 | 2048        |

### Depth Weighting Schemes

**Foreground Weights** (Ground):
$$w_{\text{fore}}(d) = d$$

**Middleground Weights** (Ground):
$$w_{\text{mid}}(d) = \begin{cases} 
\frac{1}{\tau} d & \text{if } d \leq 0.5 \\
\frac{1-d}{d} & \text{otherwise}
\end{cases}$$

**Background Weights** (Ground):
$$w_{\text{back}}(d) = 1 - d$$

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