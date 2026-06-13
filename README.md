# Industrial Surface Defect Detection — A Controlled Backbone Comparison

Unsupervised, feature-based anomaly detection (PatchCore-style) for industrial visual
inspection. Models are trained on **normal images only** — no defect labels — and flag
anything that deviates from "normal", with a heatmap localizing the defect.

This repository is a **controlled comparative study**: the same minimal pipeline is run
with three different feature backbones (**ResNet50**, **DINO ViT-S/8**, **DINOv2 ViT-S/14**)
across all 15 categories of **MVTec-AD**, changing *only the backbone* so the comparison is
fair. It is not a new method — it builds on [PatchCore](https://arxiv.org/abs/2106.08265)
and [AnomalyDINO](https://arxiv.org/abs/2405.14529) and asks: *how much does the feature
representation alone determine anomaly-detection quality?*

## Highlights

- 🧩 **One pipeline, swappable backbone** — feature extractor → memory bank (k-center
  greedy coreset) → nearest-neighbor distance → decision threshold + smoothed heatmap.
- 🧪 **Leakage-free evaluation** — the decision threshold is calibrated on a *held-out* set
  of good images (μ + 2σ), never on test labels. More rigorous than picking the
  F1-optimal threshold on the test set, which many reports do.
- 📊 **Full MVTec-AD (15 categories)**, image-level AUROC + F1, with per-category and
  threshold-policy ablations.
- 🟢 **Competitive with SOTA, minimal design** — DINOv2 reaches ~**99.3% mean image-AUROC**
  with no local-aggregation or score re-weighting.

## Results (full MVTec-AD, 15 categories)

| Backbone | Model | Input | Mean AUROC | Mean F1 |
|----------|-------|------:|-----------:|--------:|
| ResNet50 | `resnet50` (layer2+3) | 224 | 0.9765 | 0.9155 |
| DINO ViT-S/8 | `vit_small_patch8_224.dino` | 224 | 0.9855 | 0.9472 |
| **DINOv2 ViT-S/14** | `vit_small_patch14_dinov2.lvd142m` | 518 | **0.9930** | **0.9615** |

Quality improves **monotonically** ResNet → DINO → DINOv2 on both metrics — clearest on the
hard, fine-grained categories (e.g. `screw` F1 0.54 → 0.72 → 0.87). Full per-category tables
and discussion are in [`results_comparison.md`](results_comparison.md).

*For context: PatchCore ≈ 99.1%, EfficientAD ≈ 99.1%, SimpleNet ≈ 99.6% mean image-AUROC.
These use different backbones/components, so the numbers are not strictly apples-to-apples.*

## Method

```
Image ──► Feature backbone ──► patch features (L2-normalized)
                                      │
                                      ▼
                         Memory Bank  (built from NORMAL images only,
                                       reduced to ~10% via k-center greedy coreset)
                                      │
        query patch ──► nearest-neighbor distance to memory bank
                                      │
                                      ▼
            patch score map ──► Gaussian smoothing ──► image score (max)
                                      │
                                      ▼
        threshold = μ + 2σ of held-out-good scores  ──►  OK / NG  +  heatmap
```

## Repository structure

| File | Purpose |
|------|---------|
| [`anomaly_system.py`](anomaly_system.py) | Core library: backbones, memory bank + coreset, scoring, threshold |
| [`evaluate_backbones.py`](evaluate_backbones.py) | Main experiment — runs one backbone over all 15 categories |
| [`tune_threshold.py`](tune_threshold.py) | Threshold-policy analysis (sigma / quantile) + oracle ceiling |
| [`Final_Code.ipynb`](Final_Code.ipynb) | End-to-end annotated demo on a single category |
| `results_*.md` | Per-backbone result tables + the combined comparison |
| [`LEARNING_NOTES.md`](LEARNING_NOTES.md) | Study notes on the concepts (Vietnamese) |

## Setup

```bash
# 1. (GPU) install the CUDA build of PyTorch that matches your driver, e.g. CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU-only users can skip this and just install the requirements below.

# 2. install the rest
pip install -r requirements.txt
```

## Dataset

Download **MVTec-AD** from the official site:
<https://www.mvtec.com/company/research/datasets/mvtec-ad> and extract it so the layout is:

```
mvtec_anomaly_detection/
  bottle/  cable/  capsule/  ...  zipper/
    train/good/*.png
    test/<defect_type>/*.png
    test/good/*.png
```

The dataset is **not** included in this repo (it is git-ignored — ~5 GB).

## Usage

**Run the full comparison** (writes `results_<backbone>.md`):

```bash
python evaluate_backbones.py resnet50
python evaluate_backbones.py dino_vit
python evaluate_backbones.py dinov2
# optional threshold-policy override:  python evaluate_backbones.py dinov2 sigma 2.0
```

**Analyze the threshold for one category** (precision/recall/F1 across policies):

```bash
python tune_threshold.py dinov2 screw
```

**Interactive demo:** open `Final_Code.ipynb` and run all cells (set the category via
`config.dataset_name` in the first cell).

## Key design decisions

- **No data leakage.** Build set and threshold-calibration set are disjoint (80/20 split of
  `train/good`); the threshold never sees test labels.
- **Only the backbone changes** between runs — the cleanest way to attribute differences to
  the feature representation.
- **Minimal pipeline on purpose.** Local-neighborhood aggregation and score re-weighting
  (PatchCore) are intentionally left out of the main comparison and treated as future
  ablations, so every component is easy to justify.

## Limitations

- Image-level detection only — **no pixel-level AUROC / PRO** localization metric yet.
- Single run per setting (no multi-seed mean ± std).
- Not a novel method: DINOv2 + patch memory bank was introduced by AnomalyDINO.

## References

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Damm et al., *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2*, 2024.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.

---

*Built as an undergraduate thesis project. The contribution is a controlled comparative
study and a rigorous, leakage-free evaluation protocol — not a new state-of-the-art method.*
