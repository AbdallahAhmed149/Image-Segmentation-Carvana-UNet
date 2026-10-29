# Carvana Image Segmentation with U-Net

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/AbdallahAhmed149/Image-Segmentation-Carvana-UNet)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Carvana-4285F4)](https://www.kaggle.com/competitions/carvana-image-masking-challenge)

A production-oriented binary image-segmentation pipeline for extracting vehicle silhouettes from the [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge) dataset. The project includes deterministic dataset splitting, train-only augmentation, a compact U-Net model, checkpoint selection, quantitative evaluation, prediction visualizations, and optional Weights & Biases experiment/model tracking.

## Abstract / Overview

The system maps an RGB vehicle image to a pixel-level binary mask with two classes: `background` and `car`. A U-Net encoder-decoder uses convolutional feature extraction and skip connections to preserve spatial detail while reconstructing the segmentation mask.

The repository is designed as a reproducible engineering baseline rather than a notebook-only experiment:

- Fixed seed and deterministic file splitting into train, validation, and test subsets.
- Nearest-neighbor mask resizing to preserve class labels.
- Horizontal-flip augmentation applied only to training data.
- Best-checkpoint selection using validation Dice score.
- Dataset-level Dice, IoU, precision, recall, and pixel accuracy computed from accumulated pixel counts.
- Exported metrics, training history, plots, prediction grids, and model metadata.

## Architecture & Pipeline

```text
Carvana JPG images + GIF masks
              │
              ▼
  File discovery and seeded 80/10/10 split
              │
              ▼
 Resize images to 160×160 ── resize masks with nearest neighbor
              │
              ▼
 Train-only horizontal flip augmentation
              │
              ▼
 RGB tensor ──► U-Net encoder ──► bottleneck ──► decoder + skip connections
              │                                      │
              └──────── CrossEntropyLoss ◄───────────┘
                                                     │
                                                     ▼
                            class logits → argmax → binary car mask
                                                     │
                             Dice / IoU / precision / recall / accuracy
```

### Model

The implemented U-Net contains:

- Three encoder stages with `3×3 Conv → BatchNorm → ReLU` blocks and max pooling.
- Bilinear interpolation in the decoder with skip concatenations from matching encoder resolutions.
- A `1×1` convolution producing two class logits per pixel.
- Approximately **7.79 million trainable parameters**.

### Training flow

`train.py` loads the dataset, trains with Adam and `CrossEntropyLoss`, applies a cosine-annealing learning-rate schedule, evaluates after every epoch, saves the best validation checkpoint, and evaluates that checkpoint once on the untouched test split. Results are written to `Metrics/` and model metadata/checkpoints to `Models/`.

### Inference flow

`predict.py` loads `Models/model_config.json` and `Models/best_model.pth`, resizes an input image to the training resolution, predicts the car class mask, restores the original image size, and writes a green overlay image. A W&B model artifact can be used instead of the local model directory.

## Performance Metrics

The following results are recorded in [`Metrics/final_metrics.json`](Metrics/final_metrics.json) from a CUDA run with seed `42`, 10 epochs, batch size `16`, and 160×160 inputs.

| Metric | Result |
|---|---:|
| Validation Dice | **0.9933** |
| Test Dice | **0.9933** |
| Test IoU | **0.9866** |
| Test precision | **0.9935** |
| Test recall | **0.9930** |
| Test pixel accuracy | **0.9972** |
| Training examples | **4,072** |
| Validation examples | **508** |
| Test examples | **508** |
| Trainable parameters | **7,788,610** |
| Best checkpoint size | **31.21 MB** |
| Recorded training time | **1,715.5 s (~28.6 min)** |
| Input resolution | **160×160** |

Inference latency and request-throughput benchmarks are not currently recorded by this repository; they depend on hardware, batch size, and deployment runtime.

## Tech Stack

- **Python 3.9+**
- **PyTorch** and **torchvision** for model definition, optimization, tensor operations, and image transforms
- **NumPy** and **Pillow** for mask processing and image I/O
- **Matplotlib** for training curves and prediction grids
- **tqdm** for progress reporting
- **Weights & Biases** for optional experiment tracking and model artifacts
- **Kaggle Carvana Image Masking Challenge** dataset

## Repository Structure

```text
.
├── train.py                 # Training, evaluation, checkpointing, and artifact export
├── predict.py               # Local or W&B-backed inference and overlay generation
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Central configuration and paths
│   ├── dataset.py           # Dataset loading, augmentation, and splitting
│   ├── engine.py            # Training and evaluation loops
│   ├── metrics.py           # Dataset-level segmentation metrics
│   ├── model.py             # U-Net implementation
│   ├── utils.py             # Seeding, plots, serialization, and samples
│   └── wandb_utils.py       # Optional W&B integration
├── Models/
│   └── model_config.json    # Reproducible model metadata
└── Metrics/
    ├── final_metrics.json   # Final evaluation report
    ├── history.csv          # Per-epoch metrics
    ├── history.json         # Per-epoch metrics in JSON
    └── *.png                # Curves and prediction visualizations
```

## Getting Started

### 1. Clone and create an environment

```bash
git clone https://github.com/AbdallahAhmed149/Image-Segmentation-Carvana-UNet.git
cd Image-Segmentation-Carvana-UNet

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build instead if GPU training is required; follow the selector at [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Download and arrange the dataset

Download the training images and masks from Kaggle and arrange them as follows:

```text
data/
├── train/
│   └── train/
│       ├── 00087a6bd4dc_01.jpg
│       └── ...
└── train_masks/
    └── train_masks/
        ├── 00087a6bd4dc_01_mask.gif
        └── ...
```

The default paths are configured in `src/config.py`:

```python
image_dir = "data/train/train"
mask_dir = "data/train_masks/train_masks"
```

A mask must match its image filename using the `*_mask.gif` convention.

### 3. Configure experiment tracking (optional)

W&B is enabled by default. To run locally without network access or an API key, use `--wandb-mode disabled`. For online tracking:

```bash
wandb login
export WANDB_PROJECT=carvana-unet-segmentation
```

No credentials are required when W&B logging is disabled.

### 4. Docker status

This repository currently does not include a `Dockerfile` or a containerized API service. The supported setup is the Python virtual environment above. If you add a Dockerfile for your deployment target, mount the dataset and model directories rather than copying private data into the image.

## Training & Inference

### Train with local logging only

```bash
python train.py \
  --epochs 10 \
  --batch-size 16 \
  --image-size 160 \
  --wandb-mode disabled
```

Useful overrides include:

```bash
python train.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.001 \
  --seed 42 \
  --wandb-mode online \
  --wandb-project carvana-unet-segmentation
```

Training writes:

- `Models/best_model.pth`
- `Models/model_config.json`
- `Metrics/final_metrics.json`
- `Metrics/history.csv` and `Metrics/history.json`
- `Metrics/training_curves.png`
- `Metrics/predictions.png`

### Run local inference

```bash
python predict.py \
  --image /path/to/car.jpg \
  --out prediction.png \
  --models-dir Models
```

The output is an RGB overlay with detected car pixels highlighted in green. The model automatically uses CUDA when available and otherwise falls back to CPU.

### Run inference from a W&B artifact

```bash
python predict.py \
  --image /path/to/car.jpg \
  --out prediction.png \
  --wandb-artifact ENTITY/carvana-unet-segmentation/carvana-unet:production
```

## Reproducibility Notes

- The default seed is `42`.
- Train, validation, and test files are split before loading with 80%/10%/10% proportions.
- Only the training split receives random horizontal flips.
- The recorded checkpoint used `Adam`, learning rate `1e-3`, `CosineAnnealingLR`, `CrossEntropyLoss`, and 10 epochs.
- Increase `--image-size` to `512` only when GPU memory permits; it changes the compute and accuracy characteristics.

## Contributing

1. Create a focused branch from `main`.
2. Keep changes small and include documentation for user-visible behavior.
3. Run a local training or inference smoke test when modifying the data, model, or engine code.
4. Do not commit datasets, credentials, W&B API keys, or generated private artifacts.
5. Open a pull request describing the change, validation performed, and any metric impact.

## License

This project is distributed under the [MIT License](LICENSE).
