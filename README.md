# 🚗 Image Segmentation on Carvana (U-Net)

Professional, end-to-end **binary image segmentation** project using **PyTorch** and a lightweight **U-Net** model to segment cars from the **Carvana Image Masking Challenge** dataset.

> Dataset: https://www.kaggle.com/competitions/carvana-image-masking-challenge

---

## ✨ Project Highlights

- 🧠 **U-Net** encoder–decoder architecture with skip connections
- 🧪 Training + evaluation loop included
- 📏 Uses **Dice score** to measure segmentation quality
- 🖼️ Simple visualization of **input / ground truth / prediction**
- ⚡ Uses GPU automatically if available

---

## 🗂️ Repository Structure

```text
.
├── Image_segmentation_UNet.py   # Main training + evaluation script
├── README.md
└── data/
    ├── train/
    │   └── train/               # Training images (jpg)
    └── train_masks/
        └── train_masks/         # Masks (gif) named *_mask.gif
```

---

## 📦 Requirements

- Python 3.9+ (recommended)
- PyTorch
- torchvision
- numpy
- Pillow
- matplotlib

Install dependencies:

```bash
pip install torch torchvision numpy pillow matplotlib
```

---

## ⬇️ Dataset Setup (Important)

1. Download the dataset from Kaggle:
   - https://www.kaggle.com/competitions/carvana-image-masking-challenge/data
2. Place files so they match the paths used in the script:

```python
IMAGE_DIR = "./data/train/train"
MASK_DIR  = "./data/train_masks/train_masks"
```

### ✅ Mask naming convention
The script expects masks to be named like:

- image: `001.jpg`
- mask:  `001_mask.gif`

---

## ▶️ How to Run

Run the training script:

```bash
python Image_segmentation_UNet.py
```

What the script does:

1. Loads the dataset
2. Splits into train/test (90% / 10%)
3. Trains U-Net
4. Evaluates with Dice score
5. Visualizes a prediction

---

## ⚙️ Configuration

You can edit these hyperparameters inside `Image_segmentation_UNet.py`:

- `BATCH_SIZE` (default: 4)
- `LR` learning rate (default: 0.001)
- `EPOCHS` (default: 3)
- Image resize (default: **160×160** for speed)

---

## 📊 Metric: Dice Score

Dice score is commonly used for segmentation:

\[
\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}
\]

Higher is better (100% is perfect). ✅

---

## 🖼️ Output / Visualization

After training, the script shows:

- Input image
- True mask
- Predicted mask

This makes it easy to sanity-check results quickly. 🔍

---

## 🧩 Notes & Possible Improvements

If you want to extend this project:

- 🚀 Add a deeper U-Net (more down/up blocks)
- 🧷 Use `BCEWithLogitsLoss` for binary segmentation (single output channel)
- 🧮 Track IoU in addition to Dice
- 💾 Save checkpoints and best model weights
- 🧪 Add a proper validation set (train/val/test)

---

## 🤝 Contributing

Pull requests and suggestions are welcome. If you find a bug or have an improvement idea, feel free to open an issue. 🛠️

---

## 📜 License

No license file is currently included. If you plan to share or reuse this code publicly, consider adding a license (e.g., MIT). 🧾
