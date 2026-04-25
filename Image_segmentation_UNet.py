# ( Image Segmentation on Carvana Dataset using U-Net )
# Dataset Link: https://www.kaggle.com/competitions/carvana-image-masking-challenge/data

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

# --- PART 1: The Dataset Class ---
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        # Carvana masks often have the same name but with '_mask.gif' appended
        # Check your files! If image is "001.jpg", mask might be "001_mask.gif"
        mask_filename = self.images[index].replace(".jpg", "_mask.gif")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        # Apply Transforms manually to keep image/mask in sync
        if self.transform:
            # Resize to smaller size for speed (160x160)
            # Use 512x512 if you have a powerful GPU (RTX 3060+)
            image = TF.resize(image, size=(160, 160))
            mask = TF.resize(mask, size=(160, 160), interpolation=Image.NEAREST)

            # Random Horizontal Flip (Data Augmentation)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Convert to Tensor
            image = TF.to_tensor(image)
            mask = np.array(mask)
            mask = torch.from_numpy(mask).long()

            # Carvana masks are 255 (white) for car. We need 1 for car.
            mask = (mask == 255).long()

        return image, mask

# --- PART 2: The U-Net Architecture ---
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # --- ENCODER (The Left Side) ---
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- DECODER (The Right Side) ---
        # We use 'Upsample' to increase size
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DoubleConv(256 + 512, 256)  # 256 from upsampled + 512 from skip

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DoubleConv(128 + 256, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = DoubleConv(64 + 128, 64)

        # Final 1x1 Conv to map to number of classes
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 1. Downward Path (Encoder)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # Bottleneck

        # 2. Upward Path (Decoder with Skip Connections)

        # Up 1
        x = self.up1(x4)
        # SKIP CONNECTION: We assume x and x3 are same size.
        # In real U-Net, we might need to pad if sizes differ slightly.
        # We concatenate along channel dimension (dim=1)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up1(x)

        # Up 2
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)

        # Up 3
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)

        # Final Classification
        logits = self.outc(x)
        return logits

# --- PART 3: Setup & Training ---

# 1. Configuration
IMAGE_DIR = "./data/train/train"
MASK_DIR = "./data/train_masks/train_masks"
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
print("Loading Carvana Dataset...")
full_dataset = CarvanaDataset(IMAGE_DIR, MASK_DIR, transform=True)

# Split into Train/Test (90% Train, 10% Test)
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 3. Model Setup
# Carvana has 2 classes: Background (0) and Car (1)
model = UNet(in_channels=3, n_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. Training Loop
print(f"Starting Training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()

        outpus = model(images)
        loss = criterion(outpus, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item()}"
            )

    print(f"Epoch {epoch+1} Complete | Avg Loss: {train_loss / len(train_loader):.4f}")

# 5. Testing Intersection of Union (IoU)
print("Testing IoU...")

def check_dice_score(loader, model):
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            output = model(x)
            preds = torch.argmax(output, dim=1)

            # Dice Formula: (2 * Intersection) / (Sum of pixels in Pred + Sum of pixels in True)
            # We only care about Class 1 (The Car)
            intersection = (preds * y).sum()
            union = preds.sum() + y.sum()

            # Add smooth term (1e-8) to avoid division by zero
            dice = (2.0 * intersection + 1e-8) / union
            dice_score += dice.item()

    print(f"Dice Score: {(dice_score / len(loader)) * 100:.4f}")

check_dice_score(test_loader, model)

# 6. Visualization
print("Visualizing results...")

model.eval()
with torch.no_grad():
    img, mask = test_dataset[0]  # Get one sample
    img_tensor = img.unsqueeze(0).to(DEVICE)  # Add batch dim

    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1).cpu().squeeze(0)  # Get class indices

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title("Input Image")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("True Mask")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("AI Prediction")
    plt.show()
