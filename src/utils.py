"""Small helpers: seeding, saving files, plots, sample predictions."""
import csv
import json
import random

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_history_csv(history, path):
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(epochs, [h["train_dice"] for h in history], marker="o", label="train")
    ax[1].plot(epochs, [h["val_dice"] for h in history], marker="o", label="val")
    ax[1].set(title="Dice", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def collect_samples(model, dataset, device, n=4):
    """Returns [(image uint8 HxWx3, gt HxW, pred HxW), ...]"""
    model.eval()
    samples = []
    for i in range(min(n, len(dataset))):
        img, mask = dataset[i]
        pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        samples.append((img_np, mask.numpy().astype(np.uint8), pred.astype(np.uint8)))
    return samples


def save_prediction_grid(samples, path):
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3.3 * len(samples)))
    axes = np.atleast_2d(axes)
    for row, (img, gt, pred) in zip(axes, samples):
        row[0].imshow(img); row[0].set_title("Input")
        row[1].imshow(gt, cmap="gray"); row[1].set_title("True mask")
        row[2].imshow(pred, cmap="gray"); row[2].set_title("Prediction")
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
