"""Train / evaluate loops."""
import torch
from tqdm import tqdm

from .metrics import SegmentationMeter


def train_one_epoch(model, loader, criterion, optimizer, device, step_callback=None):
    model.train()
    meter, total_loss = SegmentationMeter(), 0.0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        meter.update(logits.argmax(dim=1), masks)
        if step_callback:
            step_callback(loss.item())

    return {"loss": total_loss / len(loader.dataset), **meter.compute()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    meter, total_loss = SegmentationMeter(), 0.0

    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, masks).item() * images.size(0)
        meter.update(logits.argmax(dim=1), masks)

    return {"loss": total_loss / len(loader.dataset), **meter.compute()}
