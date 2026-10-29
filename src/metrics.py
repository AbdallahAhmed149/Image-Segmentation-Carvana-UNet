"""Segmentation metrics for the 'car' class (class 1)."""
import torch


class SegmentationMeter:
    """Accumulates TP/FP/FN/TN over the whole dataset, then computes metrics once.
    (More accurate than averaging per-batch Dice.)"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, preds, targets):
        p, t = preds == 1, targets == 1
        self.tp += (p & t).sum().item()
        self.fp += (p & ~t).sum().item()
        self.fn += (~p & t).sum().item()
        self.tn += (~p & ~t).sum().item()

    def compute(self, eps=1e-8):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = max(tp + fp + fn + tn, 1)
        return {
            "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
            "iou": (tp + eps) / (tp + fp + fn + eps),
            "precision": (tp + eps) / (tp + fp + eps),
            "recall": (tp + eps) / (tp + fn + eps),
            "pixel_acc": (tp + tn) / total,
        }
