"""Carvana dataset + train/val/test splitting."""
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode


class CarvanaDataset(Dataset):
    """Images are `xxx.jpg`, masks are `xxx_mask.gif` (255 = car)."""

    def __init__(self, image_dir, mask_dir, files, image_size=160, augment=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = self.files[index]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name.replace(".jpg", "_mask.gif"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        # augmentation is applied to the TRAIN split only
        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        return image, mask


def split_files(image_dir, val_split, test_split, seed):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test, n_val = int(n * test_split), int(n * val_split)
    test = files[:n_test]
    val = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
    return train, val, test


def get_datasets(cfg):
    image_dir = str(cfg.path(cfg.image_dir))
    mask_dir = str(cfg.path(cfg.mask_dir))
    train_f, val_f, test_f = split_files(image_dir, cfg.val_split, cfg.test_split, cfg.seed)
    mk = lambda files, aug: CarvanaDataset(image_dir, mask_dir, files, cfg.image_size, aug)
    return mk(train_f, True), mk(val_f, False), mk(test_f, False)


def get_dataloaders(cfg):
    train_ds, val_ds, test_ds = get_datasets(cfg)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    # small splits: no worker processes -> saves a lot of memory on Windows
    eval_kw = dict(batch_size=cfg.batch_size, shuffle=False,
                   num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, **eval_kw)
    test_loader = DataLoader(test_ds, **eval_kw)
    return train_loader, val_loader, test_loader
