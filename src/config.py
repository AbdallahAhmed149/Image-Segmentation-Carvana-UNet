"""Central configuration for the whole project (single source of truth)."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- data ----
    image_dir: str = "data/train/train"
    mask_dir: str = "data/train_masks/train_masks"
    image_size: int = 160          # use 512 on a strong GPU
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 2

    # ---- model ----
    in_channels: int = 3
    n_classes: int = 2             # background (0) / car (1)

    # ---- training ----
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 5
    seed: int = 42

    # ---- output folders ----
    metrics_dir: str = "Metrics"
    models_dir: str = "Models"

    # ---- Weights & Biases ----
    wandb_project: str = "carvana-unet-segmentation"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"     # online | offline | disabled
    model_artifact_name: str = "carvana-unet"
    registry_path: Optional[str] = None   # e.g. "wandb-registry-model/carvana-unet"
    promote: bool = True           # tag the uploaded model with the "production" alias

    def path(self, p: str) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        return asdict(self)
