"""Run inference with the model from local Models/ or straight from W&B (production alias).

    python predict.py --image car.jpg                                   # local Models/
    python predict.py --image car.jpg \
        --wandb-artifact my-entity/carvana-unet-segmentation/carvana-unet:production
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from src.config import PROJECT_ROOT
from src.model import UNet


def load_model(model_dir: Path, device):
    cfg = json.loads((model_dir / "model_config.json").read_text())
    model = UNet(cfg["in_channels"], cfg["n_classes"]).to(device)
    model.load_state_dict(torch.load(model_dir / cfg["weights_file"], map_location=device))
    model.eval()
    return model, cfg


@torch.no_grad()
def predict(model, cfg, image: Image.Image, device):
    orig_size = image.size  # (W, H)
    x = TF.resize(image.convert("RGB"), [cfg["image_size"]] * 2,
                  interpolation=InterpolationMode.BILINEAR)
    x = TF.to_tensor(x).unsqueeze(0).to(device)
    pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    mask = Image.fromarray(pred * 255).resize(orig_size, Image.NEAREST)
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="prediction.png")
    p.add_argument("--models-dir", default="Models")
    p.add_argument("--wandb-artifact", help="entity/project/carvana-unet:production")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb_artifact:
        import wandb
        model_dir = Path(wandb.Api().artifact(args.wandb_artifact, type="model").download())
    else:
        model_dir = PROJECT_ROOT / args.models_dir

    model, cfg = load_model(model_dir, device)
    image = Image.open(args.image)
    mask = predict(model, cfg, image, device)

    # overlay: car in green
    overlay = np.array(image.convert("RGB")).copy()
    m = np.array(mask) > 0
    overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(args.out)
    print(f"Saved overlay -> {args.out}")


if __name__ == "__main__":
    main()
