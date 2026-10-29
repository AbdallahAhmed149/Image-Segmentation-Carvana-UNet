"""Train U-Net on Carvana, save metrics -> Metrics/, model -> Models/, log everything to W&B.

Example:
    python train.py --epochs 10 --wandb-entity my-team
    python train.py --wandb-mode disabled          # quick local run without W&B
"""
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.dataset import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import UNet
from src.utils import (collect_samples, count_parameters, plot_history, save_history_csv,
                       save_json, save_prediction_grid, set_seed)
from src.wandb_utils import init_wandb, log_model_artifact, log_predictions


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir"); p.add_argument("--mask-dir")
    p.add_argument("--image-size", type=int); p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int); p.add_argument("--lr", type=float)
    p.add_argument("--num-workers", type=int); p.add_argument("--seed", type=int)
    p.add_argument("--wandb-project"); p.add_argument("--wandb-entity")
    p.add_argument("--wandb-run-name"); p.add_argument("--registry-path")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.add_argument("--no-promote", action="store_true",
                   help="don't tag the uploaded model as 'production'")
    args = vars(p.parse_args())

    no_promote = args.pop("no_promote")
    cfg = Config()
    for k, v in args.items():
        if v is not None:
            setattr(cfg, k, v)
    if no_promote:
        cfg.promote = False
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_dir, models_dir = cfg.path(cfg.metrics_dir), cfg.path(cfg.models_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Carvana dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset),
             "test": len(test_loader.dataset)}
    print(f"Split sizes: {sizes}")

    model = UNet(cfg.in_channels, cfg.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    total_params, trainable_params = count_parameters(model)

    run = init_wandb(cfg)
    run.config.update({"total_params": total_params, "split_sizes": sizes,
                       "device": str(device)}, allow_val_change=True)

    best_path = models_dir / "best_model.pth"
    best_dice, best_epoch, history = -1.0, 0, []
    start = time.time()

    print(f"Starting training on {device}...")
    for epoch in range(1, cfg.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             step_callback=lambda l: run.log({"batch/train_loss": l}))
        va = evaluate(model, val_loader, criterion, device, desc="val")
        scheduler.step()

        row = {"epoch": epoch, "lr": lr,
               **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)

        run.log({"epoch": epoch, "lr": lr,
                 **{f"train/{k}": v for k, v in tr.items()},
                 **{f"val/{k}": v for k, v in va.items()}})

        if va["dice"] > best_dice:
            best_dice, best_epoch = va["dice"], epoch
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch}/{cfg.epochs} | train loss {tr['loss']:.4f} dice {tr['dice']:.4f}"
              f" | val loss {va['loss']:.4f} dice {va['dice']:.4f} iou {va['iou']:.4f}"
              f"{'  <- best' if epoch == best_epoch else ''}")

    train_time = time.time() - start

    # ---- final evaluation of the BEST checkpoint on the untouched test split ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(model, test_loader, criterion, device, desc="test")
    print("Test:", {k: round(v, 4) for k, v in test.items()})

    samples = collect_samples(model, test_loader.dataset, device, n=4)
    save_prediction_grid(samples, metrics_dir / "predictions.png")
    plot_history(history, metrics_dir / "training_curves.png")
    save_history_csv(history, metrics_dir / "history.csv")
    save_json(history, metrics_dir / "history.json")

    final_metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "training_time_sec": round(train_time, 1),
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "test": test,
        "dataset_sizes": sizes,
        "model": {
            "architecture": "UNet",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "weights_size_mb": round(best_path.stat().st_size / 1e6, 2),
        },
        "config": cfg.to_dict(),
    }
    save_json(final_metrics, metrics_dir / "final_metrics.json")

    # model_config.json = everything needed to rebuild + serve the model
    model_config = {
        "architecture": "UNet",
        "in_channels": cfg.in_channels,
        "n_classes": cfg.n_classes,
        "image_size": cfg.image_size,
        "class_labels": {0: "background", 1: "car"},
        "weights_file": best_path.name,
        "best_epoch": best_epoch,
        "val_dice": best_dice,
        "test_dice": test["dice"],
        "test_iou": test["iou"],
        "hyperparameters": {"lr": cfg.lr, "batch_size": cfg.batch_size,
                            "epochs": cfg.epochs, "optimizer": "Adam",
                            "scheduler": "CosineAnnealingLR", "loss": "CrossEntropyLoss"},
    }
    save_json(model_config, models_dir / "model_config.json")

    # ---- W&B: summary, prediction overlays, curves, model artifact ----
    run.summary.update({"best_epoch": best_epoch, "best_val_dice": best_dice,
                        **{f"test/{k}": v for k, v in test.items()},
                        "training_time_sec": train_time})
    log_predictions(run, samples)
    run.log({"charts/training_curves": __import__("wandb").Image(str(metrics_dir / "training_curves.png"))})

    aliases = ["latest", "best"] + (["production"] if cfg.promote else [])
    log_model_artifact(
        run, cfg,
        files=[best_path, models_dir / "model_config.json",
               metrics_dir / "final_metrics.json", metrics_dir / "training_curves.png"],
        metadata={"test_dice": test["dice"], "test_iou": test["iou"],
                  "val_dice": best_dice, "image_size": cfg.image_size,
                  "params": total_params},
        aliases=aliases,
    )
    run.finish()
    print(f"Done. Metrics -> {metrics_dir} | Model -> {models_dir}")


if __name__ == "__main__":
    main()
