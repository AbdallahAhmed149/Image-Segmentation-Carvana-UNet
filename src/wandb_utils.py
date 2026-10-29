"""Everything related to Weights & Biases."""
from pathlib import Path

import wandb

CLASS_LABELS = {0: "background", 1: "car"}


def init_wandb(cfg):
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config=cfg.to_dict(),
        mode=cfg.wandb_mode,
        job_type="train",
    )
    run.define_metric("epoch")
    for prefix in ("train/*", "val/*", "lr"):
        run.define_metric(prefix, step_metric="epoch")
    return run


def log_predictions(run, samples, key="test/predictions"):
    images = []
    for i, (img, gt, pred) in enumerate(samples):
        images.append(
            wandb.Image(
                img,
                caption=f"sample {i}",
                masks={
                    "prediction": {"mask_data": pred, "class_labels": CLASS_LABELS},
                    "ground_truth": {"mask_data": gt, "class_labels": CLASS_LABELS},
                },
            )
        )
    run.log({key: images})


def log_model_artifact(run, cfg, files, metadata, aliases):
    """Upload weights + config + metrics as a W&B model Artifact.
    `aliases` containing "production" is what marks the live version."""
    if cfg.wandb_mode == "disabled":
        return None

    artifact = wandb.Artifact(
        cfg.model_artifact_name,
        type="model",
        description="U-Net (Carvana) binary car segmentation",
        metadata=metadata,
    )
    for f in files:
        artifact.add_file(str(f), name=Path(f).name)

    logged = run.log_artifact(artifact, aliases=aliases)
    logged.wait()

    if cfg.registry_path:  # optional: link to the W&B Model Registry
        try:
            run.link_artifact(logged, cfg.registry_path, aliases=aliases)
        except Exception as e:  # registry path/permissions problem shouldn't kill the run
            print(f"[warn] could not link artifact to registry '{cfg.registry_path}': {e}")
    return logged
