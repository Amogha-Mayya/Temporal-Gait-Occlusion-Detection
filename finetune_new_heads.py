#!/usr/bin/env python
"""
finetune_new_heads.py
----------------------
Fine-tune ONLY the two newly added heads (RegionHead, ConfidenceHead) on top
of the already-trained 60-epoch model, WITHOUT touching the backbone,
transformer, or the original detection/severity heads.

WHAT THIS SCRIPT DOES
-----------------------
1. Loads checkpoints/best_model.pth into a 4-head model
   (build_model_with_pretrained_base). The 2 original heads + backbone +
   transformer come in pretrained and frozen; region_head and
   confidence_head start randomly initialised.
2. Freezes backbone / transformer / det_head / sev_head (requires_grad=False,
   eval() mode — so BatchNorm/Dropout statistics from the 60-epoch run are
   not disturbed).
3. Trains region_head and confidence_head for a small number of epochs
   (default 8) on the SAME train split, using:
     - region_labels  (ground truth, generated automatically by the
       occlusion generator's compute_region_labels)
     - confidence_targets (constructed from the FROZEN det/sev head outputs
       vs. ground truth — see utils/confidence.py)
4. Saves a new checkpoint checkpoints/best_model_4heads.pth that contains
   ALL FOUR heads. Your original checkpoints/best_model.pth is NEVER
   overwritten or modified.

Because only two small MLPs (hidden_dim -> 128 -> {1 or 4}) receive
gradients, this finishes in a small fraction of the time the original
60-epoch full-model training took — no CNN or Transformer backward pass is
computed for the frozen modules' parameters.

Usage:
    python finetune_new_heads.py \\
        --config configs/config.yaml \\
        --base_checkpoint checkpoints/best_model.pth \\
        --epochs 8 \\
        --output checkpoints/best_model_4heads.pth
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.casia_dataset import build_dataset
from data.transforms import build_transforms
from models.model import build_model_with_pretrained_base
from utils.checkpoint import save_checkpoint
from utils.confidence import compute_confidence_targets
from utils.logger import Logger
from utils.metrics import RegionConfidenceMetricAccumulator
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune RegionHead + ConfidenceHead on a frozen backbone."
    )
    p.add_argument("--config",          type=str, default="configs/config.yaml")
    p.add_argument("--base_checkpoint", type=str, default="checkpoints/best_model.pth",
                   help="Original 2-head checkpoint to load backbone/transformer/det/sev from.")
    p.add_argument("--output",          type=str, default="checkpoints/best_model_4heads.pth",
                   help="Where to save the resulting 4-head checkpoint.")
    p.add_argument("--epochs",          type=int, default=8,
                   help="Number of head-only fine-tuning epochs (small, since only 2 MLPs train).")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate for the new heads (can be higher than full-model LR).")
    p.add_argument("--batch_size",      type=int, default=None,
                   help="Override training.batch_size from config if set.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    device_str = cfg["gpu"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    logger = Logger(
        log_file="outputs/finetune_heads.log",
        tensorboard_dir="outputs/tensorboard_finetune_heads",
    )
    logger.info(f"Device: {device}")
    logger.info(f"Base checkpoint: {args.base_checkpoint}")

    # ---- Build 4-head model with pretrained base --------------------------
    model = build_model_with_pretrained_base(
        cfg, args.base_checkpoint, device=str(device)
    ).to(device)

    # Freeze everything except region_head / confidence_head
    model.freeze_pretrained_components()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameters ({len(trainable)} tensors):")
    for n in trainable:
        logger.info(f"  {n}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_trainable:,} / Total: {n_total:,} "
                f"({100 * n_trainable / n_total:.3f}%)")

    # ---- Datasets -----------------------------------------------------------
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_transform = build_transforms(cfg, split="train")
    val_transform   = build_transforms(cfg, split="val")

    train_dataset = build_dataset(cfg, split="train", transform=train_transform)
    val_dataset   = build_dataset(cfg, split="val",   transform=val_transform)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- Optimizer: only the 2 new heads' parameters -----------------------
    new_head_params = (
        list(model.region_head.parameters())
        + list(model.confidence_head.parameters())
    )
    optimizer = torch.optim.AdamW(new_head_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"])

    best_macro_f1 = -1.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        # =====================================================================
        # TRAIN (region_head + confidence_head only)
        # =====================================================================
        model.region_head.train()
        model.confidence_head.train()
        # backbone/transformer/det_head/sev_head remain in eval() — set by
        # freeze_pretrained_components() and never toggled back.

        acc = RegionConfidenceMetricAccumulator()
        pbar = tqdm(train_loader, desc=f"[Finetune Epoch {epoch}]", leave=False)

        for batch in pbar:
            frames        = batch["frames"].to(device)
            det_labels    = batch["det_labels"].to(device)
            sev_labels    = batch["sev_labels"].to(device)
            region_labels = batch["region_labels"].to(device)   # (B, T, 4)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"]):
                # Frozen backbone/transformer/det/sev still run forward (no_grad
                # not strictly required since requires_grad=False already stops
                # gradient flow into them, but torch.no_grad() saves activation
                # memory for those frozen layers too).
                with torch.no_grad():
                    ctx_feats  = model.encode(frames)
                    det_logits = model.det_head(ctx_feats)
                    sev_preds  = model.sev_head(ctx_feats)
                    det_prob   = torch.sigmoid(det_logits)

                    conf_targets = compute_confidence_targets(
                        det_prob=det_prob,
                        det_labels=det_labels,
                        sev_preds=sev_preds,
                        sev_labels=sev_labels,
                    )

                # Only these two calls build a graph with requires_grad=True
                region_logits = model.region_head(ctx_feats)
                conf_preds    = model.confidence_head(ctx_feats)

                region_loss = model.region_loss_fn(region_logits, region_labels)
                conf_loss   = model.conf_loss_fn(conf_preds, conf_targets)
                total_loss  = (
                    model.lambda_region * region_loss
                    + model.lambda_conf * conf_loss
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(new_head_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            acc.update(
                region_logits.detach(), region_labels,
                conf_preds.detach(), conf_targets,
                total_loss=total_loss.item(),
                region_loss=region_loss.item(),
                conf_loss=conf_loss.item(),
            )
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        train_metrics = acc.compute()
        logger.log_epoch_metrics(train_metrics, epoch, prefix="finetune_train")
        logger.info(
            f"[Epoch {epoch}] Train  loss={train_metrics['loss']:.4f}  "
            f"region_f1={train_metrics['region_macro_f1']:.4f}  "
            f"conf_mae={train_metrics['confidence_mae']:.4f}"
        )

        # =====================================================================
        # VALIDATE
        # =====================================================================
        model.region_head.eval()
        model.confidence_head.eval()
        val_acc = RegionConfidenceMetricAccumulator()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val Epoch {epoch}]", leave=False):
                frames        = batch["frames"].to(device)
                det_labels    = batch["det_labels"].to(device)
                sev_labels    = batch["sev_labels"].to(device)
                region_labels = batch["region_labels"].to(device)

                with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"]):
                    ctx_feats  = model.encode(frames)
                    det_logits = model.det_head(ctx_feats)
                    sev_preds  = model.sev_head(ctx_feats)
                    det_prob   = torch.sigmoid(det_logits)

                    conf_targets = compute_confidence_targets(
                        det_prob=det_prob, det_labels=det_labels,
                        sev_preds=sev_preds, sev_labels=sev_labels,
                    )

                    region_logits = model.region_head(ctx_feats)
                    conf_preds    = model.confidence_head(ctx_feats)

                    region_loss = model.region_loss_fn(region_logits, region_labels)
                    conf_loss   = model.conf_loss_fn(conf_preds, conf_targets)
                    total_loss  = (
                        model.lambda_region * region_loss
                        + model.lambda_conf * conf_loss
                    )

                val_acc.update(
                    region_logits, region_labels, conf_preds, conf_targets,
                    total_loss=total_loss.item(),
                    region_loss=region_loss.item(),
                    conf_loss=conf_loss.item(),
                )

        val_metrics = val_acc.compute()
        logger.log_epoch_metrics(val_metrics, epoch, prefix="finetune_val")
        logger.info(
            f"[Epoch {epoch}] Val    loss={val_metrics['loss']:.4f}  "
            f"region_f1={val_metrics['region_macro_f1']:.4f}  "
            f"conf_mae={val_metrics['confidence_mae']:.4f}"
        )
        for name in RegionConfidenceMetricAccumulator.REGION_NAMES:
            logger.info(
                f"    region_{name}: acc={val_metrics[f'region_{name}_accuracy']:.4f}  "
                f"f1={val_metrics[f'region_{name}_f1']:.4f}"
            )

        # ---- Save best (by region macro F1) ---------------------------------
        if val_metrics["region_macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["region_macro_f1"]
            save_checkpoint(
                args.output, model, optimizer, None, scaler,
                epoch, best_macro_f1, cfg,
            )
            logger.info(f"  ✓ Saved new best 4-head checkpoint: {args.output} "
                        f"(region_macro_f1={best_macro_f1:.4f})")

    logger.info("Head-only fine-tuning complete.")
    logger.info(f"Best region macro F1: {best_macro_f1:.4f}")
    logger.info(f"4-head checkpoint saved at: {args.output}")
    logger.info(f"Original checkpoint UNCHANGED at: {args.base_checkpoint}")
    logger.close()


if __name__ == "__main__":
    main()
