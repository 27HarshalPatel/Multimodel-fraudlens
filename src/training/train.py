"""FraudLens — Distributed training entrypoint for Colab / multi-GPU.

Usage
-----
**Single GPU (Colab H100)**::

    python -m src.training.train --device cuda --epochs 30

**Multi-GPU DDP**::

    torchrun --nproc_per_node=2 -m src.training.train --ddp

**Quick dev run (100 samples, CPU)**::

    python -m src.training.train --sample-size 100 --device cpu --epochs 2

This script wraps the existing ``Trainer`` class with:
  - ``torch.distributed`` + ``DistributedDataParallel`` (DDP)
  - Automatic mixed-precision on CUDA (``torch.amp``)
  - Colab-friendly single-GPU fast-path
  - TensorBoard logging + checkpoint saving
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudlens.train")


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    """True when running non-distributed OR on rank 0."""
    return int(os.environ.get("RANK", "0")) == 0


def setup_ddp() -> tuple[int, int]:
    """Initialise the NCCL process-group; returns (local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FraudLens distributed training")

    # Data
    p.add_argument("--tabular-dir", default="data/tabular")
    p.add_argument("--paysim-path", default="data/paysim/paysim.csv")
    p.add_argument("--image-dir", default="data/images")
    p.add_argument("--text-path", default="data/text/descriptions.csv")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Cap dataset size (for quick dev runs)")

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--patience", type=int, default=7,
                   help="Early-stopping patience (epochs)")

    # Distributed
    p.add_argument("--ddp", action="store_true",
                   help="Use DistributedDataParallel (launched via torchrun)")
    p.add_argument("--device", default=None,
                   help="Force device (cuda / mps / cpu)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="runs")

    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Device setup
    use_ddp = args.ddp and torch.cuda.is_available()
    if use_ddp:
        local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        if is_main_process():
            logger.info("DDP initialised: world_size=%d", world_size)
    else:
        local_rank, world_size = 0, 1
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if is_main_process():
        logger.info("Device: %s  |  DDP: %s  |  AMP: %s",
                     device, use_ddp, not args.no_amp and device.type == "cuda")

    # ── Dataset ──────────────────────────────────────────────────────────
    from src.data.dataset import FraudLensDataset  # noqa: late import

    dataset = FraudLensDataset(
        tabular_dir=args.tabular_dir,
        paysim_path=args.paysim_path,
        image_dir=args.image_dir,
        text_path=args.text_path,
        sample_size=args.sample_size,
        augment=True,
        seed=args.seed,
    )

    # Train / val split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    if is_main_process():
        logger.info("Dataset: %d train, %d val, %d features",
                     train_size, val_size, dataset.input_dim)

    # Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────
    from src.models.fraudlens import FraudLensModel  # noqa: late import

    model = FraudLensModel(tabular_input_dim=dataset.input_dim)
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main_process():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Parameters: %s total, %s trainable",
                     f"{total:,}", f"{trainable:,}")

    # ── Train ────────────────────────────────────────────────────────────
    config = {
        "max_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.grad_clip,
        "mixed_precision": not args.no_amp,
        "early_stopping_patience": args.patience,
        "focal_loss_alpha": 0.75,
        "focal_loss_gamma": 2.0,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
    }

    from src.training.trainer import Trainer  # noqa: late import

    # For DDP the Trainer receives the wrapped model; it still calls
    # model.train() / model.eval() which DDP handles transparently.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Patch: set DDP epoch on sampler each epoch so shuffling varies
    _orig_train_epoch = trainer.train_epoch

    def _ddp_train_epoch(epoch: int) -> dict:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        return _orig_train_epoch(epoch)

    if use_ddp:
        trainer.train_epoch = _ddp_train_epoch  # type: ignore[method-assign]

    best = trainer.train()

    # ── Report ───────────────────────────────────────────────────────────
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Best validation metrics:")
        for k, v in best.items():
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
        logger.info("-" * 60)
        logger.info("Summary (use optimal threshold for deployment):")
        logger.info("  AUROC:  %.4f", best.get("auroc", 0))
        logger.info("  AUPRC:  %.4f", best.get("auprc", 0))
        logger.info("  F1 @0.5:     %.4f  |  F1 @optimal:     %.4f",
                     best.get("f1", 0), best.get("f1_optimal", 0))
        logger.info("  Prec @0.5:   %.4f  |  Prec @optimal:   %.4f",
                     best.get("precision", 0), best.get("precision_optimal", 0))
        logger.info("  Recall @0.5: %.4f  |  Recall @optimal: %.4f",
                     best.get("recall", 0), best.get("recall_optimal", 0))
        logger.info("  Optimal threshold: %.4f", best.get("optimal_threshold", 0.5))
        logger.info("=" * 60)

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
