"""Main entry point for FraudLens training."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.config import get_training_config
from src.data.multimodal_dataset import MultimodalDataset
from src.models.fraudlens import FraudLensModel
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FraudLens multimodal model")
    parser.add_argument("--tabular-dir", type=str, help="Path to tabular data directory")
    parser.add_argument("--image-dir", type=str, help="Path to image data directory")
    parser.add_argument("--text-path", type=str, help="Path to text descriptions CSV")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Max training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--sample-size", type=int, help="Limit dataset size (for dev)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_training_config()

    # Override config with CLI args
    if args.tabular_dir:
        config["tabular_dir"] = args.tabular_dir
    if args.image_dir:
        config["image_dir"] = args.image_dir
    if args.text_path:
        config["text_path"] = args.text_path
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.epochs:
        config["max_epochs"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr
    if args.sample_size:
        config["sample_size"] = args.sample_size
    if args.no_amp:
        config["mixed_precision"] = False

    set_seed(config["seed"])

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Device: %s", device)
    logger.info("Config: %s", config)

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading multimodal dataset...")
    dataset = MultimodalDataset(
        tabular_dir=config["tabular_dir"],
        image_dir=config["image_dir"],
        text_path=config["text_path"],
        text_kwargs={"max_length": config["text_max_length"]},
        sample_size=config.get("sample_size"),
        seed=config["seed"],
    )

    # Train/val split
    val_size = int(len(dataset) * config["val_split"])
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(config["seed"])
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ────────────────────────────────────────────────────────────────
    model = FraudLensModel(
        tabular_input_dim=dataset.tabular_ds.input_dim,
        tabular_hidden_dims=config["tabular_hidden_dims"],
        tabular_dropout=config["tabular_dropout"],
        image_model_name=config["image_model"],
        image_freeze_layers=config["image_freeze_layers"],
        text_model_name=config["text_model"],
        text_freeze_layers=config["text_freeze_layers"],
        embedding_dim=config["embedding_dim"],
        fusion_num_heads=config["fusion_num_heads"],
        fusion_dropout=config["fusion_dropout"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    best_metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("Best validation metrics:")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        elif isinstance(v, str):
            logger.info("  %s:\n%s", k, v)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
