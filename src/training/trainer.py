"""Training loop for FraudLens multimodal model."""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.training.losses import MultiModalLoss
from src.training.metrics import MetricsAccumulator

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with mixed precision, early stopping, and TensorBoard logging.

    Args:
        model: FraudLensModel instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration dict.
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("max_epochs", 50) // 3,
            T_mult=2,
        )

        # Loss
        self.criterion = MultiModalLoss(
            alpha=config.get("focal_loss_alpha", 0.75),
            gamma=config.get("focal_loss_gamma", 2.0),
        )

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        # Early stopping
        self.patience = config.get("early_stopping_patience", 7)
        self.best_auprc = 0.0
        self.epochs_without_improvement = 0

        # Gradient clipping
        self.grad_clip = config.get("gradient_clip_norm", 1.0)

        # Logging
        log_dir = Path(config.get("log_dir", "runs"))
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        metrics = MetricsAccumulator()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            tabular = batch["tabular"].to(self.device)
            image = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(tabular, image, input_ids, attention_mask)
                losses = self.criterion(output, labels)

            self.scaler.scale(losses["total"]).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate
            probs = output["probability"].detach().cpu().numpy().flatten()
            batch_labels = labels.detach().cpu().numpy().flatten()
            metrics.update(batch_labels, probs, losses["total"].item())

            if batch_idx % 50 == 0:
                logger.info(
                    "Epoch %d [%d/%d] loss=%.4f",
                    epoch, batch_idx, len(self.train_loader), losses["total"].item(),
                )

        self.scheduler.step()
        return metrics.compute()

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on the validation set."""
        self.model.eval()
        metrics = MetricsAccumulator()

        for batch in self.val_loader:
            tabular = batch["tabular"].to(self.device)
            image = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(tabular, image, input_ids, attention_mask)
                losses = self.criterion(output, labels)

            probs = output["probability"].cpu().numpy().flatten()
            batch_labels = labels.cpu().numpy().flatten()
            metrics.update(batch_labels, probs, losses["total"].item())

        return metrics.compute()

    def train(self) -> dict:
        """Full training loop with early stopping."""
        max_epochs = self.config.get("max_epochs", 50)
        best_metrics = {}

        logger.info("Starting training for %d epochs on %s", max_epochs, self.device)
        logger.info("Mixed precision: %s", self.use_amp)

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            elapsed = time.time() - t0

            # Log to TensorBoard
            self.writer.add_scalars("loss", {
                "train": train_metrics["loss"],
                "val": val_metrics["loss"],
            }, epoch)

            for key in ["auroc", "auprc", "f1", "precision", "recall"]:
                if key in val_metrics:
                    self.writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
                if key in train_metrics:
                    self.writer.add_scalar(f"train/{key}", train_metrics[key], epoch)

            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                "Epoch %d/%d (%.1fs) — train_loss=%.4f, val_loss=%.4f, "
                "val_auroc=%.4f, val_auprc=%.4f, val_f1=%.4f",
                epoch, max_epochs, elapsed,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics.get("auroc", 0), val_metrics.get("auprc", 0),
                val_metrics.get("f1", 0),
            )

            # Early stopping on AUPRC (better for imbalanced data)
            val_auprc = val_metrics.get("auprc", 0)
            if val_auprc > self.best_auprc:
                self.best_auprc = val_auprc
                self.epochs_without_improvement = 0
                best_metrics = val_metrics

                # Save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_auprc": self.best_auprc,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
                logger.info("✓ Saved best model (AUPRC=%.4f)", val_auprc)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, self.patience,
                    )
                    break

        self.writer.close()
        logger.info("Training complete. Best AUPRC: %.4f", self.best_auprc)
        return best_metrics
