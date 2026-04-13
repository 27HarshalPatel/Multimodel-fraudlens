"""Focal Loss for handling class imbalance in fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for binary classification.

    Down-weights well-classified examples, focusing training on hard negatives.
    Essential for the ~3.5% fraud rate in IEEE-CIS data.

    Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for the positive class (fraud). Default: 0.75.
        gamma: Focusing parameter. Higher = more focus on hard examples. Default: 2.0.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, 1) or (B,) raw model logits (before sigmoid).
            targets: (B,) or (B, 1) binary labels.

        Returns:
            Scalar loss if reduction='mean'/'sum', else (B,) per-sample loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1)

        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiModalLoss(nn.Module):
    """Combined loss for multimodal training.

    Sums the fused loss with auxiliary per-branch losses (weighted lower).
    This provides gradient signal to each branch independently.

    Total = L_fused + aux_weight * (L_tabular + L_image + L_text)

    Args:
        alpha: Focal loss alpha.
        gamma: Focal loss gamma.
        aux_weight: Weight for auxiliary branch losses. Default: 0.3.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, aux_weight: float = 0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.aux_weight = aux_weight

    def forward(self, model_output: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute combined multimodal loss.

        Args:
            model_output: dict from FraudLensModel.forward()
            targets: (B,) binary labels.

        Returns:
            dict with 'total', 'fused', 'tabular', 'image', 'text' losses.
        """
        loss_fused = self.focal(model_output["logit"], targets)
        loss_tab = self.focal(model_output["tabular_logit"], targets)
        loss_img = self.focal(model_output["image_logit"], targets)
        loss_txt = self.focal(model_output["text_logit"], targets)

        total = loss_fused + self.aux_weight * (loss_tab + loss_img + loss_txt)

        return {
            "total": total,
            "fused": loss_fused,
            "tabular": loss_tab,
            "image": loss_img,
            "text": loss_txt,
        }
