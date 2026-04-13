"""FraudLens — Complete multimodal fraud detection model."""

import torch
import torch.nn as nn

from src.models.fusion import AttentionFusion
from src.models.image_branch import ImageBranch
from src.models.tabular_branch import TabularBranch
from src.models.text_branch import TextBranch


class FraudLensModel(nn.Module):
    """End-to-end multimodal fraud detection model.

    Combines three modality branches with attention-based fusion:
      - TabularBranch  (MLP)       → 128-d embedding + logit
      - ImageBranch    (SigLIP 2)  → 128-d embedding + logit
      - TextBranch     (DistilBERT)→ 128-d embedding + logit
      - AttentionFusion            → weighted fused prediction

    Args:
        tabular_input_dim: Number of preprocessed tabular features.
        tabular_hidden_dims: Hidden layer widths for MLP.
        tabular_dropout: Dropout for tabular branch.
        image_model_name: HuggingFace SigLIP 2 model name.
        image_freeze_layers: Layers to freeze in SigLIP 2.
        text_model_name: HuggingFace DistilBERT model name.
        text_freeze_layers: Layers to freeze in DistilBERT.
        embedding_dim: Shared embedding dimensionality.
        fusion_num_heads: Attention heads in fusion.
        fusion_dropout: Dropout in fusion layer.
    """

    def __init__(
        self,
        tabular_input_dim: int = 128,
        tabular_hidden_dims: list[int] | None = None,
        tabular_dropout: float = 0.3,
        image_model_name: str = "google/siglip2-base-patch16-224",
        image_freeze_layers: int = 8,
        text_model_name: str = "distilbert-base-uncased",
        text_freeze_layers: int = 4,
        embedding_dim: int = 128,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.2,
    ):
        super().__init__()

        self.tabular_branch = TabularBranch(
            input_dim=tabular_input_dim,
            hidden_dims=tabular_hidden_dims,
            embedding_dim=embedding_dim,
            dropout=tabular_dropout,
        )

        self.image_branch = ImageBranch(
            model_name=image_model_name,
            embedding_dim=embedding_dim,
            freeze_layers=image_freeze_layers,
        )

        self.text_branch = TextBranch(
            model_name=text_model_name,
            embedding_dim=embedding_dim,
            freeze_layers=text_freeze_layers,
        )

        self.fusion = AttentionFusion(
            embedding_dim=embedding_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )

    def forward(
        self,
        tabular: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through all branches and fusion.

        Args:
            tabular: (B, F) tabular features.
            image: (B, 3, 224, 224) preprocessed images.
            input_ids: (B, seq_len) text token IDs.
            attention_mask: (B, seq_len) text attention mask.

        Returns:
            dict with:
              - 'logit': (B, 1) fused logit
              - 'probability': (B, 1) fused sigmoid probability
              - 'attention_weights': (B, 3) [tabular, image, text] weights
              - 'tabular_logit': (B, 1) tabular branch logit
              - 'image_logit': (B, 1) image branch logit
              - 'text_logit': (B, 1) text branch logit
              - 'tabular_embedding': (B, D) tabular embedding
              - 'image_embedding': (B, D) image embedding
              - 'text_embedding': (B, D) text embedding
              - 'fused_embedding': (B, D) fused embedding
        """
        # Branch forward passes
        tab_out = self.tabular_branch(tabular)
        img_out = self.image_branch(image)
        txt_out = self.text_branch(input_ids, attention_mask)

        # Fusion
        fusion_out = self.fusion(
            tab_out["embedding"],
            img_out["embedding"],
            txt_out["embedding"],
        )

        return {
            # Fused outputs
            "logit": fusion_out["logit"],
            "probability": fusion_out["probability"],
            "attention_weights": fusion_out["attention_weights"],
            "fused_embedding": fusion_out["fused_embedding"],
            # Per-branch outputs
            "tabular_logit": tab_out["logit"],
            "image_logit": img_out["logit"],
            "text_logit": txt_out["logit"],
            "tabular_embedding": tab_out["embedding"],
            "image_embedding": img_out["embedding"],
            "text_embedding": txt_out["embedding"],
        }
