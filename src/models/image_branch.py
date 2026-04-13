"""Image branch — SigLIP 2 vision encoder for check image analysis."""

import logging

import torch
import torch.nn as nn
from transformers import SiglipVisionModel

logger = logging.getLogger(__name__)


class ImageBranch(nn.Module):
    """SigLIP 2 vision encoder for detecting visual fraud artifacts in check images.

    Architecture:
        SigLIP 2 ViT → [CLS] token → LayerNorm → Linear → embedding (128-d)
        embedding → logit head (scalar)

    The ViT backbone can be partially frozen to reduce compute:
        freeze_layers=8 freezes the first 8 transformer blocks.

    Args:
        model_name: HuggingFace model identifier for SigLIP 2.
        embedding_dim: Output embedding dimensionality.
        freeze_layers: Number of initial ViT layers to freeze (0 = full fine-tune).
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        embedding_dim: int = 128,
        freeze_layers: int = 8,
    ):
        super().__init__()

        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        hidden_size = self.vision_model.config.hidden_size  # typically 768 for base

        # The SiglipVisionModel wraps a SiglipVisionTransformer as .vision_model
        vit = self.vision_model.vision_model

        # Freeze early layers
        if freeze_layers > 0:
            # Freeze embeddings
            for param in vit.embeddings.parameters():
                param.requires_grad = False
            # Freeze specified encoder layers
            for layer_idx in range(min(freeze_layers, len(vit.encoder.layers))):
                for param in vit.encoder.layers[layer_idx].parameters():
                    param.requires_grad = False
            n_frozen = freeze_layers + 1  # +1 for embeddings
            n_total = len(vit.encoder.layers) + 1
            logger.info("ImageBranch: Froze %d/%d layer groups", n_frozen, n_total)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.logit_head = nn.Linear(embedding_dim, 1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: (B, 3, 224, 224) preprocessed image tensor.

        Returns:
            dict with 'embedding' (B, embedding_dim) and 'logit' (B, 1).
        """
        outputs = self.vision_model(pixel_values=pixel_values)

        # Pool: use the last hidden state's mean across patches
        # SigLIP doesn't use a [CLS] token, so we mean-pool across spatial tokens
        last_hidden = outputs.last_hidden_state  # (B, num_patches, hidden_size)
        pooled = last_hidden.mean(dim=1)  # (B, hidden_size)

        pooled = self.layer_norm(pooled)
        embedding = self.projection(pooled)
        logit = self.logit_head(embedding)

        return {"embedding": embedding, "logit": logit}
