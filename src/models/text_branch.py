"""Text branch — DistilBERT encoder for transaction description analysis."""

import logging

import torch
import torch.nn as nn
from transformers import DistilBertModel

logger = logging.getLogger(__name__)


class TextBranch(nn.Module):
    """DistilBERT encoder for free-text merchant descriptions.

    Architecture:
        DistilBERT → [CLS] token → LayerNorm → Linear → embedding (128-d)
        embedding → logit head (scalar)

    Args:
        model_name: HuggingFace model identifier for DistilBERT.
        embedding_dim: Output embedding dimensionality.
        freeze_layers: Number of transformer layers to freeze (0 = full fine-tune).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 128,
        freeze_layers: int = 4,
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # Freeze early layers
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.bert.transformer.layer))):
                for param in self.bert.transformer.layer[layer_idx].parameters():
                    param.requires_grad = False
            logger.info(
                "TextBranch: Froze embeddings + %d/%d transformer layers",
                freeze_layers, len(self.bert.transformer.layer),
            )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.logit_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) attention mask.

        Returns:
            dict with 'embedding' (B, embedding_dim) and 'logit' (B, 1).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token is at position 0
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)

        cls_output = self.layer_norm(cls_output)
        embedding = self.projection(cls_output)
        logit = self.logit_head(embedding)

        return {"embedding": embedding, "logit": logit}
