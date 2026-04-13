"""Attention-based fusion head for combining multimodal embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Learned attention-based fusion for multimodal embeddings.

    Takes embeddings from each modality branch and produces:
      - A weighted fused embedding
      - Per-modality attention weights (for explainability)
      - A final fraud probability

    Architecture:
        [e_tab, e_img, e_txt] stacked → Multi-Head Attention → weighted sum
        → LayerNorm → MLP classifier → sigmoid probability

    Args:
        embedding_dim: Dimension of each modality embedding.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Query vector (learnable — "what combination of modalities is most informative?")
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(
        self,
        tabular_emb: torch.Tensor,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            tabular_emb: (B, D) tabular embedding.
            image_emb: (B, D) image embedding.
            text_emb: (B, D) text embedding.

        Returns:
            dict with:
              - 'logit': (B, 1) raw logit
              - 'probability': (B, 1) sigmoid probability
              - 'attention_weights': (B, 3) per-modality attention weights
              - 'fused_embedding': (B, D) fused representation
        """
        B = tabular_emb.size(0)

        # Stack modality embeddings: (B, 3, D)
        modality_stack = torch.stack([tabular_emb, image_emb, text_emb], dim=1)

        # Expand query for batch: (B, 1, D)
        query = self.query.expand(B, -1, -1)

        # Cross-attention: query attends to modality embeddings
        attn_output, attn_weights = self.attention(
            query=query,
            key=modality_stack,
            value=modality_stack,
        )
        # attn_output: (B, 1, D), attn_weights: (B, 1, 3)

        fused = attn_output.squeeze(1)  # (B, D)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        logit = self.classifier(fused)  # (B, 1)
        probability = torch.sigmoid(logit)

        # Extract attention weights for explainability
        weights = attn_weights.squeeze(1)  # (B, 3) — [tabular, image, text]

        return {
            "logit": logit,
            "probability": probability,
            "attention_weights": weights,
            "fused_embedding": fused,
        }
