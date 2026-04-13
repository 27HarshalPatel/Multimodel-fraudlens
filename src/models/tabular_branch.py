"""Tabular branch — MLP encoder for structured transaction features."""

import torch
import torch.nn as nn


class TabularBranch(nn.Module):
    """Multi-layer perceptron for tabular transaction features.

    Architecture:
        input → [Linear → BN → ReLU → Dropout] × N → embedding (128-d)
        embedding → logit head (scalar)

    Args:
        input_dim: Number of input features after preprocessing.
        hidden_dims: List of hidden layer widths.
        embedding_dim: Output embedding dimensionality.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(prev_dim, embedding_dim)
        self.logit_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, input_dim) tabular feature tensor.

        Returns:
            dict with 'embedding' (B, embedding_dim) and 'logit' (B, 1).
        """
        h = self.encoder(x)
        embedding = self.projection(h)
        logit = self.logit_head(embedding)
        return {"embedding": embedding, "logit": logit}
