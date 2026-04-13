"""Text dataset for transaction description analysis using DistilBERT."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """PyTorch dataset for free-text merchant descriptions.

    Tokenises with DistilBertTokenizer and returns input_ids + attention_mask.
    """

    def __init__(
        self,
        text_path: str | Path,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        sample_size: Optional[int] = None,
    ):
        self.text_path = Path(text_path)
        self.max_length = max_length

        # Load descriptions
        df = pd.read_csv(self.text_path)
        if sample_size is not None:
            df = df.head(sample_size)

        self.transaction_ids = df["TransactionID"].values
        self.texts = df["description"].fillna("").values.tolist()
        self.labels = df["is_fraud"].values.astype(float)

        # Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        logger.info(
            "TextDataset: %d samples, max_length=%d, %.2f%% fraud",
            len(self), max_length, 100 * self.labels.mean(),
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "transaction_id": int(self.transaction_ids[idx]),
        }
