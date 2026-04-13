"""Multimodal dataset combining tabular, image, and text modalities."""

import logging
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.data.image_dataset import ImageDataset
from src.data.tabular_dataset import TabularDataset
from src.data.text_dataset import TextDataset

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """Combines tabular, image, and text datasets into a single multimodal dataset.

    Each sample returns a dict with all three modality inputs plus the label.
    Image samples are randomly paired when the image dataset size differs
    from the tabular/text dataset size.
    """

    def __init__(
        self,
        tabular_dir: str | Path,
        image_dir: str | Path,
        text_path: str | Path,
        tabular_kwargs: Optional[dict] = None,
        image_kwargs: Optional[dict] = None,
        text_kwargs: Optional[dict] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ):
        tabular_kwargs = tabular_kwargs or {}
        image_kwargs = image_kwargs or {}
        text_kwargs = text_kwargs or {}

        self.tabular_ds = TabularDataset(tabular_dir, sample_size=sample_size, **tabular_kwargs)
        self.text_ds = TextDataset(text_path, sample_size=sample_size, **text_kwargs)
        self.image_ds = ImageDataset(image_dir, **image_kwargs)

        # Map image indices by label for stratified pairing
        self._normal_indices = [i for i, (_, l) in enumerate(self.image_ds.samples) if l == 0]
        self._fraud_indices = [i for i, (_, l) in enumerate(self.image_ds.samples) if l == 1]

        self._rng = random.Random(seed)
        self._size = min(len(self.tabular_ds), len(self.text_ds))

        logger.info(
            "MultimodalDataset: %d samples (tabular=%d, images=%d, text=%d)",
            self._size, len(self.tabular_ds), len(self.image_ds), len(self.text_ds),
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        tabular_sample = self.tabular_ds[idx]
        text_sample = self.text_ds[idx]
        label = tabular_sample["label"]

        # Pair with a same-class image
        is_fraud = label.item() > 0.5
        if is_fraud and self._fraud_indices:
            img_idx = self._fraud_indices[idx % len(self._fraud_indices)]
        elif self._normal_indices:
            img_idx = self._normal_indices[idx % len(self._normal_indices)]
        else:
            img_idx = idx % len(self.image_ds)

        image_sample = self.image_ds[img_idx]

        return {
            "tabular": tabular_sample["tabular"],
            "image": image_sample["image"],
            "input_ids": text_sample["input_ids"],
            "attention_mask": text_sample["attention_mask"],
            "label": label,
        }
