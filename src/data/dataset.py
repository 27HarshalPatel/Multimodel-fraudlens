"""FraudLens — Unified multimodal dataset loader.

Normalizes three heterogeneous real-world sources into a single PyTorch
Dataset that returns (tabular, image, text) triples:

  1. **IEEE-CIS Fraud Detection** (Kaggle) — tabular transaction metadata
     with 392 features and ~3.5 % fraud rate.
  2. **PaySim** (Kaggle) — synthetic mobile money logs with 11 columns.
  3. **SSBI Check Images** (GitHub/saifkhichi96) — real bank-check photos
     split into ``normal`` and ``tampered`` directories.

When a data source is not yet downloaded the loader falls back to the
locally-generated synthetic data already created by
``src/data/generate_synthetic.py``.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, DistilBertTokenizerFast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schemas per source
# ---------------------------------------------------------------------------
IEEE_NUMERIC = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2",
]
IEEE_CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain",
    "R_emaildomain", "DeviceType", "DeviceInfo",
]

PAYSIM_RENAME = {
    "amount": "TransactionAmt",
    "type": "ProductCD",
    "isFraud": "isFraud",
}


# ---------------------------------------------------------------------------
# Tabular normalisation helpers
# ---------------------------------------------------------------------------
def _load_ieee(tabular_dir: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load IEEE-CIS data and merge identity table."""
    trans = tabular_dir / "train_transaction.csv"
    ident = tabular_dir / "train_identity.csv"
    if not trans.exists():
        return pd.DataFrame()

    df_t = pd.read_csv(trans)
    if ident.exists():
        df_i = pd.read_csv(ident)
        df = df_t.merge(df_i, on="TransactionID", how="left")
    else:
        df = df_t

    df["_source"] = "ieee"
    if sample_size:
        df = df.head(sample_size)
    return df


def _load_paysim(paysim_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load PaySim CSV and rename columns to the shared schema."""
    if not paysim_path.exists():
        logger.info("PaySim CSV not found at %s — skipping.", paysim_path)
        return pd.DataFrame()

    df = pd.read_csv(paysim_path)
    df = df.rename(columns=PAYSIM_RENAME)

    # Map PaySim's type → ProductCD codes used by IEEE-CIS
    type_map = {"PAYMENT": "W", "TRANSFER": "H", "CASH_OUT": "C",
                "DEBIT": "S", "CASH_IN": "R"}
    if "ProductCD" in df.columns:
        df["ProductCD"] = df["ProductCD"].map(type_map).fillna("W")

    # Synthesise missing columns expected by the feature engineer
    for col in IEEE_NUMERIC + IEEE_CATEGORICAL:
        if col not in df.columns:
            df[col] = np.nan

    if "TransactionID" not in df.columns:
        df["TransactionID"] = np.arange(len(df)) + 900_000  # avoid ID clash

    df["_source"] = "paysim"
    if sample_size:
        df = df.head(sample_size)
    return df


def _engineer_tabular(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the same feature engineering as ``TabularDataset``.

    Returns (features, labels, input_dim).
    """
    features = pd.DataFrame()

    # Log amount
    features["log_amount"] = np.log1p(
        df["TransactionAmt"].fillna(0).clip(lower=0).astype(float)
    )

    # Cyclical time
    if "TransactionDT" in df.columns:
        tod = df["TransactionDT"].fillna(0).astype(float) % 86400
        features["time_sin"] = np.sin(2 * np.pi * tod / 86400)
        features["time_cos"] = np.cos(2 * np.pi * tod / 86400)
    else:
        features["time_sin"] = 0.0
        features["time_cos"] = 0.0

    # Numeric
    for col in IEEE_NUMERIC:
        if col in df.columns:
            features[col] = df[col].astype(float)
        else:
            features[col] = 0.0

    # V-features (20)
    for i in range(1, 21):
        c = f"V{i}"
        features[c] = df[c].astype(float) if c in df.columns else 0.0

    # C-features (10)
    for i in range(1, 11):
        c = f"C{i}"
        features[c] = df[c].astype(float) if c in df.columns else 0.0

    # D-features (5)
    for i in range(1, 6):
        c = f"D{i}"
        features[c] = df[c].astype(float) if c in df.columns else 0.0

    # Median-impute
    features = features.fillna(features.median())

    # Categoricals → label encode
    for col in IEEE_CATEGORICAL:
        if col in df.columns:
            le = LabelEncoder()
            features[f"cat_{col}"] = le.fit_transform(
                df[col].fillna("missing").astype(str)
            )
        else:
            features[f"cat_{col}"] = 0

    mat = features.values.astype(np.float32)
    mat = np.nan_to_num(mat, nan=0.0)

    scaler = StandardScaler()
    mat = scaler.fit_transform(mat).astype(np.float32)

    labels = df["isFraud"].values.astype(np.float32)
    return mat, labels, mat.shape[1]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _discover_images(image_dir: Path) -> list[tuple[Path, int]]:
    """Return a sorted list of (path, label) for check images."""
    samples: list[tuple[Path, int]] = []
    for subdir, label in [("normal", 0), ("tampered", 1)]:
        d = image_dir / subdir
        if d.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for p in sorted(d.glob(ext)):
                    samples.append((p, label))
    return samples


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _load_text_csv(text_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    if not text_path.exists():
        logger.warning("Text CSV not found at %s — synthesising placeholder.", text_path)
        return pd.DataFrame({"TransactionID": [], "description": [], "is_fraud": []})
    df = pd.read_csv(text_path)
    if sample_size:
        df = df.head(sample_size)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main unified dataset
# ═══════════════════════════════════════════════════════════════════════════
class FraudLensDataset(Dataset):
    """Unified multimodal dataset that fuses IEEE-CIS, PaySim, and SSBI.

    Each ``__getitem__`` returns a dict::

        {
            "tabular":        FloatTensor (input_dim,),
            "image":          FloatTensor (3, 224, 224),
            "input_ids":      LongTensor  (max_length,),
            "attention_mask": LongTensor  (max_length,),
            "label":          FloatTensor scalar,
        }

    Args:
        tabular_dir:  Root of IEEE-CIS CSVs  (``data/tabular``).
        paysim_path:  Path to PaySim CSV      (``data/paysim/paysim.csv``).
        image_dir:    Root of check images    (``data/images``).
        text_path:    Path to descriptions CSV (``data/text/descriptions.csv``).
        image_model:  HuggingFace model for image preprocessing.
        text_model:   HuggingFace tokeniser model.
        max_length:   Token limit for DistilBERT.
        sample_size:  Cap per source (for dev / Colab).
        augment:      Apply image augmentations.
        seed:         Reproducibility seed.
    """

    def __init__(
        self,
        tabular_dir: str | Path = "data/tabular",
        paysim_path: str | Path = "data/paysim/paysim.csv",
        image_dir: str | Path = "data/images",
        text_path: str | Path = "data/text/descriptions.csv",
        image_model: str = "google/siglip2-base-patch16-224",
        text_model: str = "distilbert-base-uncased",
        max_length: int = 128,
        sample_size: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.seed = seed
        self.max_length = max_length
        self._rng = random.Random(seed)

        # ── 1. Tabular ────────────────────────────────────────────────
        df_ieee = _load_ieee(Path(tabular_dir), sample_size)
        df_paysim = _load_paysim(Path(paysim_path), sample_size)

        parts = [df for df in (df_ieee, df_paysim) if len(df) > 0]
        if not parts:
            raise FileNotFoundError(
                "No tabular data found. Run `python -m src.data.download_data` first."
            )
        df_all = pd.concat(parts, ignore_index=True)
        logger.info(
            "Tabular sources: IEEE=%d, PaySim=%d → total %d",
            len(df_ieee), len(df_paysim), len(df_all),
        )

        self.features, self.labels, self.input_dim = _engineer_tabular(df_all)
        self.transaction_ids = df_all["TransactionID"].values

        # ── 2. Images ─────────────────────────────────────────────────
        self.image_samples = _discover_images(Path(image_dir))
        self._normal_idx = [i for i, (_, l) in enumerate(self.image_samples) if l == 0]
        self._fraud_idx = [i for i, (_, l) in enumerate(self.image_samples) if l == 1]
        logger.info(
            "Images: %d normal, %d tampered",
            len(self._normal_idx), len(self._fraud_idx),
        )

        # Image processor
        try:
            self._img_proc = AutoImageProcessor.from_pretrained(image_model)
            self._use_proc = True
        except Exception:
            self._use_proc = False

        aug_list: list = []
        if augment:
            aug_list += [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ]
        self._aug = transforms.Compose(aug_list) if aug_list else None
        self._fallback = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # ── 3. Text ──────────────────────────────────────────────────
        df_text = _load_text_csv(Path(text_path), sample_size)
        self.texts = df_text["description"].fillna("").values.tolist() if len(df_text) else []
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(text_model)

        # ── Summary ──────────────────────────────────────────────────
        self._size = len(self.labels)
        logger.info(
            "FraudLensDataset ready: %d samples, %d features, "
            "%d images, %d texts, %.2f%% fraud",
            self._size, self.input_dim,
            len(self.image_samples), len(self.texts),
            100 * self.labels.mean(),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        # ── Tabular ──────────────────────────────────────────────────
        tab = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # ── Image (stratified pair) ──────────────────────────────────
        is_fraud = self.labels[idx] > 0.5
        if is_fraud and self._fraud_idx:
            img_i = self._fraud_idx[idx % len(self._fraud_idx)]
        elif self._normal_idx:
            img_i = self._normal_idx[idx % len(self._normal_idx)]
        else:
            img_i = 0

        if self.image_samples:
            img_path, _ = self.image_samples[img_i]
            pil = Image.open(img_path).convert("RGB")
            if self._aug:
                pil = self._aug(pil)
            if self._use_proc:
                pixel_values = self._img_proc(images=pil, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                if not isinstance(pil, Image.Image):
                    pil = transforms.ToPILImage()(pil)
                pixel_values = self._fallback(pil)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        # ── Text ─────────────────────────────────────────────────────
        text = self.texts[idx % len(self.texts)] if self.texts else ""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "tabular": tab,
            "image": pixel_values,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label,
        }
