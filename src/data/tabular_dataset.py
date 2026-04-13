"""Tabular dataset for IEEE-CIS transaction data with feature engineering."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """PyTorch dataset for tabular transaction features.

    Performs feature engineering:
      - Log-transform transaction amounts
      - Cyclical encoding of time-of-day from TransactionDT
      - Label-encoding of categorical columns
      - Standard scaling of numeric columns
      - Missing-value imputation (median for numeric, mode for categorical)
    """

    NUMERIC_COLS = [
        "TransactionAmt", "card1", "card2", "card3", "card5",
        "addr1", "addr2",
    ]
    CATEGORICAL_COLS = [
        "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ]

    def __init__(
        self,
        tabular_dir: str | Path,
        sample_size: Optional[int] = None,
        fit_preprocessors: bool = True,
        scaler: Optional[StandardScaler] = None,
        label_encoders: Optional[dict] = None,
    ):
        self.tabular_dir = Path(tabular_dir)
        self.fit_preprocessors = fit_preprocessors

        # Load data
        trans_path = self.tabular_dir / "train_transaction.csv"
        ident_path = self.tabular_dir / "train_identity.csv"

        df_trans = pd.read_csv(trans_path)
        if ident_path.exists():
            df_ident = pd.read_csv(ident_path)
            df = df_trans.merge(df_ident, on="TransactionID", how="left")
        else:
            df = df_trans

        if sample_size is not None:
            df = df.head(sample_size)

        self.transaction_ids = df["TransactionID"].values
        self.labels = df["isFraud"].values.astype(np.float32)

        # ── Feature engineering ──────────────────────────────────────
        features = pd.DataFrame()

        # Log amount
        features["log_amount"] = np.log1p(df["TransactionAmt"].fillna(0).clip(lower=0))

        # Time-of-day cyclical encoding
        if "TransactionDT" in df.columns:
            seconds_in_day = 86400
            time_of_day = df["TransactionDT"] % seconds_in_day
            features["time_sin"] = np.sin(2 * np.pi * time_of_day / seconds_in_day)
            features["time_cos"] = np.cos(2 * np.pi * time_of_day / seconds_in_day)

        # Numeric features
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                features[col] = df[col].astype(float)

        # V-features (principal components)
        v_cols = [c for c in df.columns if c.startswith("V")]
        for col in v_cols[:20]:  # cap at 20 for manageable size
            features[col] = df[col].astype(float)

        # C-features (counting)
        c_cols = [c for c in df.columns if c.startswith("C")]
        for col in c_cols[:10]:
            features[col] = df[col].astype(float)

        # D-features (timedelta)
        d_cols = [c for c in df.columns if c.startswith("D")]
        for col in d_cols[:5]:
            features[col] = df[col].astype(float)

        # Impute numeric: median fill
        features = features.fillna(features.median())

        # Categorical → label encode
        self.label_encoders = label_encoders or {}
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                series = df[col].fillna("missing").astype(str)
                if fit_preprocessors:
                    le = LabelEncoder()
                    features[f"cat_{col}"] = le.fit_transform(series)
                    self.label_encoders[col] = le
                elif col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    features[f"cat_{col}"] = series.map(
                        lambda x, _le=le: (
                            _le.transform([x])[0] if x in _le.classes_ else len(_le.classes_)
                        )
                    )

        # Scale
        feature_matrix = features.values.astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        if fit_preprocessors:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(feature_matrix).astype(np.float32)
        else:
            self.scaler = scaler
            self.features = (
                scaler.transform(feature_matrix).astype(np.float32)
                if scaler is not None
                else feature_matrix
            )

        self.input_dim = self.features.shape[1]
        logger.info(
            "TabularDataset: %d samples, %d features, %.2f%% fraud",
            len(self), self.input_dim, 100 * self.labels.mean(),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tabular": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "transaction_id": int(self.transaction_ids[idx]),
        }
