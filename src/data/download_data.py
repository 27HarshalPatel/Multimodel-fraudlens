"""Download IEEE-CIS Fraud Detection dataset from Kaggle.

Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
Falls back to generating a small synthetic sample for CI / quick testing.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COMPETITION = "ieee-fraud-detection"
DATA_DIR = Path("data/tabular")


def download_from_kaggle(output_dir: Path = DATA_DIR) -> None:
    """Download and extract IEEE-CIS dataset via Kaggle API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        logger.warning(
            "KAGGLE_USERNAME / KAGGLE_KEY not set. "
            "Generating synthetic fallback dataset instead."
        )
        generate_fallback(output_dir)
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(COMPETITION, path=str(output_dir), unzip=True)
        logger.info("IEEE-CIS dataset downloaded to %s", output_dir)
    except Exception as e:
        logger.error("Kaggle download failed: %s. Generating fallback.", e)
        generate_fallback(output_dir)


def generate_fallback(output_dir: Path = DATA_DIR, n_samples: int = 5000) -> None:
    """Generate a small synthetic transaction dataset for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    fraud_rate = 0.035
    is_fraud = rng.random(n_samples) < fraud_rate

    # Transaction features
    data = {
        "TransactionID": np.arange(n_samples),
        "isFraud": is_fraud.astype(int),
        "TransactionDT": rng.integers(86400, 86400 * 180, size=n_samples),
        "TransactionAmt": np.where(
            is_fraud,
            rng.lognormal(mean=6.0, sigma=1.5, size=n_samples),
            rng.lognormal(mean=3.5, sigma=1.0, size=n_samples),
        ),
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n_samples),
        "card1": rng.integers(1000, 20000, size=n_samples),
        "card2": rng.choice([100, 200, 300, 400, 500, np.nan], size=n_samples),
        "card3": rng.choice([150, 185, 200, np.nan], size=n_samples),
        "card4": rng.choice(["visa", "mastercard", "discover", "american express", np.nan], size=n_samples),
        "card5": rng.choice([100, 117, 166, 224, 226, np.nan], size=n_samples),
        "card6": rng.choice(["debit", "credit", "charge", np.nan], size=n_samples),
        "addr1": rng.choice([200, 300, 400, 500, np.nan], size=n_samples),
        "addr2": rng.choice([87.0, 60.0, 96.0, np.nan], size=n_samples),
        "P_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com", np.nan],
            size=n_samples,
        ),
        "R_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", np.nan],
            size=n_samples,
        ),
    }

    # Add V-features (principal components)
    for i in range(1, 21):
        data[f"V{i}"] = rng.normal(0, 1, size=n_samples)
        if is_fraud.any():
            fraud_idx = np.where(is_fraud)[0]
            data[f"V{i}"][fraud_idx] += rng.normal(0.5, 0.3, size=len(fraud_idx))

    # Add C-features (counting features)
    for i in range(1, 11):
        data[f"C{i}"] = rng.integers(0, 100, size=n_samples).astype(float)

    # Add D-features (timedelta)
    for i in range(1, 6):
        data[f"D{i}"] = rng.choice(
            [*rng.uniform(0, 500, size=20).tolist(), np.nan],
            size=n_samples,
        )

    df = pd.DataFrame(data)
    df.to_csv(output_dir / "train_transaction.csv", index=False)

    # Identity table
    id_data = {
        "TransactionID": np.arange(n_samples),
        "DeviceType": rng.choice(["desktop", "mobile", np.nan], size=n_samples),
        "DeviceInfo": rng.choice(
            ["Windows", "iOS Device", "MacOS", "Trident/7.0", np.nan],
            size=n_samples,
        ),
        "id_30": rng.choice(["Windows 10", "iOS 12.1", "Android 7.0", np.nan], size=n_samples),
        "id_31": rng.choice(["chrome", "safari", "firefox", "edge", np.nan], size=n_samples),
    }
    pd.DataFrame(id_data).to_csv(output_dir / "train_identity.csv", index=False)

    logger.info("Generated fallback dataset with %d samples (%.1f%% fraud)", n_samples, fraud_rate * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_from_kaggle()

    # Always generate synthetic images & text (even for real Kaggle data,
    # since IEEE-CIS doesn't ship with check images or text descriptions).
    from src.data.generate_synthetic import generate_check_images, generate_text_descriptions

    generate_text_descriptions()
    generate_check_images()
