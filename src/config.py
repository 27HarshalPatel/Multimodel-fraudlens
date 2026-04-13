"""Default configuration for FraudLens training."""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
TABULAR_DIR = DATA_DIR / "tabular"
IMAGE_DIR = DATA_DIR / "images"
TEXT_PATH = DATA_DIR / "text" / "descriptions.csv"

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("runs")

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
IMAGE_MODEL = "google/siglip2-base-patch16-224"
TEXT_MODEL = "distilbert-base-uncased"
IMAGE_FREEZE_LAYERS = 8
TEXT_FREEZE_LAYERS = 4
TABULAR_HIDDEN_DIMS = [256, 128]
TABULAR_DROPOUT = 0.3

# Fusion
FUSION_NUM_HEADS = 4
FUSION_DROPOUT = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0
MIXED_PRECISION = True

# Focal loss
FOCAL_LOSS_ALPHA = 0.75
FOCAL_LOSS_GAMMA = 2.0

# Early stopping
EARLY_STOPPING_PATIENCE = 7

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
TEXT_MAX_LENGTH = 128
IMAGE_SIZE = 224
SAMPLE_SIZE = None  # None = use all data
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 0  # 0 = main process only (avoids Docker /dev/shm issues)


def get_training_config() -> dict:
    """Return training config as a flat dict."""
    return {
        "tabular_dir": str(TABULAR_DIR),
        "image_dir": str(IMAGE_DIR),
        "text_path": str(TEXT_PATH),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "log_dir": str(LOG_DIR),
        "embedding_dim": EMBEDDING_DIM,
        "image_model": IMAGE_MODEL,
        "text_model": TEXT_MODEL,
        "image_freeze_layers": IMAGE_FREEZE_LAYERS,
        "text_freeze_layers": TEXT_FREEZE_LAYERS,
        "tabular_hidden_dims": TABULAR_HIDDEN_DIMS,
        "tabular_dropout": TABULAR_DROPOUT,
        "fusion_num_heads": FUSION_NUM_HEADS,
        "fusion_dropout": FUSION_DROPOUT,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mixed_precision": MIXED_PRECISION,
        "focal_loss_alpha": FOCAL_LOSS_ALPHA,
        "focal_loss_gamma": FOCAL_LOSS_GAMMA,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "text_max_length": TEXT_MAX_LENGTH,
        "image_size": IMAGE_SIZE,
        "sample_size": SAMPLE_SIZE,
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
    }
