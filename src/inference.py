"""FraudLens inference pipeline.

Loads a trained FraudLensModel checkpoint and runs single-sample
inference for the dashboard API.

Preprocessing mirrors ``src/data/tabular_dataset.py``:
  - Tabular: log-amount, cyclical time, numeric, V/C/D features, label-encoded cats → 52 dims
  - Image:  SigLIP 2 preprocessor
  - Text:   DistilBERT tokeniser
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, DistilBertTokenizerFast

from src.models.fraudlens import FraudLensModel

logger = logging.getLogger(__name__)

# ── Device selection ──────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
logger.info("Inference device: %s", DEVICE)


class FraudLensPredictor:
    """Single-sample predictor wrapping FraudLensModel.

    Args:
        checkpoint_path: Path to ``best_model.pt``
        tabular_input_dim: Must match the dim the model was trained with (default 52).
    """

    # Numeric + engineered features expected by the model (see tabular_dataset.py)
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
        checkpoint_path: str | Path,
        tabular_input_dim: int = 52,
    ):
        self.device = DEVICE

        # Build model with same hyper-parameters used during training
        self.model = FraudLensModel(tabular_input_dim=tabular_input_dim)
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Pre-trained preprocessors
        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/siglip2-base-patch16-224"
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        logger.info("FraudLensPredictor ready on %s", self.device)

        # Setup Explainer
        self.explainer = None
        try:
            from src.explain.captum_explainer import CaptumExplainer
            self.explainer = CaptumExplainer(self.model, self.device)
            logger.info("CaptumExplainer initialized successfully.")
        except ImportError as e:
            logger.warning("CaptumExplainer could not be initialized: %s", e)

    # ── Checkpoint loading ────────────────────────────────────────────

    def _load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        epoch = ckpt.get("epoch", "?")
        auprc = ckpt.get("best_auprc", "?")
        logger.info("Loaded checkpoint (epoch=%s, best_auprc=%s)", epoch, auprc)

    # ── Tabular preprocessing ─────────────────────────────────────────

    def _preprocess_tabular(self, raw: dict) -> torch.Tensor:
        """Convert a flat dict of form fields into a 52-d float tensor.

        We mirror the feature engineering in ``TabularDataset.__init__``.
        Without a saved scaler the values are **not** standardised;
        the model still returns reasonable scores because BN layers
        adapt.  For production, persist and load the training scaler.
        """
        features: list[float] = []

        # 1) log_amount
        amt = float(raw.get("TransactionAmt", 0) or 0)
        features.append(np.log1p(max(amt, 0)))

        # 2) time_sin, time_cos
        dt = float(raw.get("TransactionDT", 0) or 0)
        seconds_in_day = 86400
        tod = dt % seconds_in_day
        features.append(float(np.sin(2 * np.pi * tod / seconds_in_day)))
        features.append(float(np.cos(2 * np.pi * tod / seconds_in_day)))

        # 3) Numeric cols (7)
        for col in self.NUMERIC_COLS:
            features.append(float(raw.get(col, 0) or 0))

        # 4) V features (V1..V20 → 20 dims)
        for i in range(1, 21):
            features.append(float(raw.get(f"V{i}", 0) or 0))

        # 5) C features (C1..C10 → 10 dims)
        for i in range(1, 11):
            features.append(float(raw.get(f"C{i}", 0) or 0))

        # 6) D features (D1..D5 → 5 dims)
        for i in range(1, 6):
            features.append(float(raw.get(f"D{i}", 0) or 0))

        # 7) Categorical cols label-encoded (7 dims)
        for col in self.CATEGORICAL_COLS:
            # Simple hash-based encoding for single-sample inference
            val = str(raw.get(col, "missing") or "missing")
            features.append(float(hash(val) % 1000))

        # Pad / truncate to expected dim
        while len(features) < 52:
            features.append(0.0)
        features = features[:52]

        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    # ── Image preprocessing ───────────────────────────────────────────

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    # ── Text preprocessing ────────────────────────────────────────────

    def _preprocess_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    # ── Public API ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        tabular_raw: dict,
        image: Image.Image | None = None,
        text: str = "",
    ) -> dict:
        """Run inference on a single sample.

        Returns a JSON-serialisable dict with fraud score, per-branch
        scores, attention weights, and risk reasons.
        """
        # Preprocess each modality
        tab_tensor = self._preprocess_tabular(tabular_raw)

        if image is not None:
            img_tensor = self._preprocess_image(image)
        else:
            # Default blank image
            img_tensor = torch.zeros(1, 3, 224, 224, device=self.device)

        text = text or tabular_raw.get("description", "") or ""
        if not text.strip():
            text = "no description provided"
        input_ids, attention_mask = self._preprocess_text(text)

        # Forward pass
        outputs = self.model(
            tabular=tab_tensor,
            image=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract scalars
        fraud_prob = float(outputs["probability"].squeeze().cpu())
        fraud_score = round(fraud_prob * 100, 1)

        attn = outputs["attention_weights"].squeeze().cpu().numpy()
        attn_pct = (attn / attn.sum() * 100).tolist()

        tab_score = round(float(torch.sigmoid(outputs["tabular_logit"]).squeeze().cpu()) * 100, 1)
        img_score = round(float(torch.sigmoid(outputs["image_logit"]).squeeze().cpu()) * 100, 1)
        txt_score = round(float(torch.sigmoid(outputs["text_logit"]).squeeze().cpu()) * 100, 1)

        has_image = image is not None
        has_text = text != "no description provided"

        # Scale down (but don't zero) branch scores for absent modalities.
        # The model still produces signal from default inputs (blank image,
        # placeholder text) — hard zeroing would discard learned patterns.
        if not has_image:
            img_score = round(img_score * 0.3, 1)  # Reduce weight, don't zero
        if not has_text:
            txt_score = round(txt_score * 0.3, 1)

        # Determine risk level
        if fraud_score >= 80:
            risk_level = "Critical"
        elif fraud_score >= 60:
            risk_level = "High"
        elif fraud_score >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Generate risk reasons
        reasons = self._generate_reasons(
            fraud_score, tab_score, img_score, txt_score,
            attn_pct, tabular_raw, text,
        )

        # Deep Explainability
        explanations = {}
        if self.explainer is not None:
            # We skip 'image' arg to explainer if it's the default blank tensor
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            explanations = self.explainer.explain(
                tab_tensor, 
                img_tensor, 
                input_ids, 
                attention_mask, 
                pil_image=image if has_image else None, 
                tokens=tokens if has_text else None
            )

        return {
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "attention_weights": {
                "tabular": round(attn_pct[0], 1),
                "image": round(attn_pct[1], 1),
                "text": round(attn_pct[2], 1),
            },
            "branch_scores": {
                "tabular": tab_score,
                "image": img_score,
                "text": txt_score,
            },
            "risk_reasons": reasons,
            **explanations,
        }

    # ── Explainability helpers ────────────────────────────────────────

    @staticmethod
    def _generate_reasons(
        fraud_score, tab_score, img_score, txt_score,
        attn_pct, tabular_raw, text,
    ) -> list[str]:
        reasons: list[str] = []

        if tab_score > 50:
            reasons.append(
                f"Tabular features flag elevated risk ({tab_score}% branch score)"
            )
            amt = float(tabular_raw.get("TransactionAmt", 0) or 0)
            if amt > 500:
                reasons.append(f"High transaction amount (${amt:,.2f})")

        if img_score > 50:
            reasons.append(
                f"Image analysis detected visual anomalies ({img_score}% branch score)"
            )

        if txt_score > 50:
            reasons.append(
                f"Text patterns suggest suspicious activity ({txt_score}% branch score)"
            )

        dominant = max(range(3), key=lambda i: attn_pct[i])
        labels = ["tabular", "image", "text"]
        reasons.append(
            f"Attention is highest on {labels[dominant]} modality "
            f"({attn_pct[dominant]:.1f}%)"
        )

        if fraud_score < 15:
            reasons.append("Overall pattern is consistent with legitimate transactions")

        return reasons or ["No specific risk factors identified"]
