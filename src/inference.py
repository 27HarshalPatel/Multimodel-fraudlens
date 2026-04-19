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

    # Approximate median values for features not provided by the UI.
    # These are derived from the IEEE-CIS training data distribution
    # and ensure the model sees "typical normal transaction" values
    # instead of all-zeros, which would be far outside the training
    # distribution and cause the tabular branch to saturate.
    _V_MEDIANS = {
        "V1": 1.0, "V2": 1.0, "V3": 1.0, "V4": 1.0, "V5": 0.0,
        "V6": 1.0, "V7": 1.0, "V8": 1.0, "V9": 0.0, "V10": 0.0,
        "V11": 1.0, "V12": 1.0, "V13": 1.0, "V14": 1.0, "V15": 0.0,
        "V16": 0.0, "V17": 1.0, "V18": 0.0, "V19": 1.0, "V20": 1.0,
    }
    _C_MEDIANS = {
        "C1": 1.0, "C2": 1.0, "C3": 0.0, "C4": 0.0, "C5": 0.0,
        "C6": 1.0, "C7": 0.0, "C8": 0.0, "C9": 0.0, "C10": 0.0,
    }
    _D_MEDIANS = {
        "D1": 14.0, "D2": 14.0, "D3": 13.0, "D4": 94.0, "D5": 56.0,
    }

    def _preprocess_tabular(self, raw: dict) -> torch.Tensor:
        """Convert a flat dict of form fields into a 52-d float tensor.

        We mirror the feature engineering in ``TabularDataset.__init__``.
        Missing V/C/D features are filled with approximate training medians
        (from IEEE-CIS data) rather than zeros, which prevents the tabular
        branch from seeing an artificially extreme input pattern.
        """
        features: list[float] = []

        # 1) log_amount
        amt = float(raw.get("TransactionAmt", 0) or 0)
        features.append(np.log1p(max(amt, 0)))

        # 2) time_sin, time_cos  (use realistic mid-day default)
        dt = float(raw.get("TransactionDT", 43200) or 43200)
        seconds_in_day = 86400
        tod = dt % seconds_in_day
        features.append(float(np.sin(2 * np.pi * tod / seconds_in_day)))
        features.append(float(np.cos(2 * np.pi * tod / seconds_in_day)))

        # 3) Numeric cols (7)
        #    For cols not provided by UI, use realistic defaults
        NUMERIC_DEFAULTS = {
            "TransactionAmt": amt,
            "card1": 4000,   # Typical card1 hash value
            "card2": 321,    # Mid-range card2
            "card3": 150,    # Typical card3
            "card5": 225,    # Typical card5
            "addr1": 299,    # Most common addr1
            "addr2": 87,     # Most common addr2 (US)
        }
        for col in self.NUMERIC_COLS:
            val = raw.get(col)
            if val is not None and val != "" and val != 0:
                features.append(float(val))
            else:
                features.append(float(NUMERIC_DEFAULTS.get(col, 0)))

        # 4) V features (V1..V20 → 20 dims, fill with training medians)
        for i in range(1, 21):
            key = f"V{i}"
            val = raw.get(key)
            if val is not None and val != "" and val != 0:
                features.append(float(val))
            else:
                features.append(float(self._V_MEDIANS.get(key, 0.0)))

        # 5) C features (C1..C10 → 10 dims, fill with training medians)
        for i in range(1, 11):
            key = f"C{i}"
            val = raw.get(key)
            if val is not None and val != "" and val != 0:
                features.append(float(val))
            else:
                features.append(float(self._C_MEDIANS.get(key, 0.0)))

        # 6) D features (D1..D5 → 5 dims, fill with training medians)
        for i in range(1, 6):
            key = f"D{i}"
            val = raw.get(key)
            if val is not None and val != "" and val != 0:
                features.append(float(val))
            else:
                features.append(float(self._D_MEDIANS.get(key, 0.0)))

        # 7) Categorical cols label-encoded (7 dims)
        #    Use stable encoding maps that approximate training LabelEncoder output.
        #    hash(val) % 1000 produced extreme values the model never saw.
        CAT_ENCODING = {
            "ProductCD": {"w": 0, "h": 1, "c": 2, "s": 3, "r": 4},
            "card4": {"visa": 0, "mastercard": 1, "discover": 2, "american express": 3},
            "card6": {"debit": 0, "credit": 1, "charge": 2, "debit or credit": 3},
            "P_emaildomain": {
                "gmail.com": 0, "yahoo.com": 1, "hotmail.com": 2, "outlook.com": 3,
                "aol.com": 4, "icloud.com": 5, "mail.com": 6, "protonmail.com": 7,
                "comcast.net": 8, "yandex.com": 9, "anonymous.com": 10,
            },
            "R_emaildomain": {
                "gmail.com": 0, "yahoo.com": 1, "hotmail.com": 2, "outlook.com": 3,
            },
            "DeviceType": {"desktop": 0, "mobile": 1},
            "DeviceInfo": {
                "windows": 0, "ios device": 1, "macos": 2, "linux": 3,
                "trident/7.0": 4, "rv:11.0": 5,
            },
        }
        for col in self.CATEGORICAL_COLS:
            val = str(raw.get(col, "missing") or "missing").lower()
            col_map = CAT_ENCODING.get(col, {})
            # Use mapped value, or len(map) as a reasonable 'unknown' code
            features.append(float(col_map.get(val, len(col_map))))

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

        # ── Determine missing modalities (True means SKIP) ──
        has_image = image is not None
        has_text = text != "no description provided"
        
        mod_mask = torch.zeros((1, 3), dtype=torch.bool, device=self.device)
        if not has_image:
            mod_mask[0, 1] = True
        if not has_text:
            mod_mask[0, 2] = True

        # Forward pass
        outputs = self.model(
            tabular=tab_tensor,
            image=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            modality_mask=mod_mask,
        )

        # Extract scalars
        raw_prob = float(outputs["probability"].squeeze().cpu())
        tab_score = round(float(torch.sigmoid(outputs["tabular_logit"]).squeeze().cpu()) * 100, 1)
        img_score = round(float(torch.sigmoid(outputs["image_logit"]).squeeze().cpu()) * 100, 1)
        txt_score = round(float(torch.sigmoid(outputs["text_logit"]).squeeze().cpu()) * 100, 1)

        # Use the model's fused probability directly — this IS the attention output
        fraud_score = round(raw_prob * 100.0, 1)

        attn = outputs["attention_weights"].squeeze().cpu().numpy()
        # If the mask zeroed out certain attns, ensure others take 100% proportionally.
        if attn.sum() > 0:
            attn_pct = (attn / attn.sum() * 100).tolist()
        else:
            attn_pct = [33.3, 33.3, 33.3]

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
                tokens=tokens if has_text else None,
                img_score=img_score
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

    # ── Image-Only Inference ─────────────────────────────────────────

    @torch.no_grad()
    def predict_image_only(
        self,
        image: Image.Image,
        ocr_text: str = "",
    ) -> dict:
        """Run inference using only an image (and optional OCR text).

        Tabular features are zeroed out and masked so the fusion layer
        ignores the tabular branch entirely.

        Args:
            image: PIL Image to analyze.
            ocr_text: Text extracted via OCR from the image (optional).

        Returns:
            JSON-serialisable dict with fraud score, branch scores,
            attention weights, OCR text, and risk reasons.
        """
        # Zero tabular tensor
        tab_tensor = torch.zeros(1, 52, device=self.device)

        # Preprocess image
        img_tensor = self._preprocess_image(image)

        # Preprocess OCR text (or placeholder)
        text = ocr_text.strip() if ocr_text else "no description provided"
        has_text = text != "no description provided"
        input_ids, attention_mask = self._preprocess_text(text)

        # Modality mask: skip tabular, keep image and text
        mod_mask = torch.zeros((1, 3), dtype=torch.bool, device=self.device)
        mod_mask[0, 0] = True   # Mask tabular (not provided)
        if not has_text:
            mod_mask[0, 2] = True  # Mask text if no OCR text

        # Forward pass
        outputs = self.model(
            tabular=tab_tensor,
            image=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            modality_mask=mod_mask,
        )

        # Extract scalars
        raw_prob = float(outputs["probability"].squeeze().cpu())
        img_score = round(float(torch.sigmoid(outputs["image_logit"]).squeeze().cpu()) * 100, 1)
        txt_score = round(float(torch.sigmoid(outputs["text_logit"]).squeeze().cpu()) * 100, 1)
        tab_score = 0.0  # Not applicable

        # Score calculation — use model's fusion output directly
        fraud_score = round(raw_prob * 100.0, 1)

        attn = outputs["attention_weights"].squeeze().cpu().numpy()
        if attn.sum() > 0:
            attn_pct = (attn / attn.sum() * 100).tolist()
        else:
            attn_pct = [0.0, 60.0, 40.0]

        # Force tabular attention to 0 for display
        attn_pct[0] = 0.0
        remaining = attn_pct[1] + attn_pct[2]
        if remaining > 0:
            attn_pct[1] = round(attn_pct[1] / remaining * 100, 1)
            attn_pct[2] = round(attn_pct[2] / remaining * 100, 1)

        if not has_text:
            txt_score = round(txt_score * 0.3, 1)

        # Risk level
        if fraud_score >= 80:
            risk_level = "Critical"
        elif fraud_score >= 60:
            risk_level = "High"
        elif fraud_score >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Risk reasons
        reasons = []
        if img_score > 50:
            reasons.append(f"Image analysis detected visual anomalies ({img_score}% branch score)")
        if has_text and txt_score > 50:
            reasons.append(f"OCR text contains suspicious patterns ({txt_score}% branch score)")
        elif has_text:
            reasons.append(f"OCR text analyzed ({txt_score}% branch score)")

        dominant = 1 if attn_pct[1] >= attn_pct[2] else 2
        labels = ["tabular", "image", "text"]
        reasons.append(
            f"Attention is highest on {labels[dominant]} modality "
            f"({attn_pct[dominant]:.1f}%)"
        )

        if fraud_score < 15:
            reasons.append("Overall pattern is consistent with legitimate content")

        # Explanations
        explanations = {}
        if self.explainer is not None:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            explanations = self.explainer.explain(
                tab_tensor, img_tensor, input_ids, attention_mask,
                pil_image=image,
                tokens=tokens if has_text else None,
                img_score=img_score
            )

        return {
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "attention_weights": {
                "tabular": 0.0,
                "image": round(attn_pct[1], 1),
                "text": round(attn_pct[2], 1),
            },
            "branch_scores": {
                "tabular": 0.0,
                "image": img_score,
                "text": txt_score,
            },
            "risk_reasons": reasons or ["No specific risk factors identified"],
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
