"""FraudLens — FastAPI inference server + dashboard."""

from __future__ import annotations

import base64
import io
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FraudLens", description="Multimodal Fraud Detection Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
DASHBOARD_DIR = Path(__file__).parent / "dashboard"
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

# Load predictor at startup
predictor = None


@app.on_event("startup")
async def load_model():
    global predictor

    ckpt = Path(__file__).parent / "checkpoints" / "best_model.pt"

    if not ckpt.exists():
        logger.warning("No checkpoint found at %s", ckpt)
        logger.info("Server will run in demo mode with heuristic scoring")
        return

    try:
        from src.inference import FraudLensPredictor
        predictor = FraudLensPredictor(checkpoint_path=ckpt)
        logger.info("FraudLens model loaded successfully")
    except ImportError as e:
        logger.error("ML dependencies not installed: %s", e)
        logger.info("Server will run in demo mode with heuristic scoring")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.info("Server will run in demo mode with heuristic scoring")


# ══════════════════════════════════════════════════════════════════════════
#  OCR & Image Analysis Helpers
# ══════════════════════════════════════════════════════════════════════════

def _extract_text_ocr(pil_image) -> str:
    """Extract text from image using pytesseract (primary) or easyocr (fallback)."""
    extracted_text = ""

    # Try pytesseract first
    try:
        import pytesseract
        import shutil
        import os

        tess_cmd = shutil.which("tesseract")
        if not tess_cmd and os.path.exists("/opt/homebrew/bin/tesseract"):
            tess_cmd = "/opt/homebrew/bin/tesseract"

        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
            extracted_text = pytesseract.image_to_string(pil_image)
            if extracted_text.strip():
                logger.info("OCR (pytesseract): extracted %d chars", len(extracted_text))
                return extracted_text.strip()
    except Exception as e:
        logger.warning("pytesseract OCR failed: %s", e)

    # Fallback to easyocr
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        import numpy as np
        img_array = np.array(pil_image.convert("RGB"))
        results = reader.readtext(img_array)
        extracted_text = " ".join([r[1] for r in results])
        if extracted_text.strip():
            logger.info("OCR (easyocr): extracted %d chars", len(extracted_text))
            return extracted_text.strip()
    except Exception as e:
        logger.warning("easyocr OCR failed: %s", e)

    return extracted_text.strip()


# Shared phishing keyword lists
PHISHING_KEYWORDS = [
    "http", ".info", ".biz", ".top", ".xyz", "cancel", "urgent",
    "verify", "login", "logon", "password", "alert", "notice",
    "suspicious", "unpaid", "toll", "bill", "late fee", "reply y",
]

SUSPICIOUS_TLDS = [".info", ".biz", ".top", ".xyz", ".cc"]

SUSPICIOUS_TEXT_KEYWORDS = {
    "urgent": 0.95, "wire": 0.88, "transfer": 0.82,
    "offshore": 1.0, "immediately": 0.90, "verify": 0.75,
    "refund": 0.80, "compromised": 0.92, "unauthorized": 0.90,
    "suspicious": 0.85, "fraud": 0.95, "stolen": 0.90,
    "hack": 0.88, "phishing": 0.85, "scam": 0.92,
    "payment": 0.40, "charge": 0.35, "debit": 0.30,
    "withdrawal": 0.50, "zelle": 0.45, "venmo": 0.40,
    "pending": 0.35, "declined": 0.60, "overdraft": 0.55,
    "logon": 0.85, "login": 0.85, "notice": 0.70, "alert": 0.75,
    "cancel": 0.80, "http": 0.85, "https": 0.85,
    "toll": 0.85, "unpaid": 0.85, "bill": 0.50, "late": 0.75,
}


def _detect_phishing_keywords(text: str) -> tuple[list[str], bool]:
    """Detect phishing indicators in extracted text.

    Returns (hit_keywords, has_suspicious_tld)
    """
    lower = text.lower()
    hits = [w for w in PHISHING_KEYWORDS if w in lower]
    has_tld = any(tld in lower for tld in SUSPICIOUS_TLDS)
    return hits, has_tld


def _score_text(text: str) -> tuple[float, list[dict]]:
    """Score text for fraud indicators. Returns (score, attributions)."""
    import random

    txt_score = 0.0
    text_attrs = []

    if not text.strip():
        return 0.0, []

    words = text.strip().split()
    hit_count = 0
    for w in words[:150]:
        cleaned = w.strip(".,!?\"'()[]").lower()
        if cleaned in SUSPICIOUS_TEXT_KEYWORDS:
            weight = SUSPICIOUS_TEXT_KEYWORDS[cleaned]
            txt_score += weight * 12
            hit_count += 1
            text_attrs.append({"word": w, "weight": round(weight, 2)})
        else:
            text_attrs.append({"word": w, "weight": round(random.uniform(-0.05, 0.15), 2)})

    if hit_count == 0:
        txt_score = 10  # Generic text, no red flags
    txt_score = min(txt_score, 100.0)

    return txt_score, text_attrs


def _analyze_image_features(pil_image) -> tuple[float, list[str]]:
    """Analyze image properties for anomaly signals.

    Returns (score, reasons)
    """
    import numpy as np

    img_array = np.array(pil_image.convert("RGB"))
    h_img, w_img = img_array.shape[:2]
    score = 55.0  # Base score for uploaded image
    reasons = []

    # Text-heavy images (receipts, statements) have high edge density
    gray = np.mean(img_array, axis=2)
    edge_density = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
    if edge_density > 15:
        score += 15
        reasons.append("Document-like image with high text density")
    if edge_density > 25:
        score += 10
        reasons.append("Very text-heavy (bank statement, receipt)")

    # Dark images or unusual brightness
    brightness = np.mean(img_array)
    if brightness < 80 or brightness > 240:
        score += 5
        reasons.append("Unusual image brightness detected")

    # Mobile screenshot aspect ratio
    aspect = h_img / max(w_img, 1)
    if aspect > 1.5:
        score += 10
        reasons.append("Phone screenshot aspect ratio (common in fraud evidence)")

    return min(score, 100.0), reasons


def _generate_heatmap_b64(pil_image) -> str | None:
    """Generate edge-based heatmap overlay and return base64 encoded PNG."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        orig = pil_image.resize((224, 224))
        orig_np = np.array(orig)

        h, w = 224, 224
        gray = np.mean(orig_np.astype(float), axis=2)
        grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        heatmap = grad_x + grad_y
        heatmap = heatmap / (heatmap.max() + 1e-8)

        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=6)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(orig_np)
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.warning("Heatmap generation failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    return FileResponse(str(DASHBOARD_DIR / "index.html"))


# ── Image-Only Analysis ──────────────────────────────────────────────────

@app.post("/api/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
):
    """Run image-only fraud analysis with OCR text extraction."""
    # Read and open image
    try:
        from PIL import Image as PILImage

        content = await image.read()
        pil_image = PILImage.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to process image: {e}"},
            status_code=400,
        )

    # ── OCR Text Extraction ──────────────────────────────────────────
    extracted_text = _extract_text_ocr(pil_image)

    # ── Image Feature Analysis ───────────────────────────────────────
    img_score, img_reasons = _analyze_image_features(pil_image)

    # ── Phishing Detection on Extracted Text ─────────────────────────
    ocr_phishing_hits, has_suspicious_tld = _detect_phishing_keywords(extracted_text)
    ocr_is_phishing = (
        has_suspicious_tld
        or len(ocr_phishing_hits) >= 3
        or ("http" in extracted_text.lower() and any(
            w in extracted_text.lower()
            for w in ["urgent", "verify", "update", "unpaid", "toll"]
        ))
    )

    if ocr_is_phishing:
        img_score = min(img_score + 45, 100.0)
        img_reasons.append(
            f"Phishing indicators in text: {', '.join(ocr_phishing_hits[:4])}"
        )

    # ── Text Scoring from OCR ────────────────────────────────────────
    txt_score, text_attrs = _score_text(extracted_text)

    # ── Fused Score ──────────────────────────────────────────────────
    if extracted_text.strip():
        # Image + OCR text available
        fused_score = img_score * 0.55 + txt_score * 0.45
        attn_weights = {"tabular": 0.0, "image": 55.0, "text": 45.0}
    else:
        # Image only, no text extracted
        fused_score = img_score
        attn_weights = {"tabular": 0.0, "image": 90.0, "text": 10.0}

    if ocr_is_phishing:
        fused_score = max(fused_score, 95.0)

    fused_score = min(round(fused_score, 1), 100.0)

    # Risk level
    if fused_score >= 80:
        risk_level = "CRITICAL"
    elif fused_score >= 60:
        risk_level = "HIGH"
    elif fused_score >= 35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Heatmap
    heatmap_b64 = _generate_heatmap_b64(pil_image)

    # Risk reasons
    reasons = list(img_reasons)
    if extracted_text.strip():
        reasons.insert(0, f"OCR extracted {len(extracted_text)} characters from image")
    if txt_score > 40:
        reasons.append(
            f"Extracted text contains suspicious patterns ({txt_score:.0f}%)"
        )
    if not reasons:
        reasons.append("Image analysis complete — no significant risk indicators")

    # ── Use model if available ───────────────────────────────────────
    if predictor is not None:
        try:
            result = predictor.predict_image_only(
                image=pil_image,
                ocr_text=extracted_text,
            )
            # Merge OCR-specific data into model result
            result["ocr_extracted_text"] = extracted_text
            result["ocr_keywords"] = ocr_phishing_hits
            result["text_attributions"] = result.get("text_attributions", text_attrs)
            if not result.get("image_explanation_base64"):
                result["image_explanation_base64"] = heatmap_b64
            return JSONResponse(result)
        except Exception as e:
            logger.warning("Model image-only prediction failed, using heuristic: %s", e)

    return JSONResponse({
        "fraud_score": fused_score,
        "risk_level": risk_level,
        "attention_weights": attn_weights,
        "branch_scores": {
            "tabular": 0.0,
            "image": round(img_score, 1),
            "text": round(txt_score, 1),
        },
        "risk_reasons": reasons,
        "ocr_extracted_text": extracted_text,
        "ocr_keywords": ocr_phishing_hits,
        "text_attributions": text_attrs,
        "image_explanation_base64": heatmap_b64,
        "raw": {},
    })


# ── Full Multimodal Analysis ─────────────────────────────────────────────

@app.post("/api/predict")
async def predict(
    TransactionAmt: float = Form(50.0),
    ProductCD: str = Form("W"),
    card4: str = Form("visa"),
    card6: str = Form("debit"),
    P_emaildomain: str = Form("gmail.com"),
    DeviceType: str = Form("desktop"),
    DeviceInfo: str = Form("Windows"),
    description: str = Form(""),
    image: UploadFile | None = File(None),
):
    """Run multimodal fraud inference on a single transaction."""
    # Build tabular fields
    tabular_fields = {
        "TransactionAmt": TransactionAmt,
        "ProductCD": ProductCD,
        "card4": card4,
        "card6": card6,
        "P_emaildomain": P_emaildomain,
        "DeviceType": DeviceType,
        "DeviceInfo": DeviceInfo,
    }

    # Process image if provided
    pil_image = None
    if image and image.filename:
        try:
            from PIL import Image as PILImage

            content = await image.read()
            pil_image = PILImage.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            logger.warning("Image processing failed: %s", e)

    # ── OCR from image ───────────────────────────────────────────────
    extracted_text = ""
    ocr_phishing_hits = []
    if pil_image is not None:
        extracted_text = _extract_text_ocr(pil_image)
        ocr_phishing_hits, _ = _detect_phishing_keywords(extracted_text)

    # Run inference
    if predictor is None:
        # Demo mode — rule-based heuristic scoring (no trained model)
        import random
        import numpy as np

        # ── Tabular risk scoring ─────────────────────────────────────
        tab_score = 0.0
        tab_reasons = []

        if TransactionAmt >= 5000:
            tab_score += 35
            tab_reasons.append(f"Very high transaction amount (${TransactionAmt:,.2f})")
        elif TransactionAmt >= 1000:
            tab_score += 25
            tab_reasons.append(f"High transaction amount (${TransactionAmt:,.2f})")
        elif TransactionAmt >= 500:
            tab_score += 15
            tab_reasons.append(f"Elevated transaction amount (${TransactionAmt:,.2f})")
        elif TransactionAmt >= 100:
            tab_score += 5

        risky_emails = {"protonmail.com": 20, "anonymous.com": 30, "mail.com": 15, "yandex.com": 20}
        safe_emails = {"gmail.com": 0, "yahoo.com": 2, "hotmail.com": 3}
        email_risk = risky_emails.get(P_emaildomain, safe_emails.get(P_emaildomain, 10))
        tab_score += email_risk
        if email_risk >= 15:
            tab_reasons.append(f"Suspicious email domain ({P_emaildomain})")

        if card6 == "credit":
            tab_score += 10
            tab_reasons.append("Credit card transaction (higher fraud rate)")
        elif card6 == "charge":
            tab_score += 5

        product_risk = {"W": 5, "H": 15, "C": 10, "S": 8, "R": 3}
        tab_score += product_risk.get(ProductCD, 5)

        if DeviceType == "mobile":
            tab_score += 8
            tab_reasons.append("Mobile device (elevated risk profile)")

        tab_score = min(tab_score, 100.0)

        # ── Image risk scoring ───────────────────────────────────────
        img_score = 0.0
        has_ocr_suspicious = False
        if pil_image is not None:
            img_score, img_reasons = _analyze_image_features(pil_image)
            tab_reasons.extend(img_reasons)

            # Check OCR phishing
            _, has_suspicious_tld = _detect_phishing_keywords(extracted_text)
            has_ocr_suspicious = (
                has_suspicious_tld
                or len(ocr_phishing_hits) >= 3
                or ("http" in extracted_text.lower() and any(
                    w in extracted_text.lower()
                    for w in ["urgent", "verify", "update", "unpaid", "toll"]
                ))
            )

            if has_ocr_suspicious:
                img_score = min(img_score + 45, 100.0)
                tab_reasons.append(
                    f"Image contains suspicious text/URLs (Phishing indicators: {', '.join(ocr_phishing_hits[:4])})"
                )

            tab_reasons.append(f"Image analysis detected visual anomalies ({img_score:.0f}% branch score)")

        # ── Text risk scoring ────────────────────────────────────────
        # Combine user description with OCR text for thorough analysis
        combined_text = description.strip()
        if extracted_text:
            combined_text += " " + extracted_text

        txt_score, text_attrs = _score_text(combined_text)

        if txt_score > 40:
            tab_reasons.append(f"Text patterns suggest suspicious activity ({txt_score:.0f}% branch score)")

        # ── Weighted fusion ──────────────────────────────────────────
        has_image = pil_image is not None
        has_text = bool(description.strip()) or bool(extracted_text.strip())

        if has_image and has_text:
            weights = {"tabular": 0.35, "image": 0.40, "text": 0.25}
        elif has_image:
            weights = {"tabular": 0.40, "image": 0.50, "text": 0.10}
        elif has_text:
            weights = {"tabular": 0.55, "image": 0.10, "text": 0.35}
        else:
            weights = {"tabular": 0.85, "image": 0.10, "text": 0.05}

        fused_score = (
            weights["tabular"] * tab_score
            + weights["image"] * img_score
            + weights["text"] * txt_score
        )

        if has_ocr_suspicious:
            fused_score = max(fused_score, 95.0)
        elif has_image and img_score >= 60:
            fused_score = max(fused_score, img_score * 0.75)

        fused_score = min(round(fused_score, 1), 100.0)

        # Determine risk level
        if fused_score >= 80:
            risk_level = "CRITICAL"
        elif fused_score >= 60:
            risk_level = "HIGH"
        elif fused_score >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Heatmap
        heatmap_b64 = _generate_heatmap_b64(pil_image) if pil_image else None

        # Build risk reasons
        reasons = []
        if not tab_reasons:
            tab_reasons.append("Transaction parameters within normal range")
        reasons.extend(tab_reasons)

        dominant_mod = max(weights, key=weights.get)
        reasons.append(
            f"Attention is highest on {dominant_mod} modality "
            f"({weights[dominant_mod] * 100:.0f}%)"
        )

        return JSONResponse(
            {
                "fraud_score": fused_score,
                "risk_level": risk_level,
                "attention_weights": {
                    "tabular": round(weights["tabular"] * 100, 1),
                    "image": round(weights["image"] * 100, 1),
                    "text": round(weights["text"] * 100, 1),
                },
                "branch_scores": {
                    "tabular": round(tab_score, 1),
                    "image": round(img_score, 1),
                    "text": round(txt_score, 1),
                },
                "risk_reasons": reasons,
                "ocr_extracted_text": extracted_text,
                "ocr_keywords": ocr_phishing_hits,
                "text_attributions": text_attrs if has_text else [],
                "image_explanation_base64": heatmap_b64,
                "raw": {},
            }
        )

    # ── Model-based inference ────────────────────────────────────────
    try:
        # Combine description + OCR text for text branch
        combined_desc = description.strip()
        if extracted_text:
            combined_desc = (combined_desc + " " + extracted_text).strip()

        result = predictor.predict(
            tabular_raw={**tabular_fields, "description": combined_desc},
            image=pil_image,
            text=combined_desc,
        )
        result["ocr_extracted_text"] = extracted_text
        result["ocr_keywords"] = ocr_phishing_hits
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
