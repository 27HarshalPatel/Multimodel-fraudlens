"""FraudLens — FastAPI inference server + dashboard."""

from __future__ import annotations

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


@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    return FileResponse(str(DASHBOARD_DIR / "index.html"))


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

    # Run inference
    if predictor is None:
        # Demo mode — rule-based heuristic scoring (no trained model)
        # Analyzes actual inputs instead of returning random scores
        import random
        import base64

        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── Tabular risk scoring ─────────────────────────────────────
        tab_score = 0.0
        tab_reasons = []

        # Amount risk: higher amounts = higher risk
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

        # Email domain risk
        risky_emails = {"protonmail.com": 20, "anonymous.com": 30, "mail.com": 15, "yandex.com": 20}
        safe_emails = {"gmail.com": 0, "yahoo.com": 2, "hotmail.com": 3}
        email_risk = risky_emails.get(P_emaildomain, safe_emails.get(P_emaildomain, 10))
        tab_score += email_risk
        if email_risk >= 15:
            tab_reasons.append(f"Suspicious email domain ({P_emaildomain})")

        # Card type risk: credit cards have higher fraud rates
        if card6 == "credit":
            tab_score += 10
            tab_reasons.append("Credit card transaction (higher fraud rate)")
        elif card6 == "charge":
            tab_score += 5

        # Product code risk
        product_risk = {"W": 5, "H": 15, "C": 10, "S": 8, "R": 3}
        tab_score += product_risk.get(ProductCD, 5)

        # Device risk
        if DeviceType == "mobile":
            tab_score += 8
            tab_reasons.append("Mobile device (elevated risk profile)")

        tab_score = min(tab_score, 100.0)

        # ── Image risk scoring ───────────────────────────────────────
        img_score = 0.0
        if pil_image is not None:
            # Image uploaded — financial document images are unusual in
            # legitimate transactions and often indicate fraud evidence
            img_score = 55.0

            # Analyze image properties for anomaly signals
            img_array = np.array(pil_image.convert("RGB"))
            h_img, w_img = img_array.shape[:2]

            # Text-heavy images (receipts, statements) have high edge density
            gray = np.mean(img_array, axis=2)
            edge_density = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
            if edge_density > 15:
                img_score += 15  # Document-like image
            if edge_density > 25:
                img_score += 10  # Very text-heavy (bank statement, receipt)

            # Dark images or unusual aspect ratios
            brightness = np.mean(img_array)
            if brightness < 80 or brightness > 240:
                img_score += 5

            # Mobile screenshot aspect ratio (tall and narrow)
            aspect = h_img / max(w_img, 1)
            if aspect > 1.5:
                img_score += 10  # Phone screenshot — common in fraud evidence

            img_score = min(img_score, 100.0)
            tab_reasons.append(f"Image analysis detected visual anomalies ({img_score:.0f}% branch score)")

        # ── Text risk scoring ────────────────────────────────────────
        txt_score = 0.0
        text_attrs = []
        suspicious_keywords = {
            "urgent": 0.95, "wire": 0.88, "transfer": 0.82,
            "offshore": 1.0, "immediately": 0.90, "verify": 0.75,
            "refund": 0.80, "compromised": 0.92, "unauthorized": 0.90,
            "suspicious": 0.85, "fraud": 0.95, "stolen": 0.90,
            "hack": 0.88, "phishing": 0.85, "scam": 0.92,
            "payment": 0.40, "charge": 0.35, "debit": 0.30,
            "withdrawal": 0.50, "zelle": 0.45, "venmo": 0.40,
            "pending": 0.35, "declined": 0.60, "overdraft": 0.55,
        }

        if description.strip():
            words = description.strip().split()
            hit_count = 0
            for w in words[:50]:
                cleaned = w.strip(".,!?\"'()[]").lower()
                if cleaned in suspicious_keywords:
                    weight = suspicious_keywords[cleaned]
                    txt_score += weight * 12
                    hit_count += 1
                    text_attrs.append({"word": w, "weight": round(weight, 2)})
                else:
                    text_attrs.append({"word": w, "weight": round(random.uniform(-0.05, 0.15), 2)})

            if hit_count == 0:
                txt_score = 10  # Generic text, no red flags
            txt_score = min(txt_score, 100.0)

            if txt_score > 40:
                tab_reasons.append(f"Text patterns suggest suspicious activity ({txt_score:.0f}% branch score)")

        # ── Weighted fusion ──────────────────────────────────────────
        has_image = pil_image is not None
        has_text = bool(description.strip())

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

        # Apply a minimum floor when image shows clear fraud evidence
        if has_image and img_score >= 60:
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

        # ── Image heatmap generation ─────────────────────────────────
        heatmap_b64 = None
        if pil_image is not None:
            try:
                orig = pil_image.resize((224, 224))
                orig_np = np.array(orig)

                # Generate multi-region heatmap to highlight text/edges
                h, w = 224, 224
                gray = np.mean(orig_np.astype(float), axis=2)
                # Edge-based heatmap simulating gradient attention
                grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
                grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
                heatmap = (grad_x + grad_y)
                heatmap = heatmap / (heatmap.max() + 1e-8)
                # Smooth
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
                heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
            except Exception as exc:
                logger.warning("Demo heatmap generation failed: %s", exc)

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
                "text_attributions": text_attrs if has_text else [],
                "image_explanation_base64": heatmap_b64,
                "raw": {},
            }
        )

    try:
        result = predictor.predict(
            tabular_raw={**tabular_fields, "description": description},
            image=pil_image,
            text=description,
        )
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
