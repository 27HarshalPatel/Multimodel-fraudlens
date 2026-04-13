"""FraudLens — FastAPI inference server + dashboard."""

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
    from src.inference import FraudLensPredictor

    try:
        ckpt = Path(__file__).parent / "checkpoints" / "best_model.pt"
        predictor = FraudLensPredictor(checkpoint_path=ckpt)
        logger.info("FraudLens model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.info("Server will run in demo mode with synthetic responses")


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
        # Demo mode — return synthetic response
        import random

        score = random.uniform(10, 90)

        # Build text attributions from description when provided
        text_attrs = []
        if description.strip():
            words = description.strip().split()
            suspicious = {"urgent", "wire", "transfer", "offshore",
                          "immediately", "verify", "refund", "compromised"}
            for w in words[:30]:
                cleaned = w.strip(".,!?").lower()
                wt = round(random.uniform(0.7, 1.0), 2) if cleaned in suspicious else round(random.uniform(-0.1, 0.3), 2)
                text_attrs.append({"word": w, "weight": wt})
        else:
            text_attrs = [
                {"word": "urgent", "weight": 0.95},
                {"word": "wire", "weight": 0.8},
                {"word": "transfer", "weight": 0.75},
                {"word": "to", "weight": 0.1},
                {"word": "offshore", "weight": 1.0},
                {"word": "account", "weight": 0.5},
            ]

        # Simulated heatmap for uploaded images
        heatmap_b64 = None
        if pil_image is not None:
            try:
                import base64
                import numpy as np
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                orig = pil_image.resize((224, 224))
                orig_np = np.array(orig)

                # Create Gaussian-blob heatmap
                h, w = 224, 224
                cx, cy = random.randint(60, 160), random.randint(60, 160)
                Y, X = np.ogrid[:h, :w]
                heatmap = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 40**2))

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

        return JSONResponse(
            {
                "fraud_score": round(score, 2),
                "risk_level": (
                    "CRITICAL"
                    if score >= 80
                    else "HIGH"
                    if score >= 60
                    else "MEDIUM"
                    if score >= 35
                    else "LOW"
                ),
                "attention_weights": {
                    "tabular": 100.0 if not pil_image and not description else 40.0,
                    "image": 30.0 if pil_image else 0.0,
                    "text": 30.0 if description else 0.0,
                },
                "branch_scores": {
                    "tabular": round(score * 0.9, 2),
                    "image": round(score * 0.7, 2) if pil_image else 0.0,
                    "text": round(score * 0.6, 2) if description else 0.0,
                },
                "risk_reasons": [
                    "⚠️ Running in demo mode (model not loaded)",
                    f"Simulated fraud score: {score:.1f}%",
                ],
                "text_attributions": text_attrs if description.strip() else [],
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
