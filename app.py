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
    # Document fraud keywords
    "fake": 1.0, "forged": 1.0, "counterfeit": 1.0, "forgery": 1.0,
    "fabricated": 0.95, "falsified": 0.95, "fraudulent": 0.95,
    "abuse": 0.90, "tampered": 0.90, "altered": 0.80,
    "specimen": 0.85, "sample": 0.70, "template": 0.75,
    "placeholder": 0.85, "dummy": 0.80, "test": 0.50,
}

# Phrases that are dead giveaways of document fraud (checked as substrings)
DOCUMENT_FRAUD_PHRASES = [
    "fake receipt", "fake invoice", "fake check", "fake cheque",
    "forged document", "forged receipt", "counterfeit",
    "not valid", "void", "specimen", "sample only",
    "interbank giro abuse",  # Specific to this type of fake receipt
]


def _detect_document_fraud(text: str) -> tuple[float, list[str]]:
    """Detect document fraud indicators in OCR text.

    Returns (fraud_boost_score, reasons) where fraud_boost_score is 0-100.
    This catches textual fraud signals the ML model misses because it was
    trained on transaction metadata, not document content analysis.

    Detection layers:
      1. Dead-giveaway keyword/phrase matching
      2. Placeholder amount detection
      3. Balance calculation verification
      4. Gibberish / nonsensical text detection
      5. Future date detection
      6. Structural anomalies (duplicate dates, missing fields)
    """
    import re
    from datetime import datetime

    lower = text.lower()
    score = 0.0
    reasons = []
    red_flags = 0  # Count independent red flags for cumulative scoring

    # ── 1. Dead-giveaway phrases ─────────────────────────────────────
    for phrase in DOCUMENT_FRAUD_PHRASES:
        if phrase in lower:
            score = max(score, 95.0)
            reasons.append(f"Document contains fraud indicator: '{phrase}'")

    # ── 2. Placeholder amounts (e.g., -RMXXX, $XXX, XXXXX) ──────────
    placeholder_patterns = [
        r'-rm[x]+',       # -RMXXX, -RMXXXX
        r'\$[x]+',        # $XXX
        r'\b[x]{3,}\b',   # XXX, XXXX (standalone)
    ]
    for pat in placeholder_patterns:
        if re.search(pat, lower):
            score = max(score, 85.0)
            reasons.append("Document contains placeholder amounts (not real monetary values)")
            red_flags += 1
            break

    # ── 3. Balance calculation verification ──────────────────────────
    # Look for monetary amounts and check if running balances are consistent
    money_pattern = r'[-+]?\$?[\d,]+\.\d{2}'
    amounts = re.findall(money_pattern, text)
    if len(amounts) >= 4:
        # Parse amounts to floats
        parsed = []
        for a in amounts:
            try:
                val = float(a.replace('$', '').replace(',', '').replace('+', ''))
                parsed.append(val)
            except ValueError:
                continue

        if len(parsed) >= 4:
            # Check for impossible balance jumps (e.g., balance goes UP after withdrawal)
            # Look for patterns like: withdrawal=-154.67, balance stays same or increases
            balance_errors = 0
            for i in range(len(parsed) - 2):
                curr = parsed[i]
                nxt = parsed[i + 1]
                # If both are "balance-sized" (>100) and differ by impossible amounts
                if curr > 100 and nxt > 100:
                    diff = abs(nxt - curr)
                    # If balance changes by implausible amount (>1000% of previous)
                    if diff > curr * 10 and diff > 1000:
                        balance_errors += 1

            # Check if running balance doesn't decrease after withdrawals
            # Look for lines with negative amounts followed by same/higher balance
            withdrawal_balance_pairs = re.findall(
                r'[-](\d+\.?\d*)\s+.*?\$?([\d,]+\.?\d*)',
                text
            )

            # Look for specific patterns: negative amount but balance unchanged/increased
            lines = text.split('\n')
            balance_inconsistencies = 0
            prev_balance = None
            for line in lines:
                line_amounts = re.findall(r'[-+]?\$?[\d,]+\.\d{2}', line)
                if len(line_amounts) >= 2:
                    try:
                        # Last amount in line is usually the balance
                        curr_balance = float(line_amounts[-1].replace('$', '').replace(',', ''))
                        # Check for withdrawal (negative amount)
                        for am in line_amounts[:-1]:
                            val = float(am.replace('$', '').replace(',', '').replace('+', ''))
                            if am.startswith('-') and prev_balance is not None:
                                expected = prev_balance - abs(val)
                                if abs(curr_balance - expected) > 1.0 and curr_balance != prev_balance:
                                    balance_inconsistencies += 1
                        prev_balance = curr_balance
                    except (ValueError, IndexError):
                        continue

            if balance_inconsistencies >= 2:
                score = max(score, 85.0)
                reasons.append(
                    f"Balance calculation errors detected ({balance_inconsistencies} "
                    f"inconsistencies in running balance)"
                )
                red_flags += 2

    # ── 4. Gibberish / nonsensical text detection ────────────────────
    # Real bank statements have proper English in fine print.
    # Fake ones often have garbled text to fill space.
    words = lower.split()
    if len(words) >= 10:
        # Common English words that should appear in real documents
        common_words = {
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on',
            'with', 'this', 'that', 'are', 'be', 'it', 'an', 'or',
            'at', 'by', 'not', 'your', 'from', 'but', 'we', 'all',
            'have', 'has', 'will', 'may', 'can', 'any', 'no', 'if',
        }

        # Check the last 30% of words (fine print / footer area)
        footer_start = int(len(words) * 0.7)
        footer_words = words[footer_start:]

        if len(footer_words) >= 8:
            # Count words that look like gibberish
            gibberish_count = 0
            for w in footer_words:
                cleaned = re.sub(r'[^a-z]', '', w)
                if len(cleaned) < 2:
                    continue
                # Long words with unusual letter patterns
                if len(cleaned) > 12:
                    gibberish_count += 1
                # Words with too many consonant clusters
                elif re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', cleaned):
                    gibberish_count += 1
                # Very unusual letter combinations
                elif re.search(r'(qq|xx|zz|ww|hh|jj|kk)', cleaned):
                    gibberish_count += 1

            gibberish_ratio = gibberish_count / max(len(footer_words), 1)
            if gibberish_ratio > 0.25:
                score = max(score, 75.0)
                reasons.append(
                    f"Gibberish/nonsensical text detected in document footer "
                    f"({gibberish_count}/{len(footer_words)} suspicious words)"
                )
                red_flags += 1

    # ── 5. Future date detection ─────────────────────────────────────
    # Financial statements dated far in the future are suspicious
    current_year = datetime.now().year
    year_matches = re.findall(r'\b(20[2-3]\d)\b', text)
    for yr_str in year_matches:
        yr = int(yr_str)
        if yr > current_year:
            score = max(score, 65.0)
            reasons.append(f"Document contains future date (year {yr})")
            red_flags += 1
            break

    # Check for month references with future year
    future_date_patterns = re.findall(
        r'(january|february|march|april|may|june|july|august|september|'
        r'october|november|december)\s+\d{1,2},?\s+(20[2-3]\d)',
        lower
    )
    for month, yr_str in future_date_patterns:
        if int(yr_str) > current_year:
            score = max(score, 70.0)
            if f"future date (year {yr_str})" not in str(reasons):
                reasons.append(f"Statement dated in the future: {month} {yr_str}")
            red_flags += 1
            break

    # ── 6. Structural anomalies ──────────────────────────────────────
    # Duplicate dates with different transactions
    date_pattern = r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{1,2})\b'
    dates_found = re.findall(date_pattern, lower)
    if dates_found:
        date_strings = [f"{m} {d}" for m, d in dates_found]
        from collections import Counter
        date_counts = Counter(date_strings)
        duplicates = {d: c for d, c in date_counts.items() if c > 1}
        if duplicates:
            dup_info = ", ".join(f"{d} ({c}x)" for d, c in duplicates.items())
            score = max(score, 55.0)
            reasons.append(f"Duplicate transaction dates found: {dup_info}")
            red_flags += 1

    # Account summary vs transaction mismatch
    # Look for "Total Withdrawals" or "Total Deposits" in summary
    summary_total = re.search(r'total\s+(?:withdrawals?|debits?)\s*:?\s*\$?([\d,]+\.?\d*)', lower)
    if summary_total:
        try:
            claimed_total = float(summary_total.group(1).replace(',', ''))
            # Sum withdrawal amounts ONLY from transaction lines (lines with dates)
            txn_line_re = re.compile(
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{1,2}',
                re.IGNORECASE,
            )
            actual_total = 0.0
            for line in text.split('\n'):
                if txn_line_re.search(line):
                    for amt in re.findall(r'-(\d+\.?\d*)', line):
                        val = float(amt)
                        if 0.01 < val < 100000:
                            actual_total += val
            if actual_total > 0 and abs(claimed_total - actual_total) > 10:
                score = max(score, 75.0)
                reasons.append(
                    f"Account summary mismatch: claimed total withdrawals "
                    f"${claimed_total:,.2f} vs actual sum ${actual_total:,.2f}"
                )
                red_flags += 1
        except (ValueError, IndexError):
            pass

    # Missing critical fields for bank statements
    is_bank_statement = any(w in lower for w in [
        'bank', 'statement', 'account summary', 'beginning balance',
        'ending balance', 'account number',
    ])
    if is_bank_statement:
        has_routing = bool(re.search(r'routing\s*(number|#|no)', lower))
        has_swift = bool(re.search(r'swift|iban|bic', lower))
        has_phone = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))
        has_website = bool(re.search(r'www\.|\.com|\.org|\.net', lower))

        missing = []
        if not has_phone:
            missing.append("phone number")
        if not has_website:
            missing.append("website")

        if len(missing) >= 2:
            score = max(score, 50.0)
            reasons.append(
                f"Bank statement missing expected fields: {', '.join(missing)}"
            )
            red_flags += 1

    # ── Cumulative scoring: multiple independent red flags ────────────
    # If 3+ independent fraud signals are found, it's almost certainly fake
    if red_flags >= 3:
        score = max(score, 90.0)
        if "Multiple independent fraud indicators" not in str(reasons):
            reasons.append(
                f"Multiple independent fraud indicators detected ({red_flags} red flags)"
            )
    elif red_flags >= 2:
        score = max(score, 80.0)

    # ── Missing critical receipt fields (existing check) ─────────────
    if score < 50:  # Only run if nothing else triggered
        has_business_name = any(w in lower for w in [
            "inc", "llc", "ltd", "corp", "store", "shop", "restaurant", "bank"
        ])
        has_tax_id = bool(re.search(r'\b(tax|tin|ein|gst|vat)\s*[:#]?\s*\d', lower))
        has_receipt_number = bool(re.search(r'(receipt|invoice|order)\s*[#:]?\s*\d', lower))

        if not has_business_name and not has_tax_id and not has_receipt_number:
            if any(w in lower for w in ["receipt", "invoice", "balance", "total"]):
                score = max(score, 60.0)
                reasons.append(
                    "Document claims to be receipt/invoice but lacks "
                    "business name, tax ID, or receipt number"
                )

    return score, reasons


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
            text_attrs.append({"word": w, "weight": 0.0})

    if hit_count == 0:
        txt_score = 10  # Generic text, no red flags
    txt_score = min(txt_score, 100.0)

    return txt_score, text_attrs


def _analyze_image_features(pil_image) -> tuple[float, list[str]]:
    """Analyze image properties for anomaly signals.

    Returns (score, reasons).

    Design: Documents/receipts are the MOST COMMON legitimate upload.
    Only penalize for genuine visual anomalies (manipulation artifacts,
    extreme quality issues), not normal document properties.
    """
    import numpy as np

    img_array = np.array(pil_image.convert("RGB"))
    h_img, w_img = img_array.shape[:2]
    score = 15.0  # Low base score — images are innocent until proven guilty
    reasons = []

    # Text-heavy images (receipts, statements) — this is NORMAL for
    # legitimate documents. Only flag if combined with other anomalies.
    gray = np.mean(img_array, axis=2)
    edge_density = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
    is_document = edge_density > 10

    # Very low resolution (possible screenshot of screenshot, or heavily compressed)
    if h_img < 200 or w_img < 200:
        score += 15
        reasons.append("Very low resolution image (possible screenshot or compression artifact)")

    # Extremely dark or washed-out images (possible tampering)
    brightness = np.mean(img_array)
    if brightness < 50:
        score += 10
        reasons.append("Very dark image — possible manipulation to hide details")
    elif brightness > 245:
        score += 10
        reasons.append("Overexposed image — possible manipulation to hide details")

    # Check for uniform color blocks (digital editing artifacts)
    # Real photos have gradient noise; edited areas tend to be perfectly uniform
    if is_document:
        # Calculate local variance — very low variance regions suggest digital editing
        block_size = min(32, h_img // 4, w_img // 4)
        if block_size >= 8:
            # Sample patches and check variance
            patches_checked = 0
            uniform_patches = 0
            for y in range(0, h_img - block_size, block_size * 2):
                for x in range(0, w_img - block_size, block_size * 2):
                    patch = gray[y:y+block_size, x:x+block_size]
                    if np.std(patch) < 1.0:  # Nearly perfectly uniform
                        uniform_patches += 1
                    patches_checked += 1
            if patches_checked > 0 and uniform_patches / patches_checked > 0.5:
                score += 15
                reasons.append("Large uniform color regions detected (possible digital editing)")

    # Color channel anomaly — check if one channel is significantly different
    # (common in poorly edited documents)
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        channel_means = [np.mean(img_array[:, :, c]) for c in range(3)]
        channel_spread = max(channel_means) - min(channel_means)
        if channel_spread > 100:
            score += 10
            reasons.append("Unusual color channel distribution (possible filter or editing)")

    if not reasons:
        reasons.append("Image analysis complete — no visual anomalies detected")

    return min(score, 100.0), reasons


def _generate_heatmap_b64(pil_image, img_score: float = 100.0) -> str | None:
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

        # Scale intensity so low-risk images don't look "red hot"
        intensity = max(0.0, min(1.0, (img_score - 15.0) / 85.0)) # 15 is base score
        if intensity < 0.1:
             return None # Don't return a heatmap if there are no visual anomalies
             
        heatmap = heatmap * intensity

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
    heatmap_b64 = _generate_heatmap_b64(pil_image, img_score)

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

    # ── Document fraud detection on OCR text ────────────────────────
    doc_fraud_score, doc_fraud_reasons = _detect_document_fraud(extracted_text)

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

            # ── Augment model score with OCR document fraud detection ──
            # The model was trained on financial transaction patterns,
            # NOT document content analysis.  When OCR text contains
            # obvious fraud signals ("FAKE RECEIPT", placeholder amounts),
            # we boost the model's score to reflect this evidence.
            if doc_fraud_score > result["fraud_score"]:
                logger.info(
                    "OCR fraud detection boosted score: %.1f → %.1f (reasons: %s)",
                    result["fraud_score"], doc_fraud_score, doc_fraud_reasons,
                )
                result["fraud_score"] = round(doc_fraud_score, 1)
                # Update risk level
                if doc_fraud_score >= 80:
                    result["risk_level"] = "Critical"
                elif doc_fraud_score >= 60:
                    result["risk_level"] = "High"
                elif doc_fraud_score >= 35:
                    result["risk_level"] = "Medium"
                # Prepend document fraud reasons
                existing_reasons = result.get("risk_reasons", [])
                result["risk_reasons"] = doc_fraud_reasons + existing_reasons

            return JSONResponse(result)
        except Exception as e:
            logger.warning("Model image-only prediction failed, using heuristic: %s", e)

    # Apply document fraud boost to heuristic score
    if doc_fraud_score > fused_score:
        fused_score = round(doc_fraud_score, 1)
        reasons = doc_fraud_reasons + reasons
        if fused_score >= 80:
            risk_level = "CRITICAL"
        elif fused_score >= 60:
            risk_level = "HIGH"
        elif fused_score >= 35:
            risk_level = "MEDIUM"

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
        heatmap_b64 = _generate_heatmap_b64(pil_image, img_score) if pil_image else None

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
