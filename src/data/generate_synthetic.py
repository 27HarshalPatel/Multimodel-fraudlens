"""Generate synthetic check images and text descriptions for fraud detection.

Creates:
  - data/images/normal/   — clean check-like images
  - data/images/tampered/ — images with fraud artifacts (splicing, erasure, font mismatch)
  - data/text/descriptions.csv — merchant memos aligned to transaction IDs
"""

import csv
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

IMAGE_DIR = Path("data/images")
TEXT_DIR = Path("data/text")


# ── Text generation templates ────────────────────────────────────────────────

NORMAL_TEMPLATES = [
    "Payment for {item} at {merchant}. Order #{order_id}.",
    "Monthly subscription renewal — {merchant} ({item}).",
    "Purchase: {item}. Ref: {ref}. Merchant: {merchant}.",
    "Auto-pay for {item} service. Account ending {acct}.",
    "Online order #{order_id} from {merchant}. Item: {item}.",
    "Recurring charge for {item}. Billed by {merchant}.",
    "POS transaction at {merchant} for {item}.",
    "In-store purchase of {item} at {merchant}, receipt #{ref}.",
    "Membership fee — {merchant}. Period: monthly.",
    "Utility payment to {merchant}. Account {acct}.",
]

FRAUD_TEMPLATES = [
    "URGENT wire transfer request. Send {amount} to account {acct} immediately. Do not verify.",
    "Invoice #{order_id} — payment overdue. Wire {amount} to avoid legal action. Contact: {email}.",
    "Refund processing error. Verify identity by sending {amount} to {acct}. Expires in 1 hour.",
    "Account compromised. Transfer {amount} to secure account {acct} now. — IT Security Team",
    "Congratulations! You've won {amount}. Send processing fee to {acct} to claim your prize.",
    "Tax refund of {amount} pending. Provide bank details to {email} for direct deposit.",
    "Emergency: CEO requests immediate wire of {amount} to vendor account {acct}. Confidential.",
    "Payment reversal required. Send {amount} to {acct}. Reference: {ref}. Time-sensitive.",
    "Inheritance claim: {amount} awaiting transfer. Processing fee required to {email}.",
    "Overpayment detected. Return excess {amount} to account {acct}. Urgent.",
]

ITEMS = ["electronics", "software license", "cloud hosting", "office supplies", "consulting services",
         "insurance premium", "gym membership", "streaming service", "grocery delivery", "phone bill"]
MERCHANTS = ["TechCorp Inc", "CloudServe LLC", "RetailMart", "ServicePro", "DataFlow Systems",
             "GreenGrocer", "FitLife Gym", "StreamMax", "SupplyChain Co", "NetBill Services"]
EMAILS = ["finance@secure-verify.com", "refunds@legit-bank.net", "admin@corporate-wire.org",
          "support@prize-claim.com", "tax@gov-refund.info"]


def _random_ref() -> str:
    return f"{random.randint(100000, 999999)}"


def _random_acct() -> str:
    return f"****{random.randint(1000, 9999)}"


def generate_text_descriptions(n_total: int = 5000, fraud_rate: float = 0.035) -> None:
    """Generate text descriptions CSV aligned to transaction IDs."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    rows = []
    for tid in range(n_total):
        is_fraud = rng.random() < fraud_rate
        if is_fraud:
            template = rng.choice(FRAUD_TEMPLATES)
            text = template.format(
                amount=f"${rng.randint(500, 50000):,}",
                acct=_random_acct(),
                order_id=_random_ref(),
                email=rng.choice(EMAILS),
                ref=_random_ref(),
            )
        else:
            template = rng.choice(NORMAL_TEMPLATES)
            text = template.format(
                item=rng.choice(ITEMS),
                merchant=rng.choice(MERCHANTS),
                order_id=_random_ref(),
                ref=_random_ref(),
                acct=_random_acct(),
            )
        rows.append({"TransactionID": tid, "description": text, "is_fraud": int(is_fraud)})

    with open(TEXT_DIR / "descriptions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["TransactionID", "description", "is_fraud"])
        writer.writeheader()
        writer.writerows(rows)

    n_fraud = sum(r["is_fraud"] for r in rows)
    logger.info("Generated %d text descriptions (%d fraud, %d normal)", n_total, n_fraud, n_total - n_fraud)


# ── Image generation ─────────────────────────────────────────────────────────

def _draw_check_base(draw: ImageDraw.Draw, width: int, height: int, rng: random.Random) -> None:
    """Draw a realistic check-like base document."""
    # Background with subtle texture
    for y in range(height):
        noise = rng.randint(-3, 3)
        color = (245 + noise, 243 + noise, 238 + noise)
        draw.line([(0, y), (width, y)], fill=color)

    # Border
    draw.rectangle([10, 10, width - 10, height - 10], outline=(180, 170, 160), width=2)

    # Bank name area
    draw.rectangle([20, 20, 250, 55], fill=(220, 230, 240))
    draw.text((30, 28), "NATIONAL BANK", fill=(40, 60, 100))

    # Date line
    draw.line([(width - 200, 30), (width - 30, 30)], fill=(150, 150, 150), width=1)
    draw.text((width - 200, 15), "Date:", fill=(100, 100, 100))

    # Pay to line
    draw.text((20, 80), "Pay to the order of:", fill=(100, 100, 100))
    draw.line([(200, 95), (width - 30, 95)], fill=(150, 150, 150), width=1)

    # Amount box
    draw.rectangle([width - 150, 70, width - 20, 100], outline=(150, 150, 150), width=1)
    draw.text((width - 145, 75), "$", fill=(40, 40, 40))

    # Memo line
    draw.text((20, 140), "Memo:", fill=(100, 100, 100))
    draw.line([(80, 155), (300, 155)], fill=(150, 150, 150), width=1)

    # Signature line
    draw.line([(width - 250, 155), (width - 30, 155)], fill=(150, 150, 150), width=1)

    # MICR line
    draw.text((30, height - 40), f"⑆{rng.randint(10000, 99999)}⑆  ⑈{rng.randint(100000000, 999999999)}⑈",
              fill=(60, 60, 60))


def _add_normal_content(draw: ImageDraw.Draw, width: int, _height: int, rng: random.Random) -> None:
    """Add normal handwriting-like content to a check."""
    # Payee name
    names = ["John Smith", "ABC Corp", "Jane Doe", "City Services", "Tech Solutions"]
    draw.text((210, 78), rng.choice(names), fill=(20, 20, 80))

    # Amount
    amount = f"{rng.randint(50, 5000)}.{rng.randint(0, 99):02d}"
    draw.text((width - 135, 75), amount, fill=(20, 20, 80))

    # Date
    draw.text((width - 190, 15), f"0{rng.randint(1, 9)}/1{rng.randint(0, 9)}/2024", fill=(20, 20, 80))

    # Signature (scribble)
    sx, sy = width - 200, 145
    for _ in range(30):
        dx, dy = rng.randint(-2, 8), rng.randint(-3, 3)
        draw.line([(sx, sy), (sx + dx, sy + dy)], fill=(20, 20, 100), width=1)
        sx += dx
        sy += dy


def _add_tampered_artifacts(draw: ImageDraw.Draw, img: Image.Image, width: int, height: int,
                             rng: random.Random) -> Image.Image:
    """Add detectable fraud artifacts to a check image."""
    artifact_type = rng.choice(["splice", "erasure", "font_mismatch", "amount_alter"])
    img_array = np.array(img)

    if artifact_type == "splice":
        # Rectangular splice from a different region
        sx, sy = rng.randint(50, width - 150), rng.randint(60, 100)
        sw, sh = rng.randint(40, 100), rng.randint(15, 30)
        splice_color = (rng.randint(230, 255), rng.randint(230, 255), rng.randint(220, 240))
        draw.rectangle([sx, sy, sx + sw, sy + sh], fill=splice_color)
        draw.text((sx + 5, sy + 2), f"${rng.randint(10000, 99999)}", fill=(10, 10, 50))
        # Visible edge artifact
        draw.rectangle([sx - 1, sy - 1, sx + sw + 1, sy + sh + 1], outline=(200, 180, 170), width=1)

    elif artifact_type == "erasure":
        # White-out / erasure mark
        ex, ey = rng.randint(200, width - 200), rng.randint(70, 100)
        ew, eh = rng.randint(50, 120), rng.randint(10, 25)
        draw.rectangle([ex, ey, ex + ew, ey + eh], fill=(252, 250, 248))
        # Overwrite with new text
        draw.text((ex + 5, ey + 2), f"${rng.randint(5000, 50000)}", fill=(30, 30, 30))
        # Smudge effect
        for _ in range(20):
            px = ex + rng.randint(0, ew)
            py = ey + rng.randint(0, eh)
            c = rng.randint(240, 255)
            if 0 <= px < width and 0 <= py < height:
                img_array[py, px] = [c, c, c]

    elif artifact_type == "font_mismatch":
        # Different font style for amount (inconsistent with rest)
        draw.rectangle([width - 145, 72, width - 25, 98], fill=(248, 246, 242))
        # Noticeably different font characteristics
        draw.text((width - 135, 72), f"${rng.randint(10000, 99999)}.00",
                  fill=(0, 0, 0))
        # Bold artificial underline
        draw.line([(width - 135, 96), (width - 30, 96)], fill=(0, 0, 0), width=2)

    else:  # amount_alter
        # Digit alteration (e.g., 1 changed to 7)
        ax, ay = width - 130, 75
        draw.rectangle([ax, ay, ax + 60, ay + 20], fill=(247, 245, 240))
        altered = f"{rng.randint(7, 9)}{rng.randint(0, 9)},{rng.randint(0, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)}"
        draw.text((ax + 2, ay + 2), altered, fill=(15, 15, 60))

    return Image.fromarray(img_array)


def generate_check_images(n_normal: int = 2500, n_tampered: int = 2500) -> None:
    """Generate synthetic check images with normal and tampered variants."""
    normal_dir = IMAGE_DIR / "normal"
    tampered_dir = IMAGE_DIR / "tampered"
    normal_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)

    width, height = 480, 224

    # Normal checks
    for i in range(n_normal):
        rng = random.Random(i)
        img = Image.new("RGB", (width, height), (245, 243, 238))
        draw = ImageDraw.Draw(img)
        _draw_check_base(draw, width, height, rng)
        _add_normal_content(draw, width, height, rng)
        img.save(normal_dir / f"check_{i:05d}.png")

    # Tampered checks
    for i in range(n_tampered):
        rng = random.Random(i + 100000)
        img = Image.new("RGB", (width, height), (245, 243, 238))
        draw = ImageDraw.Draw(img)
        _draw_check_base(draw, width, height, rng)
        _add_normal_content(draw, width, height, rng)
        img = _add_tampered_artifacts(draw, img, width, height, rng)
        img.save(tampered_dir / f"check_{i:05d}.png")

    logger.info("Generated %d normal + %d tampered check images", n_normal, n_tampered)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_check_images()
    generate_text_descriptions()
    logger.info("Synthetic data generation complete.")
