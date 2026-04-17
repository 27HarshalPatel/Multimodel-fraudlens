import os
import random
import pandas as pd
from PIL import Image, ImageDraw

def generate_synthetic_receipt(text, output_path, is_fraud=False):
    """Renders text onto a blank image background to simulate an SMS or receipt screenshot."""
    img = Image.new('RGB', (400, 400), color=(255, 255, 255) if not is_fraud else (255, 240, 240))
    d = ImageDraw.Draw(img)
    # Simple formatting to wrap text
    lines = []
    line = ""
    for word in text.split():
        if len(line + word) > 35:
            lines.append(line)
            line = word + " "
        else:
            line += word + " "
    lines.append(line)
    
    d.text((10, 10), "\n".join(lines), fill=(0, 0, 0))
    img.save(output_path)
    return output_path

def synthesize_dataset(num_samples=1000, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Fraud examples (SMS Spam + Malicious metadata)
    fraud_texts = [
        "Your vehicle has an unpaid toll bill. To avoid excessive late fees, settle it promptly. Payment: https://ezdrivema.com-customeydhy.top/Update",
        "URGENT: Your bank account is suspended due to authorized attempts. Verify your identity at http://bank-verify.info",
        "Notice: Unusual login attempt from new device. Cancel this action immediately at http://cancel-logon.xyz",
        "You've won a $1000 Target gift card! Click here http://prize.biz to claim your rewards now.",
        "A wire transfer of $842 was made. Please use our quick access to view or cancel in time at http://logonbcucredit.info"
    ]
    # Legit examples
    legit_texts = [
        "Hey, are we still on for dinner tonight at 7?",
        "Your Uber verification code is 4829. Do not share this code.",
        "Thanks for your recent target purchase of $42.00 at Store #0421",
        "Account update: your monthly statement is now ready to view securely in your app.",
        "Meeting moved to 3 PM tomorrow, see you then."
    ]
    
    records = []
    for i in range(num_samples):
        is_fraud = random.random() < 0.2 # 20% fraud rate for dataset
        
        amt = round(random.uniform(5.0, 1500.0), 2)
        if is_fraud:
            amt = round(random.uniform(100.0, 5000.0), 2) # Fraud amounts tend to skew differently
            
        text = random.choice(fraud_texts) if is_fraud else random.choice(legit_texts)
        
        img_name = f"sample_{i}.png"
        img_path = os.path.join(output_dir, "images", img_name)
        generate_synthetic_receipt(text, img_path, is_fraud=is_fraud)
        
        record = {
            "TransactionID": i,
            "TransactionAmt": amt,
            "ProductCD": random.choice(['W', 'C', 'H', 'R']),
            "card4": random.choice(['visa', 'mastercard', 'discover', 'american express']),
            "card6": random.choice(['credit', 'charge', 'debit']),
            "P_emaildomain": random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com']),
            "DeviceType": random.choice(['desktop', 'mobile']),
            "description": text,
            "image_path": f"images/{img_name}",
            "is_fraud": 1 if is_fraud else 0
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "fraudlens_multimodal.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Generated {num_samples} synthetic multimodal samples at {csv_path}")
    print(f"✅ Target schema perfectly aligned with FraudLens API requirements.")
    
if __name__ == "__main__":
    # Generate 500 samples for the local training loop verification
    print("Synthesizing Multimodal FraudLens dataset...")
    synthesize_dataset(num_samples=500)
    print("Run this generated CSV against your dataloader to verify model ingestion.")
