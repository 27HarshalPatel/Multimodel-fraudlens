import os
import random
import glob
import pandas as pd
from PIL import Image, ImageDraw

def generate_synthetic_receipt(text, output_path, is_fraud=False):
    """Renders text onto a blank image background to simulate an SMS or receipt screenshot."""
    img = Image.new('RGB', (400, 400), color=(255, 255, 255) if not is_fraud else (255, 240, 240))
    d = ImageDraw.Draw(img)
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
    
    # SMS/Phishing Texts
    sms_fraud_texts = [
        "Your vehicle has an unpaid toll bill. To avoid excessive late fees, settle it promptly. Payment: https://ezdrivema.com-customeydhy.top/Update",
        "URGENT: Your bank account is suspended. Verify your identity at http://bank-verify.info",
        "Notice: Unusual login attempt from new device. Cancel this action immediately at http://cancel-logon.xyz"
    ]
    sms_legit_texts = [
        "Hey, are we still on for dinner tonight at 7?",
        "Your Uber verification code is 4829. Do not share this code."
    ]
    
    # Check Deposit Texts
    check_fraud_texts = [
        "Mobile check deposit for contractor invoice. Urgent clearance requested.",
        "Overpayment refund check deposit. Please wire back the difference.",
        "Lottery winnings check deposit via mobile app."
    ]
    check_legit_texts = [
        "Monthly rent check deposit.",
        "Salary bonus check deposit for Q3.",
        "Reimbursement for office supplies."
    ]
    
    # Attempt to locate existing check images from the SSBI dataset (if the user ran README Step 4)
    normal_checks = glob.glob("data/images/normal/*.png") + glob.glob("data/images/normal/*.jpg")
    tampered_checks = glob.glob("data/images/tampered/*.png") + glob.glob("data/images/tampered/*.jpg")
    
    records = []
    for i in range(num_samples):
        is_fraud = random.random() < 0.2 # 20% fraud rate
        amt = round(random.uniform(5.0, 1500.0), 2) if not is_fraud else round(random.uniform(100.0, 5000.0), 2)
        
        # Decide Fraud Type: 50% SMS Phishing, 50% Check Fraud
        scenario = random.choice(["sms", "check"])
        
        if scenario == "check":
            text = random.choice(check_fraud_texts) if is_fraud else random.choice(check_legit_texts)
            # Use real check images if they exist, otherwise fallback to generating synthetic image
            if is_fraud and tampered_checks:
                img_path = random.choice(tampered_checks)
            elif not is_fraud and normal_checks:
                img_path = random.choice(normal_checks)
            else:
                img_name = f"sample_{i}_check.png"
                img_path = os.path.join(output_dir, "images", img_name)
                generate_synthetic_receipt("CHECK DEPOSIT IMAGE: " + text, img_path, is_fraud=is_fraud)
        else:
            text = random.choice(sms_fraud_texts) if is_fraud else random.choice(sms_legit_texts)
            img_name = f"sample_{i}_sms.png"
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
            "image_path": img_path,
            "is_fraud": 1 if is_fraud else 0
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "fraudlens_multimodal.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Generated {num_samples} BLENDED multimodal samples at {csv_path}")
    print(f"   - Check images mapped: {len([r for r in records if 'sample' not in r['image_path']])}")
    print(f"   - Synthesized SMS/Fallbacks: {len([r for r in records if 'sample' in r['image_path']])}")
    
if __name__ == "__main__":
    print("Synthesizing Blended Multimodal FraudLens dataset...")
    synthesize_dataset(num_samples=500)
