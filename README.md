# FraudLens — Multimodal Fraud Detection via Cross-Modal Attention Fusion

> A multimodal fraud detection pipeline that fuses **computer vision (SigLIP 2)**, **NLP (DistilBERT)**, and **structured transaction analysis (MLP)** through a learned **cross-modal attention mechanism** to detect fraud that single-modality systems miss.

**Author:** Harshal Patel
**Contact:** [GitHub Profile](https://github.com/27HarshalPatel)
**Course:** Advanced Deep Learning — Spring 2026

---

## Table of Contents

- [Core Contribution](#core-contribution)
- [Architecture](#architecture)
  - [Cross-Modal Attention Fusion (Central Mechanism)](#cross-modal-attention-fusion-central-mechanism)
  - [Why Attention Fusion Over Simpler Strategies](#why-attention-fusion-over-simpler-strategies)
- [Component Maturity Levels](#component-maturity-levels)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Dataset Information](#dataset-information)
- [Training](#training)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Training the Model](#step-2-training-the-model)
  - [Step 3: Monitor Training](#step-3-monitor-training)
  - [Multi-GPU DDP Training](#multi-gpu-ddp-training)
  - [Google Colab Pro (H100 GPU) Full Pipeline](#google-colab-pro-h100-gpu-full-pipeline)
  - [Training Hyperparameters](#training-hyperparameters)
- [Running the Dashboard](#running-the-dashboard)
  - [Image Scan Mode (OCR)](#image-scan-mode-ocr)
  - [Full Analysis Mode (3-Modal)](#full-analysis-mode-3-modal)
- [Explainability Pipeline](#explainability-pipeline)
- [Experimental Design & Evaluation](#experimental-design--evaluation)
- [API Reference](#api-reference)
- [Docker Quick Start](#docker-quick-start)
- [Tech Stack](#tech-stack)
- [Limitations & Honest Assessment](#limitations--honest-assessment)
- [License](#license)

---

## Core Contribution

The central contribution of this project is the **cross-modal attention fusion mechanism** that dynamically learns which combination of modalities (tabular transaction data, check images, and text descriptions) is most informative for each individual transaction. This is distinct from simpler fusion strategies (concatenation, averaging) because:

1. **Adaptive Weighting** — The fusion layer learns a query vector that attends to modality embeddings, producing per-sample attention weights that explain *which modality matters most* for each fraud decision.
2. **Graceful Degradation** — When one or more modalities are missing (e.g., no image uploaded), the `modality_mask` mechanism ensures the fusion layer correctly ignores absent inputs rather than being biased by zero-valued placeholders.
3. **Interpretability by Design** — The attention weights are not a post-hoc explanation; they are integral to the model's decision path, providing trustworthy per-modality importance scores.

| Modality | Model | Role |
|----------|-------|------|
| **Tabular** | Custom MLP with BatchNorm | Processes 52 engineered features (amount, card type, device, V/C/D signals) |
| **Image** | SigLIP 2 (`google/siglip2-base-patch16-224`) | Detects visual anomalies in transaction-related check images |
| **Text** | DistilBERT (`distilbert-base-uncased`) | Analyses transaction descriptions and OCR-extracted text for suspicious patterns |

---

## Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Tabular    │  │    Image     │  │     Text     │
│  MLP Branch  │  │  SigLIP 2    │  │  DistilBERT  │
│  [256→128]   │  │ (8 frozen)   │  │ (4 frozen)   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └──── 128-d ──────┼──── 128-d ─────┘
                         │
              ┌──────────▼──────────┐
              │  Cross-Modal        │
              │  Attention Fusion   │
              │  (4-head, 128-d)    │
              │  + modality_mask    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  LayerNorm → MLP    │
              │  → σ(logit)         │
              │  Fraud Score 0–100% │
              └─────────────────────┘
```

### Cross-Modal Attention Fusion (Central Mechanism)

The fusion module (`src/models/fusion.py`) implements a **learned cross-attention** mechanism:

```
[e_tabular, e_image, e_text]  →  Stack: (B, 3, 128)
     ↓
Learnable query Q: (1, 1, 128) expanded to (B, 1, 128)
     ↓
MultiHeadAttention(Q, K=stack, V=stack, key_padding_mask=modality_mask)
     ↓
attn_output: (B, 1, 128) → squeeze → LayerNorm → Dropout
     ↓
Classifier: Linear(128, 64) → ReLU → Dropout → Linear(64, 1) → Sigmoid
     ↓
Output: probability (B, 1), attention_weights (B, 3)
```

**Key design choice:** The query is a *learned parameter*, not derived from any single modality. This means the fusion layer learns "what combination of modality signals best predicts fraud" rather than privileging any one input.

The `modality_mask` is a `(B, 3)` boolean tensor where `True` means "this modality is absent." It is passed directly to `nn.MultiheadAttention` as `key_padding_mask`, which zeros out the attention weight for missing modalities and re-normalizes the remaining weights. This is architecturally cleaner than alternatives like replacing missing embeddings with learned placeholders, because:

- It operates at the attention level, not the representation level
- It requires no additional parameters per modality
- The attention weights remain directly interpretable even with missing inputs

### Why Attention Fusion Over Simpler Strategies

| Strategy | Description | Weakness |
|----------|-------------|----------|
| **Concatenation** | `[e_tab ‖ e_img ‖ e_txt]` → MLP | No per-modality importance; fixed linear combination |
| **Average Pooling** | `mean(e_tab, e_img, e_txt)` | Equal weight regardless of input quality |
| **Max Pooling** | `max(e_tab, e_img, e_txt)` | Loses information from non-dominant modalities |
| **Gated Fusion** | Sigmoid gates per modality | Fixed per-sample but no inter-modality interaction |
| **Cross-Modal Attention** *(ours)* | Learned query attends to all modalities | Adaptive per-sample weighting with interpretability |

The auxiliary branch losses (`MultiModalLoss` in `src/training/losses.py`) ensure each modality branch receives independent gradient signal. The total loss is:

```
L_total = L_fused + 0.3 × (L_tabular + L_image + L_text)
```

This prevents the fusion layer from collapsing gradients into a single dominant branch.

---

## Component Maturity Levels

> **Transparency note:** Not all components of this system are at the same level of maturity. The following table explicitly distinguishes between validated claims and proof-of-concept components.

| Component | Maturity | Notes |
|-----------|----------|-------|
| **Tabular Branch (MLP)** | **Validated** | Trained on real-world IEEE-CIS + PaySim data (~6.9M transactions). Features and labels are authentic. |
| **Text Branch (DistilBERT)** | **Validated with caveats** | Trained on synthetic transaction descriptions + synthesized SMS phishing texts. The DistilBERT encoder transfers well from pre-training, but the text data itself is generated — real-world distribution may differ. |
| **Image Branch (SigLIP 2)** | **Proof of concept** | Trained on SSBI check forgery dataset (~800 real scans) supplemented with synthetic check images. The SigLIP 2 backbone has strong visual understanding from pre-training, but the fraud-specific fine-tuning data is limited. **This branch should not be treated as deployment-ready without additional real-world check image data.** |
| **Cross-Modal Attention Fusion** | **Central contribution** | The attention mechanism, modality masking, and interpretable weighting are the primary technical contributions. Validated through training metrics (AUPRC used for early stopping) and attention weight analysis. |
| **Explainability (Captum)** | **Validated** | Integrated Gradients on both text tokens and image pixels. Explanations are evaluated alongside model predictions to ensure consistency with claimed behavior (see [Explainability Pipeline](#explainability-pipeline)). |
| **OCR Integration** | **Functional** | Pytesseract (primary) + EasyOCR (fallback) extract text from uploaded images. This extracted text is fed into the DistilBERT text branch for combined scoring. |
| **Dashboard** | **Communication tool** | The UI helps demonstrate and communicate the model's behavior. It is not the primary evidence for the project's contribution. |

---

## Project Structure

```
Multimodal-fraudlens/
├── app.py                       # FastAPI server with OCR + image-only endpoint
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Compose for API + training
├── pyproject.toml               # Python project metadata & deps
├── requirements.txt             # Pip requirements
├── configs/
│   └── default.yaml             # Model & training hyperparameters
├── checkpoints/
│   └── best_model.pt            # Trained model weights
├── dashboard/
│   ├── index.html               # Tab-based UI (Image Scan / Full Analysis)
│   ├── style.css                # Premium dark glassmorphism theme
│   └── app.js                   # Tab switching, OCR rendering, gauge animation
├── src/
│   ├── config.py                # Config loader
│   ├── inference.py             # Single-sample predictor (predict + predict_image_only)
│   ├── models/
│   │   ├── fraudlens.py         # Main multimodal model (FraudLensModel)
│   │   ├── fusion.py            # ★ Cross-modal attention fusion (AttentionFusion)
│   │   ├── tabular_branch.py    # MLP branch (52 features → 128-d)
│   │   ├── image_branch.py      # SigLIP 2 vision encoder → 128-d
│   │   └── text_branch.py       # DistilBERT → 128-d
│   ├── data/
│   │   ├── dataset.py           # Unified multimodal dataset loader
│   │   ├── tabular_dataset.py   # IEEE-CIS + PaySim tabular loader
│   │   ├── multimodal_dataset.py
│   │   ├── image_dataset.py
│   │   ├── text_dataset.py
│   │   ├── generate_synthetic.py  # Synthetic image + text generators
│   │   ├── synthesizer.py       # SMS phishing + toll scam data synthesizer
│   │   └── download_data.py     # Kaggle data downloader
│   ├── training/
│   │   ├── train.py             # CLI entrypoint (single GPU / DDP)
│   │   ├── trainer.py           # Training loop (mixed precision, early stopping)
│   │   ├── losses.py            # Focal Loss + MultiModal auxiliary loss
│   │   └── metrics.py           # AUROC, AUPRC, F1, optimal threshold
│   └── explain/
│       └── captum_explainer.py  # Integrated Gradients (text + image)
├── data/
│   ├── tabular/                 # IEEE-CIS transaction CSVs
│   ├── images/                  # Check images (normal + tampered)
│   ├── text/                    # Transaction descriptions
│   ├── paysim/                  # PaySim synthetic financial data
│   └── processed/               # Blended multimodal CSVs
├── docs/                        # Architecture diagrams, blueprint
├── runs/                        # TensorBoard logs
└── tests/                       # Test suite
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **pip** or **conda**
- ~4 GB disk space (for model weights download on first run)
- **Tesseract OCR** (optional, for image text extraction): `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)

### 1. Clone the Repository

```bash
git clone https://github.com/27HarshalPatel/FraudLens.git
cd FraudLens
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install all project dependencies including dev tools:

```bash
pip install -e ".[dev]"
```

---

## Dataset Information

FraudLens uses a unified loader (`src/data/dataset.py`) that merges multiple sources into synchronized multimodal batches:

### Tabular Data (Validated)

| Source | Format | Size | Description |
|--------|--------|------|-------------|
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) | CSV | ~590K transactions | Real Vesta Corporation e-commerce data with 433 features (engineered down to 52) |
| [PaySim](https://www.kaggle.com/ealaxi/paysim1) | CSV | ~6.3M transactions | Synthetic mobile money transfers calibrated from real African operator logs |

The tabular branch processes 52 engineered features covering transaction amount, card type/category, email domain, device info, and Vesta's proprietary V/C/D signal columns. IEEE-CIS uses authentic fraud labels; PaySim labels are synthetically generated but distribution-calibrated.

### Image Data (Proof of Concept)

| Source | Format | Size | Description |
|--------|--------|------|-------------|
| [SSBI Check Forgery](https://github.com/saifkhichi96/ssbi-dataset) | PNG | ~800 real scans | Genuine and forged bank check images from academic research |
| Synthetic generator | PNG, 224×224 | ~5,000 generated | Script-generated normal/tampered checks with known tamper regions |
| SMS phishing screenshots | PNG | ~2,000 synthesized | Toll scam and phishing SMS screenshots (`src/data/synthesizer.py`) |

> **Important:** The image branch has limited real-world training data. The SSBI dataset provides authentic forged checks, but the synthetic supplements (generated via PIL scripts) do not capture the full diversity of real-world fraud imagery. **Claims about the image branch's performance should be understood as proof-of-concept**, demonstrating that the architecture can ingest and process visual data, not that it has been validated at deployment scale.

### Text Data (Synthetic)

| Source | Format | Size | Description |
|--------|--------|------|-------------|
| Transaction descriptions | CSV | ~5,000 | Synthetically generated merchant memos with fraud/normal patterns |
| SMS phishing texts | CSV | ~2,000 | Toll scam, phishing, and urgency-based text messages |

Text data is generated by `src/data/generate_synthetic.py` and `src/data/synthesizer.py`. While the DistilBERT backbone has strong linguistic pre-training, the fine-tuning data is synthetic. The OCR integration (`pytesseract` / `easyocr`) bridges a real-world gap by allowing the system to extract text from uploaded images at inference time.

---

## Training

### Step 1: Data Preparation

**Option A — Synthetic data only (no Kaggle API needed):**

```bash
python -c "
from src.data.download_data import generate_fallback
from src.data.generate_synthetic import generate_check_images, generate_text_descriptions
from pathlib import Path
generate_fallback(Path('data/tabular'), n_samples=5000)
generate_check_images(n_normal=500, n_tampered=500)
generate_text_descriptions(n_total=5000)
print('Synthetic data generated!')
"
```

**Option B — Real-world datasets (recommended for meaningful evaluation):**

```bash
# Authenticate Kaggle
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="your-kaggle-key"

# Download IEEE-CIS tabular data
python -m src.data.download_data

# Download PaySim financial simulation data
mkdir -p data/paysim
kaggle datasets download ealaxi/paysim1 -p data/paysim --unzip

# Download SSBI check forgery images
git clone https://github.com/saifkhichi96/ssbi-dataset.git data/ssbi
mkdir -p data/images/normal
cp -r data/ssbi/data/sources/checks/data/* data/images/normal/

# Generate synthetic supplements (tampered checks, text descriptions)
python -c "
from src.data.generate_synthetic import generate_check_images, generate_text_descriptions
generate_check_images(n_normal=0, n_tampered=500)
generate_text_descriptions()
"

# Generate SMS phishing/toll scam multimodal data
python src/data/synthesizer.py
```

### Step 2: Training the Model

**Local training (CPU/MPS):**

```bash
python -m src.training.train \
    --tabular-dir data/tabular \
    --image-dir data/images \
    --text-path data/text/descriptions.csv \
    --epochs 30 \
    --batch-size 16 \
    --device cpu
```

**Single GPU (CUDA):**

```bash
python -m src.training.train \
    --tabular-dir data/tabular \
    --image-dir data/images \
    --text-path data/text/descriptions.csv \
    --epochs 50 \
    --batch-size 64 \
    --device cuda
```

**With blended SMS phishing dataset:**

```bash
python -m src.training.train \
    --tabular-dir data/tabular \
    --paysim-path data/paysim/paysim.csv \
    --image-dir . \
    --text-path data/processed/fraudlens_multimodal.csv \
    --epochs 50 \
    --batch-size 64 \
    --device cuda
```

> **Note:** `--image-dir .` and `--text-path data/processed/fraudlens_multimodal.csv` map to the blended dataset (SMS + Checks) generated by `synthesizer.py`. To train only on forged checks, use `--image-dir data/images` and `--text-path data/text/descriptions.csv`.

### Step 3: Monitor Training

```bash
tensorboard --logdir runs/
```

Metrics logged to TensorBoard:
- **Loss curves:** train/val total loss, per-branch focal losses
- **Validation metrics:** AUROC, AUPRC (used for early stopping), F1, Precision, Recall
- **Learning rate:** CosineAnnealingWarmRestarts schedule
- **Attention weights:** (when logged) per-modality contribution distribution

### Multi-GPU DDP Training

```bash
torchrun --nproc_per_node=2 -m src.training.train \
    --tabular-dir data/tabular \
    --paysim-path data/paysim/paysim.csv \
    --image-dir data/images \
    --text-path data/text/descriptions.csv \
    --ddp \
    --epochs 50 \
    --batch-size 128
```

### Google Colab Pro (H100 GPU) Full Pipeline

```bash
# 1. Install
pip install -r requirements.txt && pip install -e . && pip install kaggle

# 2. Authenticate Kaggle
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-key"

# 3. Download real-world data
python -m src.data.download_data
mkdir -p data/paysim && kaggle datasets download ealaxi/paysim1 -p data/paysim --unzip
git clone https://github.com/saifkhichi96/ssbi-dataset.git data/ssbi
mkdir -p data/images/normal && cp -r data/ssbi/data/sources/checks/data/* data/images/normal/

# 4. Generate synthetic supplements + SMS phishing data
python -c "from src.data.generate_synthetic import generate_check_images, generate_text_descriptions; generate_check_images(n_normal=0, n_tampered=500); generate_text_descriptions()"
python src/data/synthesizer.py

# 5. Train (H100 80GB VRAM → batch_size=128)
python -m src.training.train \
    --tabular-dir data/tabular \
    --paysim-path data/paysim/paysim.csv \
    --image-dir . \
    --text-path data/processed/fraudlens_multimodal.csv \
    --epochs 50 --batch-size 128 --device cuda
```

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | lr=1e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingWarmRestarts | T_0=epochs/3, T_mult=2 |
| Loss | Focal Loss | α=0.75, γ=2.0 (handles ~3.5% fraud rate) |
| Auxiliary loss weight | 0.3 | Per-branch losses for independent gradients |
| Early stopping | AUPRC-based | Patience=7 epochs |
| Gradient clipping | 1.0 | Max gradient norm |
| Mixed precision | FP16 (CUDA only) | Via `torch.amp` |
| SigLIP 2 frozen layers | 8/12 | First 8 transformer blocks frozen |
| DistilBERT frozen layers | 4/6 | First 4 transformer layers frozen |
| Embedding dimension | 128 | Shared across all branches |
| Fusion attention heads | 4 | Multi-head cross-attention |
| Validation split | 20% | Stratified random split |

| CLI Flag | Description | Default |
|----------|-------------|---------|
| `--epochs` | Maximum training epochs | 50 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 1e-4 |
| `--sample-size` | Limit dataset for debugging | Full |
| `--device` | Force device (`cuda`/`mps`/`cpu`) | Auto-detect |
| `--ddp` | Enable DistributedDataParallel | False |
| `--no-amp` | Disable automatic mixed precision | False |
| `--patience` | Early stopping patience | 7 |

---

## Running the Dashboard

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

> **Demo Mode:** If no checkpoint exists at `checkpoints/best_model.pt`, the server runs with heuristic-based scoring.

The dashboard provides two analysis modes:

### Image Scan Mode (OCR)

Upload a check image, receipt, or SMS screenshot. The system will:

1. **Extract text via OCR** (pytesseract primary, easyocr fallback)
2. **Detect phishing keywords** (URLs, urgency patterns, suspicious TLDs)
3. **Run image through SigLIP 2** for visual anomaly detection
4. **Feed OCR text through DistilBERT** for linguistic fraud pattern analysis
5. **Fuse via attention** with tabular zeroed/masked out
6. **Generate anomaly heatmap** using integrated gradients
7. Display: fraud score, OCR text panel, keyword tags, attention weights, heatmap

### Full Analysis Mode (3-Modal)

Fill in transaction details (amount, card type, email domain, device) and optionally add a text description and/or check image. The system:

1. Processes tabular features through the MLP branch
2. If an image is uploaded, runs OCR and feeds extracted text + user description into DistilBERT
3. Fuses all available modalities via cross-modal attention (with missing modalities masked)
4. Displays: fraud score, per-modality attention weights, per-branch scores, risk reasons, text attributions, image heatmap

> **Note:** The dashboard is a **communication tool** that demonstrates the model's multimodal behavior. It helps visualize attention weights, OCR results, and per-branch contributions, but the primary evidence for the model's effectiveness comes from the training metrics and the ablation analysis, not from the UI.

---

## Explainability Pipeline

The Captum-based explainability pipeline (`src/explain/captum_explainer.py`) is evaluated as follows:

### What is Explained

| Method | Target | Output |
|--------|--------|--------|
| **Layer Integrated Gradients** | DistilBERT word embeddings | Per-token attribution weights (-1 to +1) |
| **Integrated Gradients** | SigLIP 2 pixel inputs | Per-pixel attribution heatmap (overlayed on original image) |
| **Attention Weights** | Fusion layer | Per-modality importance (tabular/image/text) |

### How Explanations Are Evaluated

Visualizations alone are insufficient. The explainability pipeline is evaluated through:

1. **Consistency check:** Do attention weights shift appropriately when modalities are added or removed? (e.g., removing the image should increase text attention weight.)
2. **Attribution sanity:** Are suspicious keywords (e.g., "urgent", "wire", "offshore") consistently assigned high positive attribution weights by Integrated Gradients?
3. **Heatmap relevance:** Do image heatmaps concentrate on tampered regions (for check forgery) or text-dense regions (for SMS screenshots)?
4. **Modality dominance monitoring:** The training loop tracks whether one branch dominates the fusion weights, which would indicate the attention mechanism is not learning useful cross-modal interactions.

These checks are experimental validations, not just visual demonstrations. The goal is to show that the attention fusion mechanism's explanations are **functionally consistent** with the model's predictions.

---

## Experimental Design & Evaluation

### Primary Experiment: Attention Fusion vs. Baselines

The core experimental claim is that cross-modal attention fusion outperforms simpler fusion strategies. The comparison is:

| Baseline | Implementation | What it Tests |
|----------|----------------|---------------|
| **Tabular only** | MLP branch alone | Is multimodal data helpful at all? |
| **Concatenation fusion** | `[e_tab ‖ e_img ‖ e_txt]` → Linear | Does attention add value over static combination? |
| **Average fusion** | `mean(e_tab, e_img, e_txt)` → Linear | Does learned weighting beat equal weighting? |
| **Attention fusion** *(ours)* | Cross-modal attention | Does per-sample adaptive weighting improve detection? |

Metrics: **AUPRC** (primary, handles class imbalance), AUROC, F1, Precision, Recall.

### Modality Dominance Check

A known risk in multimodal systems is one modality dominating the fusion weights. We track:

- Average attention weight per modality across the validation set
- Per-epoch attention weight distribution (should not collapse to a single modality)
- Ablation: performance with each modality individually vs. combined

### Synthetic Data Gap Analysis

Since the image and text data are partially or fully synthetic:

- We compare model behavior on synthetic vs. real inputs (where available)
- We explicitly report performance on the real SSBI check images separately from synthetic supplements
- We do not claim deployment readiness for the image branch without additional real-world data

---

## API Reference

### `POST /api/analyze-image` (Image-Only)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | Check image, receipt, or SMS screenshot |

**Response:** JSON with `fraud_score`, `risk_level`, `attention_weights`, `branch_scores`, `risk_reasons`, `ocr_extracted_text`, `ocr_keywords`, `text_attributions`, `image_explanation_base64`.

### `POST /api/predict` (Full Multimodal)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `TransactionAmt` | float | Yes | Transaction amount (USD) |
| `ProductCD` | string | Yes | Product code (W/H/C/S/R) |
| `card4` | string | Yes | Card network |
| `card6` | string | Yes | Card category |
| `P_emaildomain` | string | Yes | Purchaser email domain |
| `DeviceType` | string | Yes | Device type |
| `description` | string | No | Free-text description |
| `image` | file | No | Check image (triggers OCR text extraction) |

**Response:** JSON with `fraud_score`, `risk_level`, `attention_weights`, `branch_scores`, `risk_reasons`, `ocr_extracted_text`, `ocr_keywords`, `text_attributions`, `image_explanation_base64`.

---

## Docker Quick Start

```bash
# Build
docker build -t fraudlens:latest .

# Run inference server
docker run -d --name fraudlens-api -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  fraudlens:latest

# Run training
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/runs:/app/runs \
  fraudlens:latest \
  python -m src.training.train --epochs 50 --batch-size 32
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11 |
| **ML Framework** | PyTorch ≥ 2.0 |
| **Vision Encoder** | SigLIP 2 (HuggingFace Transformers) |
| **Text Encoder** | DistilBERT (HuggingFace Transformers) |
| **Fusion** | Cross-Modal Multi-Head Attention (4 heads, 128-d) |
| **Explainability** | Captum (Integrated Gradients + Layer IG) |
| **OCR** | pytesseract (primary) + easyocr (fallback) |
| **API Server** | FastAPI + Uvicorn |
| **Dashboard** | Vanilla HTML/CSS/JS (glassmorphic dark theme) |
| **Containerisation** | Docker + Docker Compose |
| **Loss Function** | Focal Loss (γ=2.0, α=0.75) + auxiliary branch losses |
| **Training** | Mixed precision, DDP, CosineAnnealingWarmRestarts |
| **Monitoring** | TensorBoard |

---

## Limitations & Honest Assessment

This section explicitly addresses the boundaries of what has been validated:

1. **Image Branch Data Quality:** The SigLIP 2 check image branch is trained on ~800 real scans (SSBI) plus synthetic data. It demonstrates architectural feasibility but has not been validated on a large-scale, diverse set of real-world fraud images. Performance claims for this branch are **proof of concept only**.

2. **Text Data is Synthetic:** Transaction descriptions and SMS texts are generated by scripts. While DistilBERT's pre-training provides strong language understanding, the fine-tuning distribution may not match production fraud text patterns.

3. **No Production Deployment Testing:** The system has been evaluated on held-out validation data in a research setting. Deployment considerations (latency, throughput, adversarial robustness, regulatory compliance) have not been tested.

4. **Class Imbalance:** Fraud rates of ~3.5% (IEEE-CIS) are addressed with Focal Loss and AUPRC-based early stopping, but the model's precision/recall tradeoff has not been tuned for specific business cost functions.

5. **OCR Accuracy:** Text extraction quality depends on image resolution, font clarity, and language. The pytesseract/easyocr pipeline works well for clean screenshots but may degrade on low-quality mobile photos.

6. **Modality Dominance Risk:** In training, the tabular branch may dominate if its features are more predictive. This is monitored via attention weight tracking but remains an ongoing experimental concern.

---

## License

This project is licensed under the **MIT License** — see [pyproject.toml](pyproject.toml) for details.

---

<p align="center">
  Built by <strong>Harshal Patel</strong> · Advanced Deep Learning · Spring 2026
</p>
