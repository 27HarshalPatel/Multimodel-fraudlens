<div align="center">

# 🔎 FraudLens

**Multimodal Fraud Detection via Cross-Modal Attention Fusion**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*A multimodal fraud detection pipeline fusing **computer vision (SigLIP 2)**, **NLP (DistilBERT)**, and **structured transaction analysis (MLP)** through a learned **cross-modal attention mechanism** to detect fraud that single-modality systems miss.*

**Author:** [Harshal Patel](https://github.com/27HarshalPatel) | **Course:** Advanced Deep Learning — Spring 2026

---

### 📺 Watch FraudLens in Action

![FraudLens Demo](demo.gif)

*Analyzing tabular features, images, and text in real-time with explainable AI!*

</div>

---

## 📑 Table of Contents

- [✨ Core Contributions](#-core-contributions)
- [🏗️ Architecture](#️-architecture)
- [📈 Component Maturity Levels](#-component-maturity-levels)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [📊 Dataset Information](#-dataset-information)
- [🎯 Training Guide](#-training-guide)
- [🖥️ Running the Dashboard](#️-running-the-dashboard)
- [🔍 Explainability Pipeline](#-explainability-pipeline)
- [🧪 Experimental Design & Evaluation](#-experimental-design--evaluation)
- [📦 API Reference](#-api-reference)
- [🐳 Docker Quick Start](#-docker-quick-start)
- [🧰 Tech Stack](#-tech-stack)
- [🚧 Limitations & Honest Assessment](#-limitations--honest-assessment)
- [📄 License](#-license)

---

## ✨ Core Contributions

The central contribution of this project is the **cross-modal attention fusion mechanism** that dynamically learns which combination of modalities (tabular transaction data, check images, and text descriptions) is most informative for each individual transaction. This is distinct from simpler fusion strategies (concatenation, averaging) because:

1. 🧠 **Adaptive Weighting** — The fusion layer learns a query vector that attends to modality embeddings, producing per-sample attention weights that explain *which modality matters most* for each fraud decision.
2. 🛡️ **Graceful Degradation** — When one or more modalities are missing (e.g., no image uploaded), the `modality_mask` mechanism ensures the fusion layer correctly ignores absent inputs rather than being biased by zero-valued placeholders.
3. 🔍 **Interpretability by Design** — The attention weights are not a post-hoc explanation; they are integral to the model's decision path, providing trustworthy per-modality importance scores.

### Modality Breakdown

| Modality | Model | Role |
|----------|-------|------|
| **Tabular** | Custom MLP with BatchNorm | Processes 52 engineered features (amount, card type, device, V/C/D signals) |
| **Image** | SigLIP 2 (`google/siglip2-base-patch16-224`) | Detects visual anomalies in transaction-related check images |
| **Text** | DistilBERT (`distilbert-base-uncased`) | Analyses transaction descriptions and OCR-extracted text for suspicious patterns |

---

## 🏗️ Architecture

### System Flow Diagram

```text
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

1. `[e_tabular, e_image, e_text]` are stacked into a sequence of shape `(B, 3, 128)`.
2. A **Learnable query Q** `(1, 1, 128)` is expanded to `(B, 1, 128)`.
3. Processed via `MultiHeadAttention(Q, K=stack, V=stack, key_padding_mask=modality_mask)`.
4. Outputs `attn_output` to a Classifier block `Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid`.
5. Yields a final probability and interpretable attention weights.

**Key Design Choice:** The query is a *learned parameter*, not derived from any single modality. This means the fusion layer learns "what combination of modality signals best predicts fraud" rather than privileging any one input.

**Auxiliary Branch Losses:** The total loss includes independent gradient signals to prevent fusion layer collapse:
`L_total = L_fused + 0.3 × (L_tabular + L_image + L_text)`

---

## 📈 Component Maturity Levels

> **Transparency Note:** Not all components of this system are at the same deployment readiness scale. We distinguish between validated claims and proof-of-concept components.

| Component | Maturity | Notes |
|-----------|----------|-------|
| **Tabular Branch (MLP)** | 🟢 **Validated** | Trained on real-world IEEE-CIS + PaySim data (~6.9M transactions). Features and labels are authentic. |
| **Cross-Modal Attention** | 🟢 **Central Contrib.** | Validated through training metrics (AUPRC early stopping) and attention weight analysis. |
| **Explainability (Captum)** | 🟢 **Validated** | Built using Integrated Gradients on text/image. Explanations verify consistency with trained behavior. |
| **OCR Integration** | 🟡 **Functional** | Pytesseract + EasyOCR extract text from uploaded images, which is fed into DistilBERT perfectly. |
| **Text Branch (DistilBERT)** | 🟡 **Caveats** | DistilBERT backbone works smoothly, but fine-tuning relies on synthetic descriptions + synthesized SMS texts. |
| **Image Branch (SigLIP 2)** | 🟠 **Proof of Concept** | Trained on SSBI check forgery dataset (~800 real scans) supplemented with synthetic checks. Needs more real-world check data for production-scale validation. |
| **Dashboard** | 🔵 **Demo Tool** | Provides a premium visualization layer to communicate behavior; not primary evaluative evidence. |

---

## 📁 Project Structure

```text
Multimodal-fraudlens/
├── app.py                       # FastAPI server with OCR + image-only endpoint
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Compose for API + training
├── pyproject.toml               # Python project metadata & deps
├── requirements.txt             # Pip requirements
├── FraudLens-Demo-video.mp4     # Demo video in .mp4 format
├── demo.gif                     # Demo video in .gif format
├── configs/
│   └── default.yaml             # Model & training hyperparameters
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
├── Screenshots/                 # System screenshots for documentation
└── tests/                       # Test suite
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **pip** or **conda**
- **Tesseract OCR** (optional, for image text extraction): 
  - Mac: `brew install tesseract`
  - Linux: `apt install tesseract-ocr`

### Quick Start Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/27HarshalPatel/FraudLens.git
   cd FraudLens
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   # .venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # OR for development installs: pip install -e ".[dev]"
   ```

---

## 📊 Dataset Information

FraudLens uses a unified loader (`src/data/dataset.py`) merging diverse modal sources:

- 📑 **Tabular (Validated)**: Real-world IEEE-CIS e-commerce (~590K) & PaySim synthetic logs (~6.3M). Analyzes 52 engineered proprietary/standard features.
- 🖼️ **Image (PoC)**: SSBI Check Forgery (~800 real scans), ~5K synthetic scripted checks, and ~2K synthesized SMS phishing screenshots.
- 📝 **Text (Synthetic)**: Synthesized merchant checks and SMS phrases matching varied transaction schemes.

> **Important:** The image branch uses limited real-world volume. It excellently demonstrates architectural ingestion, but production-scale models require larger datasets for total generalizability.

---

## 🎯 Training Guide

### 1️⃣ Data Preparation
**Synthetic Generation (No Kaggle API)**
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

**Real-world Blend (Using Kaggle API)**
*(See previous details in the repository's source code for Kaggle automated download)*

### 2️⃣ Training the Model

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

**Multi-GPU DDP Training:**
```bash
torchrun --nproc_per_node=2 -m src.training.train --ddp --epochs 50 --batch-size 128 ...
```

### 3️⃣ Compute Adjustments (H100 vs T4)
- **H100 GPU (80GB VRAM):** Use `--batch-size 128`, `--lr 1e-4`
- **T4 GPU (16GB VRAM):** Safely drop to `--batch-size 16`, `--lr 5e-5` to avoid CUDA OOM constraints on SigLIP 2 embeddings.

---

## 🖥️ Running the Dashboard

Launch the elegant glassmorphic interface:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** in your browser.

### Interactive Modes
1. 📸 **Image Scan Mode (OCR)**: Upload a check receipt or SMS screenshot. The UI automatically runs OCR, detects phishing keywords, generates anomaly heatmaps out of SigLIP 2, and renders text attribution tokens.
2. 🎛️ **Full Analysis Mode (3-Modal)**: Input complete transaction features plus free text and imagery. The system dynamically fuses the modalities present, showcasing interactive pie charts of exactly what guided the model's inference!

---

## 🔍 Explainability Pipeline

FraudLens utilizes the `Captum` library to establish functional trust:
- **Language Level**: Layer Integrated Gradients calculates per-token attributions identifying flagged phrases like "urgent" or "wire".
- **Visual Level**: Integrated Gradients overlays per-pixel heatmaps directly onto the uploaded imagery ensuring attention to forged details and not generic background.
- **Fusion Weighting**: Visual representation validates whether removing components accurately redirects the model's awareness.

---

## 🧪 Experimental Design & Evaluation

Tested aggressively against simpler topological implementations:
- Tabular Only vs Concatenation vs Averaging vs **Attention Fusion (Ours)**
- Monitored continuously via AUPRC precision/recall curves. 
- Implements ablations ensuring no single modality (i.e. overpowering tabular sets) dictates the cross-modal decision.

---

## 📦 API Reference

### `POST /api/analyze-image`
- **Request:** `image` file
- **Response:** JSON detailing `fraud_score`, `risk_reasons`, `text_attributions`, `ocr_extracted_text`, and a Base64 spatial heatmap image.

### `POST /api/predict`
- **Request:** Requires transactional fields (`TransactionAmt`, `ProductCD`, `card4`, etc.). Accepts optional `description` constraint & `image`.
- **Response:** Holistic multimodal JSON mapping containing fused metrics.

---

## 🐳 Docker Quick Start

Deploy containerized testing immediately:

```bash
# Build
docker build -t fraudlens:latest .

# Run inference server
docker run -d --name fraudlens-api -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  fraudlens:latest
```

---

## 🧰 Tech Stack

- **Foundational Engine**: Python 3.11, PyTorch ≥ 2.0, FastAPI + Uvicorn
- **AI Backbone**: SigLIP 2, DistilBERT, Multi-Head Custom MLPs
- **Transformers**: HuggingFace Transformers, Captum (IGs)
- **Frontend**: Vanilla HTML/CSS/JS (Dark Glassmorphic UI)
- **DevOps**: Docker, TensorBoard 

---

## 🚧 Limitations & Honest Assessment

We want developers and researchers examining FraudLens to confidently understand where further exploration belongs:
1. **Validation Gap**: Our SigLIP2 check pipeline excels synthetically but demands diverse, larger organic datasets.
2. **Text Generation Base**: Training descriptions are fundamentally synthetic against normal pre-trained DistilBERT abilities.
3. **Imbalance Metrics**: Tuned generally but specific production deployment cost-functions require manual threshold tweaking.
4. **Adversarial Resilience**: Not thoroughly evaluated against active adversarial bypass attacks.

---

## 📄 License

This project is licensed under the **MIT License** — see [pyproject.toml](pyproject.toml) for details.

---
