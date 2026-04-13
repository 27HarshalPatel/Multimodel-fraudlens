# 🔍 FraudLens — Multimodal Fraud Detection

> A production-grade fraud detection pipeline that fuses **computer vision (SigLIP 2)**, **NLP (DistilBERT)**, and **structured transaction analysis** through learned cross-modal attention to catch fraud that single-modality systems miss.

**Author:** Harshal Patel  
**Contact:** [GitHub Profile](https://github.com/harshalanilpatel)  
**Course:** Advanced Deep Learning — Spring 2026

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
  - [Run the Setup Notebook](#run-the-setup-notebook)
  - [Run Training](#run-training)
  - [Run the Dashboard](#run-the-dashboard)
- [Dataset Information](#dataset-information)
- [Docker Quick Start](#docker-quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

FraudLens analyses transactions across three modalities simultaneously:

| Modality | Model | Role |
|----------|-------|------|
| **Tabular** | Custom MLP with BatchNorm | Processes 52 engineered features (amount, card type, device, V/C/D signals) |
| **Image** | SigLIP 2 (`google/siglip2-base-patch16-224`) | Detects visual anomalies in transaction-related check images |
| **Text** | DistilBERT (`distilbert-base-uncased`) | Analyses transaction descriptions for suspicious patterns |

A **cross-modal attention fusion** layer learns the optimal weighting between modalities for each transaction, producing a unified fraud score with per-modality explainability via Captum integrated gradients.

---

## Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Tabular    │  │    Image     │  │     Text     │
│  MLP Branch  │  │ SigLIP 2     │  │  DistilBERT  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └──── 128-d ──────┼──── 128-d ─────┘
                         │
              ┌──────────▼──────────┐
              │  Cross-Modal        │
              │  Attention Fusion   │
              │  (4-head, 128-d)    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Fraud Classifier   │
              │  Score 0–100%       │
              └─────────────────────┘
```

See `docs/architecture_diagram.png` for the full data-flow diagram.

---

## Project Structure

```
Multimodal-fraudlens/
├── app.py                  # FastAPI inference server + dashboard
├── main.py                 # Training entry point
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Compose for API + training
├── pyproject.toml          # Python project metadata & deps
├── requirements.txt        # Minimal pip requirements
├── configs/
│   └── default.yaml        # Model & training hyperparameters
├── checkpoints/
│   └── best_model.pt       # Trained model weights
├── dashboard/
│   ├── index.html          # Web dashboard UI
│   ├── style.css           # Dashboard styles
│   └── app.js              # Dashboard logic
├── notebooks/
│   └── setup.ipynb         # Environment verification & data exploration
├── src/
│   ├── config.py           # Config loader
│   ├── inference.py        # Single-sample predictor
│   ├── models/
│   │   ├── fraudlens.py    # Main multimodal model
│   │   ├── fusion.py       # Cross-modal attention fusion
│   │   ├── tabular_branch.py
│   │   ├── image_branch.py
│   │   └── text_branch.py
│   ├── data/
│   │   ├── tabular_dataset.py
│   │   ├── multimodal_dataset.py
│   │   ├── image_dataset.py
│   │   ├── text_dataset.py
│   │   ├── generate_synthetic.py
│   │   └── download_data.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── explain/
│       └── captum_explainer.py
├── data/
│   ├── tabular/            # IEEE-CIS transaction CSVs
│   ├── images/             # Check images (normal + tampered)
│   └── text/               # Transaction descriptions
├── ui/                     # Placeholder for Gradio/Streamlit interface
├── results/                # Exploratory visuals and outputs
├── docs/                   # Architecture diagrams and project visuals
│   ├── architecture_diagram.png
│   └── wireframe_mockup.png
├── runs/                   # TensorBoard logs
└── tests/                  # Test suite
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **pip** or **conda**
- ~4 GB disk space (for model weights download on first run)

### 1. Clone the Repository

```bash
git clone https://github.com/harshalanilpatel/Multimodal-fraudlens.git
cd Multimodal-fraudlens
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

### 4. Generate Synthetic Data

The project can run entirely with synthetic data (no Kaggle API needed):

```bash
python -c "
from src.data.download_data import generate_fallback
from src.data.generate_synthetic import generate_check_images, generate_text_descriptions
from pathlib import Path
generate_fallback(Path('data/tabular'), n_samples=5000)
generate_check_images(n_normal=500, n_tampered=500)
generate_text_descriptions(n_total=5000)
print('Done!')
"
```

---

## How to Run

### Run the Setup Notebook

The `notebooks/setup.ipynb` notebook verifies the environment, loads data, and produces exploration plots:

```bash
cd notebooks
jupyter notebook setup.ipynb
```

Or run it headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/setup.ipynb --output setup_executed.ipynb
```

### Run Training

We support standard training and Multi-GPU Distributed Data Parallel (DDP) training.

**Single GPU (or CPU) Training:**
```bash
python -m src.training.train --epochs 50 --batch-size 32
```

**Multi-GPU DDP Training (e.g., node with 2 GPUs):**
```bash
torchrun --nproc_per_node=2 -m src.training.train --ddp --batch-size 32
```

| Flag | Description | Default |
|------|------------|---------|
| `--epochs` | Max epochs | 50 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 1e-4 |
| `--sample-size` | Limit dataset for debugging | Full |
| `--device` | Force device (`cuda`/`mps`/`cpu`) | Auto |
| `--ddp` | Enable DistributedDataParallel | False |

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

### Run the Dashboard

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

> **Demo Mode:** If no checkpoint exists at `checkpoints/best_model.pt`, the server runs with synthetic responses.

---

## Dataset Information

FraudLens uses a unified loader (`src/data/dataset.py`) that merges three distinct sources into a synced multimodal batch generator:

| Modality | Source | Format | Size |
|----------|--------|--------|------|
| **Tabular** | [IEEE-CIS Fraud](https://www.kaggle.com/c/ieee-fraud-detection) & [PaySim](https://www.kaggle.com/ealaxi/paysim1) | CSV, 52 features after engineering | ~6.9M combined records |
| **Image** | [SSBI Check Forgery](https://github.com/saifkhichi96/ssbi-dataset) & Synthetic generator | PNG, 224×224 (resized) | ~800 real scans + 5,000 synthetic |
| **Text** | Synthetically generated transaction memos | CSV with descriptions | 5,000 descriptions |

- **Tabular data** from IEEE-CIS and PaySim can be downloaded via Kaggle API.
- **Image data** uses the genuine SSBI repository supplemented by script-generated fallback.
- **Evaluation:** See `docs/EVALUATION_PLAN.md` for our full validation schema, metrics, and ablation strategy.

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
  python main.py --epochs 50 --batch-size 32
```

---

## API Reference

### `POST /api/predict`

| Field | Type | Description |
|-------|------|-------------|
| `TransactionAmt` | float | Transaction amount (USD) |
| `ProductCD` | string | Product code (W/H/C/S/R) |
| `card4` | string | Card network |
| `card6` | string | Card category |
| `P_emaildomain` | string | Purchaser email domain |
| `DeviceType` | string | Device type |
| `description` | string | Free-text description |
| `image` | file | Optional check image |

**Response:** JSON with `fraud_score`, `risk_level`, `attention_weights`, `branch_scores`, `risk_reasons`, `text_attributions`, and `image_explanation_base64`.

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Value |
|-----------|-------|
| Tabular hidden dims | [256, 128] |
| SigLIP 2 model | `google/siglip2-base-patch16-224` |
| DistilBERT model | `distilbert-base-uncased` |
| Fusion embedding dim | 128 |
| Attention heads | 4 |
| Focal Loss (α, γ) | 0.75, 2.0 |
| Early stopping patience | 7 epochs |
| Mixed precision | Enabled |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11 |
| **ML Framework** | PyTorch ≥ 2.0 |
| **Vision Encoder** | SigLIP 2 (Hugging Face Transformers) |
| **Text Encoder** | DistilBERT (Hugging Face Transformers) |
| **Fusion** | Cross-Modal Multi-Head Attention |
| **Explainability** | Captum (Integrated Gradients) |
| **API Server** | FastAPI + Uvicorn |
| **Dashboard** | Vanilla HTML/CSS/JS |
| **Containerisation** | Docker + Docker Compose |
| **Loss Function** | Focal Loss (γ=2.0, α=0.75) |

---

## License

This project is licensed under the **MIT License** — see [pyproject.toml](pyproject.toml) for details.

---

<p align="center">
  Built with ❤️ by <strong>Harshal Patel</strong>
</p>
