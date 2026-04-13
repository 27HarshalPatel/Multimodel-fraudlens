# FraudLens — MacBook Air M4 (Apple Silicon) Setup Guide

This guide is tailored specifically for running the FraudLens inference, training, and UI pipelines on an Apple Silicon M4 MacBook Air using real-world datasets.

## 1. Compatibility Analysis
I have audited the codebase for M4 / Apple Silicon compatibility:
- **PyTorch MPS Backend**: The training loop (`trainer.py`) and inference layer (`inference.py`) dynamically check for `torch.backends.mps.is_available()` and correctly assign tensors and models to the `mps` device. This ensures the M4’s Neural Engine and integrated GPU are fully utilized.
- **Mixed Precision (AMP)**: `GradScaler` is correctly disabled dynamically in `trainer.py` when running on `mps` because PyTorch's `autocast` handles mixed precision differently on Apple Silicon (and `GradScaler` is CUDA-only). 
- **Distributed Training (DDP)**: DistributedDataParallel (DDP) relies on the `nccl` backend, which is NVIDIA-only. Since an M4 Air is a single unified processor, you will bypass the `--ddp` flag entirely, running exclusively on the high-speed main path.

---

## 2. Environment Setup

The M4 architecture handles PyTorch best when installed natively without Docker to fully expose the Metal Performance Shaders (MPS).

```bash
# 1. Clean up the old virtual environment if it exists, as it was missing packages
rm -rf .venv

# 2. Re-create and activate
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies, ensuring Apple-native PyTorch is fetched
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Install the FastAPI serving components (missed in some headless setups)
pip install uvicorn fastapi python-multipart
```

---

## 3. Downloading Real-World Datasets

To train on real data instead of synthetic fallback data, you need to pull from Kaggle and GitHub directly.

### A. Set Up Kaggle Credentials
Create a Kaggle account, generate an API token (`kaggle.json`), and export your credentials in your terminal:
```bash
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
```

### B. Download Scripts
Run the following commands from the root directory to structure your real datasets:

```bash
# 1. IEEE-CIS Fraud Detection (Tabular)
# Our script handles downloading and unzipping this directly to data/tabular/
python -m src.data.download_data

# 2. PaySim Mobile Money (Tabular)
# Install Kaggle CLI and download manually to the specific folder
pip install kaggle
mkdir -p data/paysim
kaggle datasets download ealaxi/paysim1 -p data/paysim --unzip

# 3. SSBI Check Forgery Dataset (Images)
# Clone the real bank check images and copy the genuine checks to our normal directory
git clone https://github.com/saifkhichi96/ssbi-dataset.git data/ssbi
mkdir -p data/images/normal
cp -r data/ssbi/data/sources/checks/data/* data/images/normal/

# 4. Synthesize Text Descriptions & Tampered Images
# IEEE-CIS and PaySim do not provide transaction memos. SSBI provides genuine checks,
# but we need to generate tampered variants and descriptions using our script.
python -c "from src.data.generate_synthetic import generate_check_images, generate_text_descriptions; generate_check_images(n_normal=0, n_tampered=500); generate_text_descriptions()"
```

---

## 4. Run Training on M4 (MPS)

With the data in place, kick off the training loop. Make sure your MacBook is plugged in, as thermal throttling on batteries can slow down matrix multiplications.

```bash
# The script will automatically detect the "mps" device and allocate correctly.
python -m src.training.train \
    --tabular-dir data/tabular \
    --paysim-path data/paysim/paysim.csv \
    --image-dir data/images \
    --text-path data/text/descriptions.csv \
    --epochs 15 \
    --batch-size 32
```
> **Note on Epochs:** Because the full real-world dataset combining PaySim and IEEE-CIS is ~6.9 million rows, an M4 Air may take considerable time for 50 epochs. It is recommended to start with `--epochs 15`.

---

## 5. Launch the Dashboard

Once the training completes and saves `checkpoints/best_model.pt`, start the backend server:

```bash
python app.py
```
Or, start it directly with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then, open your browser and navigate to `http://localhost:8000` to interact with your MacBook-trained model!
