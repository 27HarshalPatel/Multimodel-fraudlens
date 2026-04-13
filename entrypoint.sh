#!/bin/bash
# ──────────────────────────────────────────────────────────────
#  FraudLens — Unified entrypoint: Train → Serve
#  1. Train the model (saves best_model.pt to /app/checkpoints)
#  2. Launch the FastAPI dashboard using that model
# ──────────────────────────────────────────────────────────────
set -e

CHECKPOINT="/app/checkpoints/best_model.pt"

echo "═══════════════════════════════════════════════════════════"
echo "  FraudLens — Multimodal Fraud Detection Pipeline"
echo "═══════════════════════════════════════════════════════════"

# ── Step 1: Train ────────────────────────────────────────────
if [ "${SKIP_TRAINING}" = "1" ] && [ -f "$CHECKPOINT" ]; then
    echo "✓ SKIP_TRAINING=1 and checkpoint exists — skipping training."
else
    echo ""
    echo "▶ Step 1/2: Training the model..."
    echo "───────────────────────────────────────────────────────"
    python main.py "$@"

    if [ -f "$CHECKPOINT" ]; then
        echo ""
        echo "✓ Training complete. Model saved to $CHECKPOINT"
    else
        echo ""
        echo "✗ Training finished but no checkpoint found!"
        echo "  The dashboard will run in demo mode."
    fi
fi

# ── Step 2: Serve ────────────────────────────────────────────
echo ""
echo "▶ Step 2/2: Starting the dashboard..."
echo "  → http://0.0.0.0:8000"
echo "───────────────────────────────────────────────────────────"
exec python -m uvicorn app:app --host 0.0.0.0 --port 8000
