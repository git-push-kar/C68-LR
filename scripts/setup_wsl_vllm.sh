#!/usr/bin/env bash
# Set up the WSL2 Ubuntu env used for Stage 1 (teacher) and Stage 3 (sampling).
#
# WSL sees the Windows C: drive at /mnt/c, so this repo is at:
#   /mnt/c/Users/Admin/Desktop/C68-LR
#
# Usage (inside WSL2 Ubuntu):
#   bash /mnt/c/Users/Admin/Desktop/C68-LR/scripts/setup_wsl_vllm.sh
set -euo pipefail

REPO="/mnt/c/Users/Admin/Desktop/C68-LR"
VENV="$HOME/vllm-env"
# Reuse the Windows HF cache on D: (optional). Comment out to use WSL's default.
export HF_HOME="${HF_HOME:-/mnt/d/hindifactguard/.cache/huggingface}"

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git build-essential

echo "==> Creating venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "==> Installing vLLM (this pulls its own torch/transformers)"
pip install "vllm>=0.6.0"

echo "==> Verifying CUDA is visible from WSL2"
python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
print("vllm", vllm.__version__)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cat <<EOF

==> Done.
Activate later with:
    source $VENV/bin/activate
    export HF_HOME=$HF_HOME
    cd $REPO

Then run Stage 1:
    python scripts/gen_teacher_data.py --set both

NOTE: keep the Windows training venv (./venv) for SFT/eval/chat. vLLM upgrades
transformers and would break InternVL's remote code path there. Only teacher +
rejection sampling run under WSL2.
EOF
