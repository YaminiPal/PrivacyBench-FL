#!/usr/bin/env bash
# Create and populate the PrivacyBench-FL virtual environment
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.11}
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

echo "Using: $($PYTHON --version)"

rm -rf venv
"$PYTHON" -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Virtual environment ready. Activate with:"
echo "  source venv/bin/activate"
echo ""
echo "Run the full pipeline:"
echo "  python main.py"
