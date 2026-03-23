#!/bin/bash
set -e

# Install uv
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Install dependencies
uv sync

# Install flash-attn (match Python version)
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-${PY_VER}-${PY_VER}-linux_x86_64.whl"

# Download models
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
uv run hf download Qwen/Qwen3-14B --local-dir ~/huggingface/Qwen3-14B

# Verify
nvidia-smi topo -m
echo "Setup complete."