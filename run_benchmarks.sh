#!/bin/bash
set -e

export PYTHONWARNINGS=ignore::UserWarning
MODEL=${1:-~/huggingface/Qwen3-14B}

# Baseline: batch size scaling (tp=2, cuda graph on)
echo "=== Batch Size Scaling (tp=2) ==="
uv run python benchmark.py --model $MODEL --batch-sizes 1 2 4 8 16 32 64 128 256 --tp 2

# Compare: eager mode (no cuda graph)
echo "=== CUDA Graph: enforce_eager ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256 --tp 2 --enforce-eager

# Compare: tp=1
echo "=== Tensor Parallelism: tp=1 ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256 --tp 1

# Compare: prefix sharing
echo "=== Prefix Cache: prefix sharing ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256 --tp 2 --prefix-sharing

# Compare: vLLM
echo "=== vLLM Comparison ==="
uv run python benchmark_vllm.py --model $MODEL --batch-sizes 256 --tp 2

echo "All benchmarks complete."