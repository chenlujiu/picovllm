#!/bin/bash
set -e

MODEL=${1:-~/huggingface/Qwen3-14B}

echo "=== Benchmark 1: CUDA Graph Acceleration ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256 --enforce-eager
uv run python benchmark.py --model $MODEL --batch-sizes 256

echo "=== Benchmark 2: Batch Size Scaling ==="
uv run python benchmark.py --model $MODEL --batch-sizes 1 2 4 8 16 32 64 128 256

echo "=== Benchmark 3: Tensor Parallelism ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256 --tp 1
uv run python benchmark.py --model $MODEL --batch-sizes 256 --tp 2

echo "=== Benchmark 4: Prefix Cache Deduplication ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256
uv run python benchmark.py --model $MODEL --batch-sizes 256 --prefix-sharing

echo "=== Benchmark 5: vLLM Comparison ==="
uv run python benchmark.py --model $MODEL --batch-sizes 256
uv run python benchmark_vllm.py --model $MODEL --batch-sizes 256

echo "All benchmarks complete."
