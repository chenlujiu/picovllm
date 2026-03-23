#!/bin/bash
set -e

MODEL=${1:-~/huggingface/Qwen3-14B}

echo "=== Benchmark 1: CUDA Graph Acceleration ==="
python benchmark.py --model $MODEL --batch-sizes 1 4 16 64 --enforce-eager
python benchmark.py --model $MODEL --batch-sizes 1 4 16 64

echo "=== Benchmark 2: Batch Size Scaling ==="
python benchmark.py --model $MODEL --batch-sizes 1 2 4 8 16 32 64 128 256

echo "=== Benchmark 3: Tensor Parallelism ==="
python benchmark.py --model $MODEL --batch-sizes 1 8 32 128 --tp 1
python benchmark.py --model $MODEL --batch-sizes 1 8 32 128 --tp 2

echo "=== Benchmark 4: Prefix Cache Deduplication ==="
python benchmark.py --model $MODEL --batch-sizes 32
python benchmark.py --model $MODEL --batch-sizes 32 --prefix-sharing

echo "All benchmarks complete."