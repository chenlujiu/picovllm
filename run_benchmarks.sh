#!/bin/bash
set -e

MODEL=${1:-~/huggingface/Qwen3-14B}

echo "=== Benchmark 1: CUDA Graph Acceleration ==="
python benchmark.py --model $MODEL --batch-sizes 1 4 16 64 --enforce-eager --output results/eager.json
python benchmark.py --model $MODEL --batch-sizes 1 4 16 64 --output results/graph.json

echo "=== Benchmark 2: Batch Size Scaling ==="
python benchmark.py --model $MODEL --batch-sizes 1 2 4 8 16 32 64 128 256 --output results/batch_scaling.json

echo "=== Benchmark 3: Tensor Parallelism ==="
python benchmark.py --model $MODEL --batch-sizes 1 8 32 128 --tp 1 --output results/tp1.json
python benchmark.py --model $MODEL --batch-sizes 1 8 32 128 --tp 2 --output results/tp2.json

echo "=== Benchmark 4: Prefix Cache Deduplication ==="
python benchmark.py --model $MODEL --batch-sizes 32 --output results/no_prefix.json
python benchmark.py --model $MODEL --batch-sizes 32 --prefix-sharing --output results/prefix.json

echo "All benchmarks complete. Results in results/"
