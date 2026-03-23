# PicoVLLM

A lightweight LLM inference engine built from scratch in Python/PyTorch. Supports batch generation, paged KV-cache with prefix deduplication, tensor parallelism, and CUDA graph optimization.

## Features

- Paged KV-cache with xxhash-based block deduplication
- CUDA graph capture/replay for decode acceleration
- Tensor parallelism via NCCL
- Flash Attention 2 integration
- Qwen3 model support

## Setup

```bash
bash scripts/setup.sh
```

## Quick Start

```bash
uv run python scripts/example.py
```
