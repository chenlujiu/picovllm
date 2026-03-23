# PicoVLLM

A lightweight LLM inference engine built from scratch in Python/PyTorch.
## Features

- Paged KV-cache with xxhash-based block deduplication
- CUDA graph capture/replay for decode acceleration
- Tensor parallelism via NCCL
- Qwen3 model support

## Installation

```bash
pip install git+https://github.com/chenlujiu/picovllm.git
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

## Quick Start

```python
from picovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B")
outputs = llm.generate(["What is a CPU?"], SamplingParams(max_tokens=128))
print(outputs[0]["text"])
```

## Benchmark

Qwen3-14B on 2× NVIDIA A100 80GB SXM. Prompts are random length (100–1024 tokens), each generating 1024 tokens.

### Batch Size Scaling (tp=2, CUDA graph on)

| Batch | Prefill tok/s | Decode tok/s | Generated tok/s | TTFT (ms) | TPOT (ms) |
|------:|--------------:|-------------:|----------------:|----------:|----------:|
| 1 | 12274 | 81 | 81 | 61.51 | 12.30 |
| 2 | 6641 | 158 | 157 | 51.35 | 6.34 |
| 4 | 22403 | 307 | 305 | 85.79 | 3.26 |
| 8 | 21880 | 570 | 562 | 220.20 | 1.75 |
| 16 | 20832 | 1070 | 1045 | 378.32 | 0.93 |
| 32 | 27117 | 2022 | 1945 | 636.14 | 0.49 |
| 64 | 24123 | 3512 | 3246 | 1006.55 | 0.28 |
| 128 | 24784 | 4798 | 4333 | 1040.47 | 0.21 |
| 256 | 25117 | 5839 | 5199 | 1018.36 | 0.17 |

### Feature Comparison (batch=256, tp=2)

| Test | Generated tok/s | Speedup |
|---|--:|--:|
| **CUDA Graph** (baseline: enforce_eager) | 5199 vs 4565 | 1.14x |
| **Tensor Parallelism** (baseline: tp=1) | 5199 vs 2351 | 2.21x |
| **Prefix Cache** (baseline: no sharing) | 5374 vs 5199 | 1.03x |

### vs vLLM (batch=256, tp=2)

| Engine | Generated tok/s | Total (s) |
|---|--:|--:|
| PicoVLLM | 5199 | 50.37 |
| vLLM | 5066 | 51.75 |

