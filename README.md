# PicoVLLM

A lightweight LLM inference engine built from scratch in Python/PyTorch. Supports batch generation, paged KV-cache with prefix deduplication, tensor parallelism, and CUDA graph optimization.

## Features

- Paged KV-cache with xxhash-based block deduplication
- CUDA graph capture/replay for decode acceleration
- Tensor parallelism via NCCL
- Flash Attention 2 integration
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

