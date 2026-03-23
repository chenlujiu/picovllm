#!/usr/bin/env python3
"""vLLM Benchmark (for comparison with PicoVLLM)

Usage:
    python benchmark_vllm.py --model ~/huggingface/Qwen3-14B
    python benchmark_vllm.py --model ~/huggingface/Qwen3-14B --tp 2
"""

import argparse
import os
import random
from time import perf_counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

random.seed(42)


def make_prompts(tokenizer, batch_size):
    prompts = []
    for i in range(batch_size):
        target = random.randint(100, 1024)
        content = " ".join(f"word{i * 1000 + j}" for j in range(1000))
        messages = [{"role": "user", "content": content}]
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer.encode(full)[:target])
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=0.6, max_tokens=1024, ignore_eos=True)

    # Warmup
    llm.generate([tokenizer.encode("Hi")], sp)

    results = []
    for bs in args.batch_sizes:
        random.seed(42)
        prompts = make_prompts(tokenizer, bs)

        t = perf_counter()
        outputs = llm.generate(prompts, sp)
        elapsed = perf_counter() - t

        total_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_prompt = sum(len(p) for p in prompts)

        results.append({
            "batch_size": bs,
            "total_tok_s": (total_prompt + total_generated) / elapsed,
            "generated_tok_s": total_generated / elapsed,
            "avg_prompt_len": total_prompt / bs,
            "total_s": elapsed,
        })

    hdr = f"{'Batch':>6} | {'Generated tok/s':>16} | {'Total tok/s':>12} | {'Avg Prompt':>11} | {'Total (s)':>10}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        print(f"{r['batch_size']:>6} | {r['generated_tok_s']:>16.1f} | {r['total_tok_s']:>12.1f} | {r['avg_prompt_len']:>11.0f} | {r['total_s']:>10.2f}")
    print(sep)


if __name__ == "__main__":
    main()
