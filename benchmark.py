#!/usr/bin/env python3
"""PicoVLLM Benchmark

Usage:
    python benchmark.py --model ~/huggingface/Qwen3-14B --batch-sizes 1 4 16 64
    python benchmark.py --model ~/huggingface/Qwen3-14B --batch-sizes 1 4 16 64 --enforce-eager
    python benchmark.py --model ~/huggingface/Qwen3-14B --batch-sizes 1 8 32 128 --tp 2
    python benchmark.py --model ~/huggingface/Qwen3-14B --batch-sizes 32 --prefix-sharing
"""

import argparse
import json
import os
from time import perf_counter

import torch
from transformers import AutoTokenizer

from picovllm import LLM, SamplingParams


def make_prompts(tokenizer, batch_size, prompt_text="Explain what a CPU is"):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return [prompt] * batch_size


def make_prefix_prompts(tokenizer, batch_size):
    system_msg = "You are a helpful, respectful and honest assistant. " * 30
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Question {i}: What is {i} times 7?"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(batch_size)
    ]


def benchmark(llm, prompts, sampling_params):
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    prefill_tokens = 0
    prefill_time = 0.0
    decode_steps = 0
    decode_time = 0.0
    ttft = None

    while not llm.is_finished():
        t = perf_counter()
        outputs, num_tokens = llm.step()
        elapsed = perf_counter() - t

        if num_tokens > 0:
            prefill_tokens += num_tokens
            prefill_time += elapsed
            if ttft is None:
                ttft = elapsed
        else:
            decode_steps += -num_tokens
            decode_time += elapsed

    return {
        "batch_size": len(prompts),
        "prefill_tok_s": round(prefill_tokens / prefill_time, 1) if prefill_time > 0 else 0,
        "decode_tok_s": round(decode_steps / decode_time, 1) if decode_time > 0 else 0,
        "ttft_ms": round((ttft or 0) * 1000, 2),
        "tpot_ms": round(decode_time / decode_steps * 1000, 2) if decode_steps > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="PicoVLLM Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prefix-sharing", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(model_path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens, ignore_eos=True)

    # Warmup
    warmup_sp = SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True)
    for bs in args.batch_sizes:
        llm.generate(make_prompts(tokenizer, bs), warmup_sp, use_tqdm=False)

    results = []
    for bs in args.batch_sizes:
        if args.prefix_sharing:
            prompts = make_prefix_prompts(tokenizer, bs)
        else:
            prompts = make_prompts(tokenizer, bs)
        r = benchmark(llm, prompts, sp)
        results.append(r)

    # Print table
    hdr = f"{'Batch':>6} | {'Prefill tok/s':>14} | {'Decode tok/s':>13} | {'TTFT (ms)':>10} | {'TPOT (ms)':>10}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        print(f"{r['batch_size']:>6} | {r['prefill_tok_s']:>14.1f} | {r['decode_tok_s']:>13.1f} | {r['ttft_ms']:>10.2f} | {r['tpot_ms']:>10.2f}")
    print(sep)

    # Save JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()