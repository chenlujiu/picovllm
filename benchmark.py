#!/usr/bin/env python3
"""PicoVLLM Benchmark

Usage:
    python benchmark.py --model ~/huggingface/Qwen3-14B
    python benchmark.py --model ~/huggingface/Qwen3-14B --enforce-eager
    python benchmark.py --model ~/huggingface/Qwen3-14B --tp 2
    python benchmark.py --model ~/huggingface/Qwen3-14B --prefix-sharing
"""

import argparse
import os
import random
from time import perf_counter

from transformers import AutoTokenizer

from picovllm import LLM, SamplingParams

random.seed(42)


def make_prompts(tokenizer, batch_size, prefix_sharing=False):
    if prefix_sharing:
        system_msg = "You are a helpful, respectful and honest assistant. " * 30
    prompts = []
    for i in range(batch_size):
        target = random.randint(100, 1024)
        if prefix_sharing:
            user_msg = f"Question {i}: " + " ".join(f"topic{j}" for j in range(1000))
            messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        else:
            content = " ".join(f"word{i * 1000 + j}" for j in range(1000))
            messages = [{"role": "user", "content": content}]
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer.encode(full)[:target])
    return prompts


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
        "prefill_tok_s": prefill_tokens / prefill_time if prefill_time > 0 else 0,
        "decode_tok_s": decode_steps / decode_time if decode_time > 0 else 0,
        "ttft_ms": (ttft or 0) * 1000,
        "tpot_ms": decode_time / decode_steps * 1000 if decode_steps > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prefix-sharing", action="store_true")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp)

    # Warmup
    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True,
    )
    warmup_sp = SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True)
    for bs in args.batch_sizes:
        llm.generate([warmup_prompt] * bs, warmup_sp, use_tqdm=False)

    # Benchmark
    sp = SamplingParams(temperature=0.6, max_tokens=1024, ignore_eos=True)
    results = []
    for bs in args.batch_sizes:
        prompts = make_prompts(tokenizer, bs, args.prefix_sharing)
        results.append(benchmark(llm, prompts, sp))

    hdr = f"{'Batch':>6} | {'Prefill tok/s':>14} | {'Decode tok/s':>13} | {'TTFT (ms)':>10} | {'TPOT (ms)':>10}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        print(f"{r['batch_size']:>6} | {r['prefill_tok_s']:>14.1f} | {r['decode_tok_s']:>13.1f} | {r['ttft_ms']:>10.2f} | {r['tpot_ms']:>10.2f}")
    print(sep)


if __name__ == "__main__":
    main()
