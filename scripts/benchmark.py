import argparse
import os
from time import perf_counter

import torch
from transformers import AutoTokenizer

from picovllm import LLM, SamplingParams


def make_prompts(tokenizer, batch_size: int, prompt_text: str = "Explain what a CPU is") -> list[str]:
    """Generate a batch of identical prompts using chat template."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return [prompt] * batch_size


def benchmark(llm, prompts: list[str], sampling_params: SamplingParams) -> dict:
    """Run benchmark using low-level step() API to separate prefill/decode metrics."""
    batch_size = len(prompts)

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

        if num_tokens > 0:  # prefill
            prefill_tokens += num_tokens
            prefill_time += elapsed
            if ttft is None:
                ttft = elapsed
        else:  # decode
            decode_steps += -num_tokens
            decode_time += elapsed

    total_time = prefill_time + decode_time
    total_generated = decode_steps + batch_size  # decode steps + first tokens from prefill

    return {
        "batch_size": batch_size,
        "prefill_tok/s": prefill_tokens / prefill_time if prefill_time > 0 else 0,
        "decode_tok/s": decode_steps / decode_time if decode_time > 0 else 0,
        "ttft_ms": ttft * 1000 if ttft else 0,
        "tpot_ms": decode_time / decode_steps * 1000 if decode_steps > 0 else 0,
        "total_s": total_time,
        "tokens_generated": total_generated,
    }


def print_results(results: list[dict]):
    header = f"{'Batch':>6} | {'Prefill tok/s':>14} | {'Decode tok/s':>13} | {'TTFT (ms)':>10} | {'TPOT (ms)':>10} | {'Total (s)':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['batch_size']:>6} | "
            f"{r['prefill_tok/s']:>14.1f} | "
            f"{r['decode_tok/s']:>13.1f} | "
            f"{r['ttft_ms']:>10.1f} | "
            f"{r['tpot_ms']:>10.2f} | "
            f"{r['total_s']:>10.2f}"
        )
    print(sep)


def print_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.max_memory_allocated()
    print(f"\nGPU Memory:")
    print(f"  Total:     {total / 1024**3:.2f} GB")
    print(f"  Used:      {used / 1024**3:.2f} GB")
    print(f"  Peak:      {peak / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="PicoVLLM Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate per request")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from {model_path} ...")
    #args.enforce_eager = False
    llm = LLM(model_path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp)

    print_gpu_memory()

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens, ignore_eos=True)

    # Warmup: trigger JIT compilation and CUDA graph first-replay for all batch sizes
    warmup_sp = SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True)
    for bs in args.batch_sizes:
        llm.generate(make_prompts(tokenizer, bs), warmup_sp, use_tqdm=False)
    print("Warmup done.")

    results = []
    for bs in args.batch_sizes:
        prompts = make_prompts(tokenizer, bs)
        print(f"\nBenchmarking batch_size={bs} ...")
        r = benchmark(llm, prompts, sampling_params)
        results.append(r)
        print(f"  Prefill: {r['prefill_tok/s']:.0f} tok/s | Decode: {r['decode_tok/s']:.0f} tok/s | TTFT: {r['ttft_ms']:.1f} ms")

    print_results(results)
    print_gpu_memory()


if __name__ == "__main__":
    main()
