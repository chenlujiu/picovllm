"""
用法：
1. 启动服务：picovllm serve ~/huggingface/Qwen3-0.6B --enforce-eager
2. 运行本脚本：python example/online_inference.py
"""

import requests


def main():
    url = "http://localhost:8000/v1/completions"

    # 非流式
    resp = requests.post(url, json={
        "prompt": "introduce yourself",
        "max_tokens": 256,
        "temperature": 0.6,
        "stream": False,
    })
    print("=== Non-streaming ===")
    print(resp.json()["choices"][0]["text"])

    # 流式
    print("\n=== Streaming ===")
    resp = requests.post(url, json={
        "prompt": "list all prime numbers within 100",
        "max_tokens": 256,
        "temperature": 0.6,
        "stream": True,
    }, stream=True)
    for line in resp.iter_lines():
        line = line.decode()
        if line.startswith("data: ") and line != "data: [DONE]":
            import json
            chunk = json.loads(line[6:])
            print(chunk["choices"][0]["text"], end="", flush=True)
    print()


if __name__ == "__main__":
    main()
