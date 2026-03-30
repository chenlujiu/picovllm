import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="PicoVLLM CLI")
    subparsers = parser.add_subparsers(dest="command")

    # picovllm serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_parser.add_argument("model", type=str, help="Path to model weights")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    serve_parser.add_argument("--max-model-len", type=int, default=4096)
    serve_parser.add_argument("--enforce-eager", action="store_true")

    args = parser.parse_args()

    if args.command == "serve":
        serve(args)
    else:
        parser.print_help()


def serve(args):
    # 先初始化 engine，再启动 uvicorn
    from picovllm.entrypoints.api_server import app, engine
    from picovllm.engine.async_engine import AsyncLLMEngine

    import picovllm.entrypoints.api_server as server_module
    server_module.engine = AsyncLLMEngine(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()