import asyncio
import multiprocessing as mp
from typing import AsyncGenerator

from transformers import AutoTokenizer

from picovllm.engine.engine_loop import engine_loop
from picovllm.sampling_params import SamplingParams


class AsyncLLMEngine:

    def __init__(self, model: str, **kwargs):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()

        # 启动子进程
        self.process = ctx.Process(
            target=engine_loop,
            args=(model, self.input_queue, self.output_queue),
            kwargs=kwargs,
        )
        self.process.start()

        # 等待子进程 ready
        msg = self.output_queue.get()
        assert msg["type"] == "ready"

        # per-request 队列（主进程内 asyncio 用）
        self.request_queues: dict[int, asyncio.Queue] = {}
        # request_id → seq_id 映射（因为 seq_id 由子进程分配）
        self._pending_requests: dict[str, asyncio.Future] = {}

        # 启动后台 asyncio task 来收输出
        self._output_task = None  # 在第一次 generate 时启动

    async def _ensure_output_loop(self):
        """确保后台 output loop 在运行"""
        if self._output_task is None:
            self._output_task = asyncio.create_task(self._output_loop())

    async def _output_loop(self):
        """后台 task：不断从 output_queue 读消息，分发到对应请求"""
        loop = asyncio.get_event_loop()
        while True:
            # 在线程池中阻塞读 mp.Queue（不阻塞 asyncio loop）
            msg = await loop.run_in_executor(None, self.output_queue.get)

            if msg["type"] == "request_added":
                seq_id = msg["seq_id"]
                self.request_queues[seq_id] = asyncio.Queue()
                # 子进程分配了 seq_id，通知等待的 generate()
                fut = self._pending_requests.pop(msg["request_id"], None)
                if fut and not fut.done():
                    fut.set_result(msg["seq_id"])

            elif msg["type"] == "token":
                seq_id = msg["seq_id"]
                queue = self.request_queues.get(seq_id)
                if queue:
                    queue.put_nowait(msg)

            elif msg["type"] == "shutdown_done":
                break

    async def generate(self, prompt: str | list[int], sampling_params: SamplingParams) -> AsyncGenerator:
        await self._ensure_output_loop()

        request_id = str(id(prompt))
        fut = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = fut

        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt

        self.input_queue.put({
            "type": "add_request",
            "request_id": request_id,
            "prompt": token_ids,
            "sampling_params": sampling_params,
        })
        seq_id = await fut
        queue = self.request_queues[seq_id]
        try:
            while True:
                out = await queue.get()
                yield out
                if out["finished"]:
                    break
        finally:
            self.request_queues.pop(seq_id, None)

    def shutdown(self):
        self.input_queue.put({"type":"shutdown"})
        self.process.join(timeout=10)


