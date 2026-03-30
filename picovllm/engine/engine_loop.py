import time
from multiprocessing import Queue
from picovllm.engine.llm_engine import LLMEngine
from picovllm.sampling_params import SamplingParams


def engine_loop(model: str, input_queue: Queue, output_queue: Queue, **kwargs):
    """子进程入口：初始化 engine，然后 busy loop（对应 vLLM EngineCoreProc.run_busy_loop）"""
    engine = LLMEngine(model, **kwargs)
    output_queue.put({"type": "ready", "model": model})

    while True:
        _process_input_queue(engine, input_queue, output_queue)
        _process_engine_step(engine, output_queue)


def _process_input_queue(engine, input_queue, output_queue):
    """收取请求。没活干时阻塞等，有活干时非阻塞收完。

    对应 vLLM EngineCoreProc._process_input_queue (core.py:1146)
    """
    # 没活干 → 阻塞等新请求
    while engine.is_finished():
        msg = input_queue.get()  # block=True
        _handle_msg(engine, msg, output_queue)

    # 有活干 → 非阻塞收完剩余请求
    while not input_queue.empty():
        msg = input_queue.get_nowait()
        _handle_msg(engine, msg, output_queue)


def _process_engine_step(engine, output_queue):
    """执行一步推理，把结果发给主进程。

    对应 vLLM EngineCoreProc._process_engine_step (core.py:1177)
    """
    seqs, is_prefill = engine.scheduler.schedule()
    token_ids = engine.model_runner.call("run", seqs, is_prefill)
    engine.scheduler.postprocess(seqs, token_ids)

    # 收集输出发给主进程
    if not is_prefill and token_ids:
        for seq, tid in zip(seqs, token_ids):
            output_queue.put({
                "type": "token",
                "seq_id": seq.seq_id,
                "token_id": tid,
                "finished": seq.is_finished,
            })


def _handle_msg(engine, msg, output_queue):
    """处理一条来自主进程的消息"""
    if msg["type"] == "add_request":
        seq_id = engine.add_request(msg["prompt"], msg["sampling_params"])
        output_queue.put({
            "type": "request_added",
            "seq_id": seq_id,
            "request_id": msg["request_id"],
        })
    elif msg["type"] == "shutdown":
        engine.exit()
        output_queue.put({"type": "shutdown_done"})
        raise SystemExit