import time
import uuid
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from picovllm.entrypoints.protocol import (
    CompletionRequest, CompletionResponse, CompletionChoice,
    CompletionStreamResponse, CompletionStreamChoice, UsageInfo,
)
from picovllm.engine.async_engine import AsyncLLMEngine
from picovllm.sampling_params import SamplingParams

app = FastAPI()
engine: AsyncLLMEngine = None  # 在启动时初始化


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": engine.model_name, "object": "model"}]}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    if request.stream:
        return StreamingResponse(
            stream_completion(request_id, request, sampling_params),
            media_type="text/event-stream",
        )
    else:
        return await non_stream_completion(request_id, request, sampling_params)


async def stream_completion(request_id, request, sampling_params):
    """SSE streaming generator"""
    async for out in engine.generate(request.prompt, sampling_params):
        # 把新 token decode 成文本
        text = engine.tokenizer.decode([out["token_id"]])
        chunk = CompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            model=engine.model_name,
            choices=[CompletionStreamChoice(
                text=text,
                finish_reason="stop" if out["finished"] else None,
            )],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def non_stream_completion(request_id, request, sampling_params):
    """非流式：等所有 token 生成完再返回"""
    if isinstance(request.prompt, str):
        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
    else:
        prompt_tokens = len(request.prompt)  # 已经是 token ids

    all_token_ids = []
    async for out in engine.generate(request.prompt, sampling_params):
        all_token_ids.append(out["token_id"])

    text = engine.tokenizer.decode(all_token_ids)
    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=engine.model_name,
        choices=[CompletionChoice(text=text, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(all_token_ids),
            total_tokens=len(all_token_ids),
        ),
    )