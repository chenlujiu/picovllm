from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str = ""          # 可忽略，picovllm 只加载一个模型
    prompt: str | list[int]  # 文本或 token ids
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None  # "stop" or "length"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]