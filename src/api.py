import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import MockPolicy, ask

app = FastAPI(title="企业知识问答 Agent")


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel):
    question: str = Field(min_length=1)
    history: list[Message] = []
    max_steps: int = Field(default=8, ge=1, le=20)


@app.post("/chat")
def chat(request: Request):
    return ask(request.question, [m.model_dump() for m in request.history], max_steps=request.max_steps, policy=MockPolicy() if os.getenv("MOCK_MODEL") == "1" else None)
