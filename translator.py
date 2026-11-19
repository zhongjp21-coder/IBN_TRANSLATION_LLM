"""意图翻译流水线（论文 Figure 1）：

  intent -> 示例选择 -> LLM(基础/微调) -> 系统验证(失败则带错误日志重试)
         -> (人工验证) -> 部署；首次翻译错误则写回示例库(持续学习)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import ollama

import config
from example_store import ExampleStore
from tasks import Task, build_messages


@dataclass
class Result:
    intent: str
    text: str
    parsed: object
    ok: bool                  # 通过系统验证
    retries: int
    latency: float            # 从发出意图到拿到完整 JSON 的秒数（含重试）
    messages: list = field(default_factory=list)


class IntentTranslator:
    def __init__(self, model: str, task: Task, store: ExampleStore | None = None,
                 m: int = config.DEFAULT_M, max_retries: int = config.MAX_RETRIES,
                 num_ctx: int = config.NUM_CTX, host: str | None = None):
        self.model, self.task, self.store, self.m = model, task, store, m
        self.max_retries, self.num_ctx = max_retries, num_ctx
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def _chat(self, messages) -> str:
        r = self.client.chat(
            model=self.model,
            messages=messages,
            format="json",  # 约束输出为合法 JSON
            options={"temperature": config.TEMPERATURE, "num_ctx": self.num_ctx},
        )
        return r["message"]["content"]

    def translate(self, intent: str) -> Result:
        examples = self.store.select(intent, self.m) if (self.store and self.m > 0) else []
        messages = build_messages(self.task, intent, examples)
        t0 = time.perf_counter()
        retries = 0
        while True:
            text = self._chat(messages)
            ok, err, parsed = self.task.verify(text)
            if ok or retries >= self.max_retries:
                break
            # 系统反馈(error log) -> LLM 重新生成
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"System verification failed:\n{err}\nOutput the corrected JSON only."},
            ]
            retries += 1
        return Result(intent, text, parsed, ok, retries, time.perf_counter() - t0, messages)

    def revise(self, prev: Result, human_feedback: str) -> Result:
        """人工反馈（如 'the destination port should be 8080, not 8081'）-> 生成新输出，再过系统验证。"""
        messages = prev.messages + [
            {"role": "assistant", "content": prev.text},
            {"role": "user", "content": f"User feedback: {human_feedback}\nOutput the corrected JSON only."},
        ]
        t0 = time.perf_counter()
        text = self._chat(messages)
        ok, err, parsed = self.task.verify(text)
        return Result(prev.intent, text, parsed, ok, prev.retries, time.perf_counter() - t0, messages)
