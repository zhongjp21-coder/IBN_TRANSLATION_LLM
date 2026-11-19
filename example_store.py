"""动态示例选择 + 持续学习（论文 3.2.1）。

做法类似 RAG：用文本嵌入模型把"意图"向量化存入 Chroma；新意图到来时，
检索 M 个"语义相近且彼此多样"的示例（用 MMR 实现，对应论文引用 [29] 的思想）。
持续学习 = 当首次翻译错误时，把 (intent, 正确输出) 追加进向量库。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config
from task_data import Item


class ExampleStore:
    def __init__(self, name: str = "examples", persist_dir: str | None = None,
                 embed_model: str = config.EMBED_MODEL):
        self._tmp = persist_dir is None
        self.persist_dir = persist_dir or tempfile.mkdtemp(prefix="chroma_")
        self.db = Chroma(
            collection_name=name,
            embedding_function=OllamaEmbeddings(model=embed_model),
            persist_directory=self.persist_dir,
        )

    @staticmethod
    def _id(intent: str) -> str:
        return hashlib.md5(intent.encode("utf-8")).hexdigest()

    def add(self, items: list[Item], batch: int = 256):
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            self.db.add_texts(
                texts=[c.intent for c in chunk],
                metadatas=[{"output": json.dumps(c.output, ensure_ascii=False)} for c in chunk],
                ids=[self._id(c.intent) for c in chunk],
            )

    def select(self, intent: str, m: int, diverse: bool = True) -> list[tuple[str, object]]:
        if m <= 0:
            return []
        try:
            if diverse:
                docs = self.db.max_marginal_relevance_search(intent, k=m, fetch_k=max(4 * m, 20))
            else:
                docs = self.db.similarity_search(intent, k=m)
        except Exception:  # 库为空等情况
            return []
        # 相关度最高的放在最后，更贴近当前意图（常见的 few-shot 排序技巧）
        pairs = [(d.page_content, json.loads(d.metadata["output"])) for d in docs]
        return pairs[::-1]

    def __len__(self):
        return self.db._collection.count()

    def close(self):
        if self._tmp:
            shutil.rmtree(self.persist_dir, ignore_errors=True)
