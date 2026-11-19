"""数据适配层：把各数据源统一成 Item(intent, output, complexity)，供 evaluate / finetune / example_store 使用。

- formal：调用 data.py 的 load_netconfeval_formal_data()（HuggingFace NetConfEval，已过滤+0.5:0.5 划分），
          字段映射 human_language -> intent，expected -> output，batch_size -> complexity。
- nfv   ：读取 data/nfv_examples.jsonl 与 data/nfv_test.jsonl（由 gen_nfv_data.py 生成）。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class Item:
    intent: str
    output: object
    complexity: int = 1


def _to_item(row: dict) -> Item:
    expected = row["expected"]
    if isinstance(expected, str):          # HF 里可能是 JSON 字符串
        expected = json.loads(expected)
    return Item(row["human_language"], expected, int(row.get("batch_size", 1)))


def load_jsonl(path: str | Path) -> list[Item]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                items.append(Item(d["intent"], d["output"], d.get("complexity", 1)))
    return items


def load_task_data(task: str, data_dir: str = "data", limit: int | None = None):
    """返回 (example_set/train_set, test_set)。"""
    if task == "formal":
        from data import load_netconfeval_formal_data
        train_raw, test_raw = load_netconfeval_formal_data()
        train, test = [_to_item(r) for r in train_raw], [_to_item(r) for r in test_raw]
    elif task == "nfv":
        d = Path(data_dir)
        train, test = load_jsonl(d / "nfv_examples.jsonl"), load_jsonl(d / "nfv_test.jsonl")
        random.Random(config.SEED).shuffle(test)   # 同一意图的不同表述分散开，持续学习才有意义
    else:
        raise ValueError(task)
    if limit:
        test = test[:limit]
    return train, test
