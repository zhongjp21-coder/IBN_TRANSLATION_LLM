"""评测脚本：复现论文 Table 2/3、Figure 5/6（消融）、Figure 7-10（示例数的影响）。

模式：
  base     B      ：不给示例（M=0）
  icl      B+I    ：动态 in-context learning
  icl_cl   B+I+C  ：动态 in-context learning + 持续学习

用法示例：
  python evaluate.py --task nfv    --model qwen2:7b --mode icl_cl --m 9
  python evaluate.py --task formal --model formal_spec_ft --mode icl --m 9
  python evaluate.py --task formal --model llama3.1:8b --mode base --limit 50   # 先小样本冒烟

微调模型的"B"对应 llama3.1:8b 的 base；formal_spec_ft + base 即"仅微调"，
formal_spec_ft + icl 即"微调+in-context"。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import config
from task_data import load_task_data
from example_store import ExampleStore
from tasks import TASKS, is_correct
from translator import IntentTranslator


def run(task_name: str, model: str, mode: str, m: int, data_dir: str, limit: int | None,
        out_dir: str = "results"):
    task = TASKS[task_name]
    train, test = load_task_data(task_name, data_dir, limit)
    print(f"[{task_name}] model={model} mode={mode} M={m} train/examples={len(train)} test={len(test)}")

    store = None
    if mode in ("icl", "icl_cl"):
        store = ExampleStore(name=f"{task_name}_{int(time.time())}")
        store.add(train)
    else:
        m = 0

    tr = IntentTranslator(model, task, store=store, m=m)
    rows, correct = [], 0
    for i, item in enumerate(test, 1):
        res = tr.translate(item.intent)
        ok = res.ok and is_correct(res.parsed, item.output)
        correct += ok
        # 持续学习：首次翻译错误 -> (intent, 人工确认后的正确输出) 写回示例库
        if mode == "icl_cl" and not ok:
            store.add([item])
        rows.append({"idx": i, "correct": int(ok), "latency": res.latency,
                     "retries": res.retries, "intent": item.intent})
        if i % 10 == 0 or i == len(test):
            print(f"  {i}/{len(test)} acc={correct / i:.4f} last_latency={res.latency:.2f}s")

    lat = np.array([r["latency"] for r in rows])
    summary = {
        "task": task_name, "model": model, "mode": mode, "M": m, "n_test": len(test),
        "accuracy": round(100 * correct / len(test), 2),
        "avg_time_s": round(float(lat.mean()), 2),
        "p99_time_s": round(float(np.percentile(lat, 99)), 2),
        "final_store_size": len(store) if store else 0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    Path(out_dir).mkdir(exist_ok=True)
    tag = f"{task_name}_{model.replace(':', '-')}_{mode}_M{m}"
    with open(Path(out_dir) / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(Path(out_dir) / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if store:
        store.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["base", "icl", "icl_cl"], default="icl_cl")
    ap.add_argument("--m", type=int, default=None, help="示例数；默认 9（70B/72B 为 1）")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    m = a.m if a.m is not None else config.M_FOR_BIG_MODELS.get(a.model, config.DEFAULT_M)
    run(a.task, a.model, a.mode, m, a.data_dir, a.limit)