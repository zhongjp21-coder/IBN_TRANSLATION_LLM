"""复现论文 Table 2（--task nfv）/ Table 3（--task formal，默认）：每个模型跑 base 与 icl_cl，汇总成表。

  冒烟：python run_table2.py --models llama3.1:8b --limit 20          # formal(Table 3)
  NFV ：python run_table2.py --task nfv --models llama3.1:8b --limit 20   # Table 2
  完整：python run_table2.py --models qwen2:7b llama3.1:8b gemma2:9b gemma2:27b qwen2:72b llama3.1:70b

Accuracy / Avg time = icl_cl（动态示例 + 持续学习）；Base = 不给示例。
"""
import argparse
import json
from pathlib import Path

import config
from evaluate import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["formal", "nfv"], default="formal")
    ap.add_argument("--models", nargs="+", default=["llama3.1:8b"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()

    res = {}
    for m_name in a.models:
        m = config.M_FOR_BIG_MODELS.get(m_name, config.DEFAULT_M)
        base = run(a.task, m_name, "base", 0, a.data_dir, a.limit)
        full = run(a.task, m_name, "icl_cl", m, a.data_dir, a.limit)
        res[m_name] = (full, base)

    cols = list(res)
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    rows = [("Accuracy (%)", lambda f, b: f["accuracy"]),
            ("Base Accuracy (%)", lambda f, b: b["accuracy"]),
            ("Avg processing time (s)", lambda f, b: f["avg_time_s"]),
            ("Base avg processing time (s)", lambda f, b: b["avg_time_s"])]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(str(fn(*res[c])) for c in cols) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/table_{a.task}.md").write_text(table + "\n", encoding="utf-8")
    Path(f"results/table_{a.task}.json").write_text(
        json.dumps({k: {"icl_cl": v[0], "base": v[1]} for k, v in res.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()