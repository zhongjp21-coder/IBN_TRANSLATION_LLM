# -*- coding: utf-8 -*-
"""
NL -> NILE 意图翻译 demo（复现 Tu et al., IJNM 2025 的实验流程，基线模型 llama3.1:8b）

论文流程与本脚本的对应关系：
  B      (base)     ：只给 输出模板 + 意图                     --mode base
  B+I    (dynamic ICL)：从示例库按 "语义相近 + 多样" 取 M 条示例   --mode icl      (论文 M=9)
  B+I+C  (continuous)：翻译错误的意图+正确答案加入示例库(在线)    --mode icl_cl
  +FT    (LoRA 微调) ：unsloth, 4bit, lr=2e-4, 3 epochs        finetune 子命令
  系统验证 (system verification)：NILE 语法检查（当前已关闭"失败重试"，只记录是否合法）

依赖：
  pip install requests numpy
  ollama pull llama3.1:8b ; ollama pull nomic-embed-text      (并保持 `ollama serve` 运行)
  微调另需：pip install unsloth trl datasets

用法（本脚本只负责"产出预测"，所有指标一律在 nile_eval.py 里算）：
  python nile_llm_demo.py check                                      # 数据检查 / 天花板估计
  python nile_llm_demo.py predict --mode base   --limit 50           # B (zero-shot)
  python nile_llm_demo.py predict --mode few_shot    --limit 50 --shots 9 # B+I # B+Few-shot（固定抽样，论文里没用到）
  python nile_llm_demo.py predict --mode icl    --limit 50 --shots 9 # B+I
  python nile_llm_demo.py predict --mode icl_cl --limit 50 --shots 9 # B+I+C（串行，结果依赖样本顺序）
  python nile_llm_demo.py predict --mode icl --workers 4 --hardware V100S-32GB   # 并发(需 OLLAMA_NUM_PARALLEL>=4)
  python nile_llm_demo.py finetune --out nile_ft                     # LoRA 微调 + 导出 GGUF
  python nile_llm_demo.py predict --model llama31-nile-ft --mode base   # 微调模型(ollama create 之后)
产出：preds_<tag>.jsonl（逐条预测，支持断点续跑）、run_<tag>.json（模型/量化/硬件/耗时元数据）
评估：python nile_eval.py metrics --preds preds_*.jsonl --train train.json
"""

import argparse
import glob
import json
import os
import pickle
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST

EMBED_HOST = os.environ.get("OLLAMA_EMBED_HOST", HOST)
if not EMBED_HOST.startswith("http"):
    EMBED_HOST = "http://" + EMBED_HOST

# ----------------------------------------------------------------------------
# 1. 提示词（已移到 prompts.py；论文 3.2：输出模板 + 示例 + 用户意图 三段）
# ----------------------------------------------------------------------------
from prompts import (
    SYSTEM_BASE,
    user_msg,
    build_system,
    prompt_fingerprint,
)  # 提示词统一放在 prompts.py


# ----------------------------------------------------------------------------
# 2. NILE 解析 / 系统验证 / 规范化比较
# ----------------------------------------------------------------------------
SCOPE = (
    r"(?:for\s+(?:endpoint|group|service|traffic)\('[^']*'\)"
    r"|from\s+endpoint\('[^']*'\)\s+to\s+endpoint\('[^']*'\))"
)
ITEM = r"[a-z]+\('[^']*'(?:\s*,\s*'[^']*')*\)"
TIME = r"start\s+(?P<tk>hour|date)\('[^']*'\)\s+end\s+(?P=tk)\('[^']*'\)"
NILE_RE = re.compile(
    rf"^define intent uniIntent:\s*(?P<scope>{SCOPE})\s+(?P<action>allow|block|set|unset|add|remove)\s+"
    rf"(?P<body>.+?)(?:\s+(?P<time>{TIME}))?\s*$"
)
ALLOWED = {
    "allow": {"service", "traffic", "protocol"},
    "block": {"service", "traffic", "protocol"},
    "set": {"bandwidth", "quota"},
    "unset": {"bandwidth", "quota"},
    "add": {"middlebox"},
    "remove": {"middlebox"},
}
BW_UNITS = {"bps", "kbps", "mbps", "gbps"}
Q_UNITS = {"mb/d", "gb/d", "mb/wk", "gb/wk", "mb/mth", "gb/mth"}


def _nc(t):
    return re.sub(r"\s*,\s*", ",", t.strip())


def parse_nile(s):
    """返回 (dict, None) 或 (None, 错误信息)。同时承担"系统验证"的语法检查。"""
    m = NILE_RE.match(s.strip())
    if not m:
        return (
            None,
            "does not match the NILE template (scope / action / items / optional time window)",
        )
    action, body = m.group("action"), m.group("body")
    items = re.findall(ITEM, body)
    if not items or _nc(body) != _nc(", ".join(items)):
        return None, "items must be a comma-separated list of name('arg') entries"
    for it in items:
        name, args = it.split("(")[0], re.findall(r"'([^']*)'", it)
        if name not in ALLOWED[action]:
            return None, f"item '{name}' is not allowed after '{action}'"
        if name == "bandwidth":
            if not (
                len(args) == 3
                and args[0] in ("max", "min")
                and args[1].isdigit()
                and args[2] in BW_UNITS
            ):
                return (
                    None,
                    "bandwidth must be bandwidth('max'|'min','<number>','<bps|kbps|mbps|gbps>')",
                )
        elif name == "quota":
            if not (
                len(args) == 3
                and args[0] in ("download", "upload")
                and args[1].isdigit()
                and args[2] in Q_UNITS
            ):
                return (
                    None,
                    "quota must be quota('download'|'upload','<number>','<mb/d|gb/d|mb/wk|gb/wk|mb/mth|gb/mth>')",
                )
        elif len(args) != 1:
            return None, f"{name}(...) takes exactly one argument"
    return {
        "scope": m.group("scope"),
        "action": action,
        "items": items,
        "time": m.group("time") or "",
    }, None


def canon(s):
    """规范形式 (scope, action, items, time)：
    - 数值去前导零（'0299' == '299'）；items 顺序无关
    - unset 的数值在自然语言里无法恢复，故忽略 unset 的数值/单位（quota 只保留周期 d/wk/mth）"""
    p, _ = parse_nile(s)
    if p is None:
        return None
    items = []
    for it in p["items"]:
        name, args = it.split("(")[0], re.findall(r"'([^']*)'", it)
        if name in ("bandwidth", "quota"):
            num, unit = str(int(args[1])), args[2]
            if p["action"] == "unset":
                num = "*"
                unit = "*" if name == "bandwidth" else "*/" + unit.split("/")[-1]
            args = [args[0], num, unit]
        items.append(name + "(" + ",".join(args) + ")")
    return (
        re.sub(r"\s+", " ", p["scope"]),
        p["action"],
        tuple(sorted(items)),
        re.sub(r"\s+", " ", p["time"]),
    )


def norm_ws(s):
    return re.sub(r"\s+", " ", s.strip())


# ----------------------------------------------------------------------------
# 3. 数据读取
# ----------------------------------------------------------------------------
def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            it for it in json.load(f) if it.get("natural_language") and it.get("nile")
        ]


# ----------------------------------------------------------------------------
# 4. Ollama 接口 + 向量化 + 示例库（论文用 langchain+Chroma；这里用 numpy 实现同样功能）
# ----------------------------------------------------------------------------
VERBOSE = False


def llm_chat(model, messages, num_ctx=4096, seed=42, stats=None):
    start = time.time()
    if VERBOSE:
        print(f"[LLM] Request started | messages={len(messages)}", flush=True)

    r = requests.post(
        f"{HOST}/api/chat",
        timeout=1800,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            # num_predict: NILE 最长约 60~70 个 token，这里放宽到 128
            # num_thread 是 CPU 线程数，必须放在 options 里才生效；全部层都在 GPU 上时影响很小
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_predict": 128,
                "num_ctx": num_ctx,
                "num_thread": 8,
            },
        },
    )
    if VERBOSE:
        print(
            f"[LLM] HTTP response received | status={r.status_code} | "
            f"elapsed={time.time() - start:.2f}s",
            flush=True,
        )
    r.raise_for_status()

    result = r.json()
    if (
        stats is not None
    ):  # 记录 token 数（few-shot 提示长、微调模型提示短，可用于成本对比）
        stats["prompt_tokens"] = result.get("prompt_eval_count")
        stats["gen_tokens"] = result.get("eval_count")
    if VERBOSE:
        print(
            f"[LLM] Completed | prompt_tokens={result.get('prompt_eval_count')} | "
            f"generated_tokens={result.get('eval_count')} | "
            f"total_duration={result.get('total_duration', 0) / 1e9:.2f}s",
            flush=True,
        )
    return result["message"]["content"]


def clean_output(text):
    text = re.sub(r"```[a-z]*", "", text)
    for line in text.splitlines():
        if "define intent" in line:
            return line[line.index("define intent") :].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else ""


class Embedder:
    def __init__(self, model, cache_path):
        self.model, self.cache_path = model, cache_path
        self.cache = (
            pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
        )

    def embed(self, texts):
        todo = [t for t in dict.fromkeys(texts) if (self.model, t) not in self.cache]
        for i in range(0, len(todo), 16):
            batch = todo[i : i + 16]
            r = requests.post(
                f"{EMBED_HOST}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=1800,
            )
            r.raise_for_status()
            for t, v in zip(batch, r.json()["embeddings"]):
                v = np.asarray(v, dtype=np.float32)
                self.cache[(self.model, t)] = v / (np.linalg.norm(v) + 1e-9)
        if todo:
            pickle.dump(self.cache, open(self.cache_path, "wb"))
        return np.stack([self.cache[(self.model, t)] for t in texts])


class ExampleDB:
    """示例库：每个示例 = (自然语言意图, 正确 NILE)。按 MMR 选"相关且多样"的 M 条（论文 3.2.1）。"""

    def __init__(self, items, embedder):
        self.items, self.emb = list(items), embedder
        self.X = embedder.embed([it["natural_language"] for it in self.items])

    def add(self, item):
        self.items.append(item)
        self.X = np.vstack([self.X, self.emb.embed([item["natural_language"]])])

    def select(self, query, m, lam=0.7):
        if m <= 0:
            return []
        sims = self.X @ self.emb.embed([query])[0]
        cand = list(np.argsort(-sims)[: max(50, m * 5)])
        chosen = []
        while len(chosen) < m and cand:
            if not chosen:
                pick = cand[0]
            else:
                sc = [
                    lam * sims[c]
                    - (1 - lam) * max(float(self.X[c] @ self.X[s]) for s in chosen)
                    for c in cand
                ]
                pick = cand[int(np.argmax(sc))]
            chosen.append(pick)
            cand.remove(pick)
        return [self.items[i] for i in reversed(chosen)]  # 最相关的放在最靠近问题的位置

    def select_static(self, m, seed=42):
        """从示例库（即 train 集）固定抽取 m 条示例，返回列表（所有查询共用同一组）。"""
        if m <= 0 or not self.items:
            return []
        k = min(m, len(self.items))
        idxs = random.Random(seed).sample(range(len(self.items)), k)
        return [self.items[i] for i in idxs]


def translate(model, nl, examples, max_retry, num_ctx, seed, stats=None):
    """一次翻译。
    当前版本：只调用一次 LLM（zero-shot / few-shot），不做"系统验证失败后带错误信息重试"。
    语法是否合法仍会记录在返回值的 ok 里，用于统计 syntax_valid_rate。"""
    system = build_system(examples)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg(nl)},
    ]

    # ---- 重试循环（已注释；需要时取消注释，并删掉下面"单次调用"三行）----
    # out, err, tries = "", None, 0
    # for tries in range(1, max_retry + 2):
    #     out = clean_output(llm_chat(model, msgs, num_ctx, seed))
    #     _, err = parse_nile(out)
    #     if err is None:
    #         break
    #     msgs += [{"role": "assistant", "content": out},
    #              {"role": "user", "content": f"System verification failed: {err}. Output the corrected NILE statement only."}]
    # return out, err is None, tries

    # ---- 单次调用 ----
    out = clean_output(llm_chat(model, msgs, num_ctx, seed, stats))
    _, err = parse_nile(out)
    return out, err is None, 1


# ----------------------------------------------------------------------------
# 5. 子命令：check
# ----------------------------------------------------------------------------
def cmd_check(a):
    sets = {"train": load(a.train), "validation": load(a.val), "test": load(a.test)}
    for k, v in sets.items():
        print(f"{k}: {len(v)} 条")
    g = {k: {it["group"] for it in v} for k, v in sets.items()}
    print(
        f"group_id 重叠 train&val={len(g['train'] & g['validation'])}  train&test={len(g['train'] & g['test'])}"
    )

    tr = sets["train"]
    bad = [it for it in tr if parse_nile(it["nile"])[0] is None]
    print(f"训练集 gold NILE 能被语法解析: {len(tr) - len(bad)}/{len(tr)}")
    for it in bad[:3]:
        print("  无法解析示例:", it["nile"], "|", parse_nile(it["nile"])[1])

    acts = Counter(
        parse_nile(it["nile"])[0]["action"] for it in tr if parse_nile(it["nile"])[0]
    )
    print("动作分布:", dict(acts))
    unset_frac = acts.get("unset", 0) / max(1, len(tr))
    print(
        f"unset 占比 {unset_frac:.1%}：其数值/单位在自然语言里无法恢复，严格 Exact Match 对这类样本不可能全对（canon 比较已忽略）"
    )

    by_nl = {}
    for it in tr:
        by_nl.setdefault(it["natural_language"].lower().strip(), set()).add(
            canon(it["nile"])
        )
    amb = sum(1 for v in by_nl.values() if len(v) > 1)
    print(
        f"同一句自然语言对应多个不同 NILE 的情形: {amb} 种（标签歧义，会拉低准确率上限）"
    )
    print(
        "训练集 group_id 最大组大小:", Counter(it["group"] for it in tr).most_common(3)
    )


# ----------------------------------------------------------------------------
# 6. 子命令：predict（B / B+I / B+I+C）—— 只产出预测，不计算任何指标
# ----------------------------------------------------------------------------
def run_tag(a):
    """输出文件标识：mode+shots、模型、数据划分、采样数、种子 —— 换任何一项都不会互相覆盖。"""
    shots = a.shots if a.mode in ("icl", "icl_cl", "few_shot") else 0
    tag = f"{a.mode}{shots}_{a.model.replace(':', '-').replace('/', '_')}_{a.split}"
    if a.limit:
        tag += f"_n{a.limit}"
    return tag + f"_s{a.seed}"


def model_info(model):
    """记录 ollama 报告的参数量/量化等级（论文里要写明量化方式）。"""
    try:
        r = requests.post(
            f"{HOST}/api/show", json={"model": model, "name": model}, timeout=60
        )
        r.raise_for_status()
        d = r.json().get("details", {})
        return {k: d.get(k) for k in ("family", "parameter_size", "quantization_level")}
    except Exception as e:
        return {"error": str(e)[:200]}


def _gold_ok(pred, gold):
    """仅供 icl_cl 决定"这条是否加入示例库"（模拟用户校验）。其余指标都在 nile_eval.py 里算。"""
    c = canon(pred)
    return c is not None and c == canon(gold)


def cmd_predict(a):
    global VERBOSE
    VERBOSE = a.verbose
    train = load(a.train)
    data = load({"train": a.train, "validation": a.val, "test": a.test}[a.split])
    if a.limit:
        data = random.Random(a.seed).sample(data, min(a.limit, len(data)))
    tag = run_tag(a)
    out_path, meta_path = f"preds_{tag}.jsonl", f"run_{tag}.json"

    done = {}  # 断点续跑：已有预测直接跳过
    if os.path.exists(out_path):
        if a.overwrite:
            os.remove(out_path)
        else:
            with open(out_path, encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        r = json.loads(l)
                        done[r["id"]] = r
            print(
                f"断点续跑：{out_path} 已有 {len(done)} 条，跳过（要重跑请加 --overwrite）"
            )
    todo = [it for it in data if it["id"] not in done]

    db = None
    static_exs = []
    if a.mode != "base":
        if a.mode == "few_shot":
            # 仅从 train 固定抽样，无需调用 embedder 向量化
            static_exs = (
                random.Random(a.seed).sample(train, min(a.shots, len(train)))
                if a.shots > 0
                else []
            )
            print(
                f"Static few-shot 已从训练集抽取 {len(static_exs)} 条固定示例 (seed={a.seed})"
            )
        else:
            db = ExampleDB(train, Embedder(a.embed_model, a.cache))
            print(f"示例库初始大小: {len(db.items)}")
            if a.mode == "icl_cl":  # 续跑时把之前加入过的样本按顺序补回，保证状态一致
                by_id = {it["id"]: it for it in data}
                for r in done.values():
                    if r.get("added_to_db") and r["id"] in by_id:
                        db.add(by_id[r["id"]])
            if todo:
                db.emb.embed(
                    [it["natural_language"] for it in todo]
                )  # 批量预热向量缓存

    n_total, t0_all, ok_n, failed = len(todo), time.time(), 0, []
    out = open(out_path, "a", encoding="utf-8")

    def run_one(it, ex):
        t0, stats = time.time(), {}
        pred, _, _ = translate(
            a.model, it["natural_language"], ex, a.max_retry, a.num_ctx, a.seed, stats
        )
        return {
            "id": it["id"],
            "nl": it["natural_language"],
            "gold": it["nile"],
            "pred": pred,
            "sec": round(time.time() - t0, 2),
            "prompt_tokens": stats.get("prompt_tokens"),
            "gen_tokens": stats.get("gen_tokens"),
        }

    def save(row):
        nonlocal ok_n
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        ok_n += 1
        if ok_n % 10 == 0 or ok_n == n_total:
            print(f"[{ok_n}/{n_total}] 已用 {time.time() - t0_all:.0f}s", flush=True)

    workers = 1 if a.mode == "icl_cl" else max(1, a.workers)
    if workers == 1:
        for it in todo:
            if a.mode == "few_shot":
                ex = static_exs
            elif db:
                ex = db.select(it["natural_language"], a.shots)
            else:
                ex = []

            try:
                row = run_one(it, ex)
            except Exception as e:
                failed.append((it["id"], str(e)[:200]))
                print(f"[FAIL] id={it['id']}: {e}", flush=True)
                continue
            if a.mode == "icl_cl":
                row["added_to_db"] = not _gold_ok(row["pred"], it["nile"])
                if row["added_to_db"]:
                    db.add(it)
            save(row)
    else:  # base / icl 的各条样本互相独立，可并发
        if a.mode == "few_shot":
            exs = [static_exs for _ in todo]
        elif db:
            exs = [db.select(it["natural_language"], a.shots) for it in todo]
        else:
            exs = [[] for _ in todo]

        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(run_one, it, ex): it for it, ex in zip(todo, exs)}
            for fu in as_completed(futs):
                try:
                    save(fu.result())
                except Exception as e:
                    failed.append((futs[fu]["id"], str(e)[:200]))
                    print(f"[FAIL] id={futs[fu]['id']}: {e}", flush=True)
    out.close()

    meta = {
        "tag": tag,
        "model": a.model,
        "model_info": model_info(a.model),
        "mode": a.mode,
        "shots": a.shots if a.mode != "base" else 0,
        "split": a.split,
        "seed": a.seed,
        "limit": a.limit,
        "num_ctx": a.num_ctx,
        "workers": workers,
        "hardware": a.hardware,
        "ollama_host": HOST,
        "prompt_sha1": prompt_fingerprint(),
        "n_requested": len(data),
        "n_done": len(done) + ok_n,
        "n_failed_this_session": len(failed),
        "wall_sec_this_session": round(time.time() - t0_all, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"\n完成 {meta['n_done']}/{len(data)} 条 -> {out_path}；运行信息 -> {meta_path}"
    )
    print(f"下一步：python nile_eval.py metrics --preds {out_path} --train {a.train}")
    if failed:
        print(
            f"有 {len(failed)} 条失败（见上面 [FAIL]）。重新运行同一条命令即可只补跑这些样本。"
        )
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# 7. 子命令：finetune（LoRA，unsloth，4bit，lr 2e-4，3 epochs —— 与论文设置一致）
# ----------------------------------------------------------------------------
def llama31_text(system, user, assistant):
    """与 ollama 的 llama3.1 模板一致（BOS 由 tokenizer 自动添加）。"""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
    )


MODELFILE = '''FROM {gguf}
TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER temperature 0
'''


def cmd_finetune(a):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    train = load(a.train)
    ds = Dataset.from_list(
        [
            {
                "text": llama31_text(
                    SYSTEM_BASE, user_msg(it["natural_language"]), it["nile"]
                )
            }
            for it in train
        ]
    )
    model, tok = FastLanguageModel.from_pretrained(
        a.hf_model,
        max_seq_length=a.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=a.seed,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=a.max_seq_len,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_steps=5,
            logging_steps=10,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=a.seed,
            output_dir=a.out + "_ckpt",
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(a.out + "_lora")
    tok.save_pretrained(a.out + "_lora")
    model.save_pretrained_gguf(a.out + "_gguf", tok, quantization_method="q4_k_m")
    ggufs = glob.glob(os.path.join(a.out + "_gguf", "*.gguf"))
    if ggufs:
        mf = os.path.join(a.out + "_gguf", "Modelfile.demo")
        open(mf, "w").write(MODELFILE.format(gguf=os.path.abspath(ggufs[0])))
        print(f"\n导入 ollama:  ollama create llama31-nile-ft -f {mf}")
        print(
            "然后评估:     python nile_llm_demo.py predict --model llama31-nile-ft --mode base"
        )
        print(
            "             python nile_llm_demo.py predict --model llama31-nile-ft --mode icl   (论文的 FT+ICL)"
        )
    else:
        print("未找到 .gguf，请检查 unsloth 的导出目录。")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd", choices=["check", "predict", "finetune", "eval"]
    )  # check=数据检查；predict=只产出预测；finetune=LoRA 微调；eval=predict 的旧名

    # ---- 数据文件 ----
    ap.add_argument(
        "--train", default="dataset/splits/train.json"
    )  # 训练集：用作 ICL 示例库、微调数据
    ap.add_argument(
        "--val", default="dataset/splits/validation.json"
    )  # 验证集：目前只在 check 和 --split validation 时用
    ap.add_argument(
        "--test", default="dataset/splits/test.json"
    )  # 测试集：eval 默认在它上面评估
    ap.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )  # eval 评估哪个数据集

    # ---- 推理设置 ----
    ap.add_argument(
        "--mode", choices=["base", "few_shot", "icl", "icl_cl"], default="base"
    )  #   base   = zero-shot，只给模板和意图，不给示例
    #   icl    = 动态 in-context learning，从训练集里挑 --shots 条相似示例放进提示
    #   icl_cl = icl + 持续学习，翻错的样本会被加入示例库，影响后面的样本
    ap.add_argument(
        "--model", default="llama3.1:8b"
    )  # ollama 里的模型名；评估微调模型时改成 llama31-nile-ft
    ap.add_argument(
        "--embed_model", default="nomic-embed-text"
    )  # 给示例检索用的向量化模型（base 模式用不到）
    ap.add_argument("--cache", default="emb_cache.pkl")  # 向量缓存文件，避免重复向量化
    ap.add_argument(
        "--shots", type=int, default=9
    )  # ICL 每次放进提示的示例条数（论文取 9）；base 模式忽略
    ap.add_argument(
        "--max_retry", type=int, default=3
    )  # 语法检查失败后的最大重试次数；重试循环已注释，目前不起作用
    ap.add_argument(
        "--num_ctx", type=int, default=4096
    )  # 上下文窗口（token 数）；zero-shot 可用 1024，ICL 要 4096
    ap.add_argument(
        "--limit", type=int, default=0
    )  # 随机抽 N 条评估，0 表示用整个数据集；用来先小规模试跑
    ap.add_argument(
        "--seed", type=int, default=42
    )  # 随机种子（抽样、生成、微调都用），保证可复现

    # ---- 运行控制（predict） ----
    ap.add_argument(
        "--workers", type=int, default=1
    )  # 并发请求数，仅 base/icl 有效；需 ollama serve 时设 OLLAMA_NUM_PARALLEL>=workers
    ap.add_argument(
        "--overwrite", action="store_true"
    )  # 默认断点续跑；加这个则删除旧预测重来
    ap.add_argument("--verbose", action="store_true")  # 打印每次 LLM 请求的细节
    ap.add_argument(
        "--hardware", default="unspecified"
    )  # 记录在 run_*.json 里的硬件说明，如 V100S-32GB / A100-80GB / CPU

    # ---- 微调设置（只对 finetune 子命令有用） ----
    ap.add_argument(
        "--out", default="nile_ft"
    )  # 输出名前缀：生成 nile_ft_lora（LoRA 权重）和 nile_ft_gguf（导出模型）
    ap.add_argument(
        "--hf_model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    )  # 微调的基座模型（HuggingFace 上的 4bit 版）
    ap.add_argument(
        "--max_seq_len", type=int, default=2048
    )  # 微调时单条样本最大 token 长度

    a = ap.parse_args()
    if a.cmd == "eval":
        print(
            "提示：eval 已更名为 predict；本脚本不再计算指标，请用 nile_eval.py metrics"
        )
        a.cmd = "predict"
    {"check": cmd_check, "predict": cmd_predict, "finetune": cmd_finetune}[a.cmd](
        a
    )  # 按子命令名调用对应函数


if __name__ == "__main__":
    main()
