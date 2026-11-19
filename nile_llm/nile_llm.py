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

用法（本脚本只负责"产出预测"，所有指标一律在 evaluation.py 里算）：
  python nile_llm.py check                                      # 数据检查 / 天花板估计
  python nile_llm.py predict --mode base   --limit 50           # B (zero-shot)
  python nile_llm.py predict --mode few_shot    --limit 50 --shots 9 # B+I # B+Few-shot（固定抽样，论文里没用到）
  python nile_llm.py predict --mode icl    --limit 50 --shots 9 # B+I
  python nile_llm.py predict --mode icl_cl --limit 50 --shots 9 # B+I+C（串行，结果依赖样本顺序）
  python nile_llm.py predict --mode icl --workers 4 --hardware V100S-32GB   # 并发(需 OLLAMA_NUM_PARALLEL>=4)
  python nile_llm.py finetune --out nile_ft                     # LoRA 微调 + 导出 GGUF
  python nile_llm.py predict --model llama31-nile-ft --mode base   # 微调模型(ollama create 之后)
产出：preds_<tag>.jsonl（逐条预测，支持断点续跑）、run_<tag>.json（模型/量化/硬件/耗时元数据）
评估：python evaluation.py metrics --preds preds_*.jsonl [--intersect]   # 然后 python evaluation.py judge --eval eval_*.jsonl --judge_models <模型>
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
# 2. NILE 解析 / 系统验证 / 规范化比较（统一来自 utils.py 与 evaluation.py，避免两份定义不同步）
# ----------------------------------------------------------------------------
from utils import parse_nile, norm_ws  # noqa: E402,F401  语法检查
from evaluation import canon, compile_nile_rules  # noqa: E402,F401  L2 规范形式 / L3 规则集合


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
    ):  # 记录 token 数；注意 prompt_eval_count 不含被 KV cache 复用的前缀，仅供参考
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
    """仅供 icl_cl 决定"这条是否加入示例库"（模拟用户校验）。
    判据 = evaluation.py 里 L3 的 equiv（规则集合完全相等：含单位换算/大小写/时间补零），
    这样"哪些样本被当作错误并学习"与最终报告的主指标口径一致。"""
    pr, _, _ = compile_nile_rules(pred)
    gr, _, _ = compile_nile_rules(gold)
    return pr is not None and gr is not None and pr == gr


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
        # prompt_tokens 来自 ollama 的 prompt_eval_count，命中 KV cache 的前缀不计入，会低估（base 尤其明显）；
        # prompt_chars 是不受缓存影响的提示长度，做"提示长度/成本"对比请用它（或另用 tokenizer 数）
        prompt_chars = len(build_system(ex)) + len(user_msg(it["natural_language"]))
        return {
            "id": it["id"],
            "nl": it["natural_language"],
            "gold": it["nile"],
            "pred": pred,
            "sec": round(time.time() - t0, 2),
            "prompt_chars": prompt_chars,
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
    print(f"下一步：python evaluation.py metrics --preds {out_path}")
    if failed:
        print(
            f"有 {len(failed)} 条失败（见上面 [FAIL]）。重新运行同一条命令即可只补跑这些样本。"
        )
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# 7. 子命令：finetune —— 对齐原论文 finetuning.py
#    保持一致：unsloth / 4bit / LoRA r=16 alpha=16 / lr 2e-4 / 3 epochs / 有效 batch 8 / adamw_8bit /
#              linear 调度 / warmup 5 / packing=False / 整段文本算 loss / 导出 q4_k_m GGUF
#    相对论文的改动：
#      ① 训练文本里的 SYSTEM 用本项目的 SYSTEM_BASE，用户输入用 user_msg（与推理完全一致）
#      ② 手动拼文本并显式加 EOS，不依赖 unsloth.apply_chat_template 的内部行为
#      ③ 不用 to_sharegpt(conversation_extension=3)：它会把 3 条无关样本拼成一段多轮对话，而推理是单轮
#      ④ A100：bf16 自动启用；batch 8 x accum 1（有效 batch 与论文 2 x 4 相同，但更快）
# ----------------------------------------------------------------------------
FT_STYLES = {
    # paper ：论文做法 —— 基座模型(非 Instruct) + 自定义 "{SYSTEM}\nUSER: {INPUT}\nASSISTANT: {OUTPUT}" 模板
    "paper": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    # llama3：Instruct 模型 + 官方 llama3 对话模板（与 ollama 的 llama3.1:8b 同一族，做对照用）
    "llama3": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
}


def ft_text(style, system, user, assistant, eos):
    """一条训练文本。BOS 由 tokenizer 自动添加。"""
    if style == "paper":
        return f"{system}\nUSER: {user}\nASSISTANT: {assistant}{eos}"
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
    )


# Ollama 的 TEMPLATE 必须与训练文本逐字符一致。
# paper 风格：提示在 "ASSISTANT:" 处结束（不带空格），模型生成的第一个 token 是 " define"，
# 与训练时 "ASSISTANT: define" 的分词一致；clean_output 会截取 "define intent" 之后的内容。
MODELFILE_PAPER = '''FROM {gguf}
TEMPLATE """{{{{ if .System }}}}{{{{ .System }}}}
{{{{ end }}}}USER: {{{{ .Prompt }}}}
ASSISTANT:{{{{ .Response }}}}"""
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "USER:"
PARAMETER temperature 0
'''

MODELFILE_LLAMA3 = '''FROM {gguf}
TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER temperature 0
'''
MODELFILES = {"paper": MODELFILE_PAPER, "llama3": MODELFILE_LLAMA3}


def cmd_finetune(a):
    from unsloth import (
        FastLanguageModel,
        is_bfloat16_supported,
    )  # unsloth 必须先于 transformers/trl 导入
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    style = a.ft_style
    hf_model = a.hf_model or FT_STYLES[style]
    bf16 = is_bfloat16_supported()  # A100: True
    train = load(a.train)
    print(
        f"微调配置：style={style}  基座={hf_model}  训练样本={len(train)}  bf16={bf16}  "
        f"batch={a.ft_batch}x{a.ft_accum}(有效 {a.ft_batch * a.ft_accum})  epochs={a.ft_epochs}  lr={a.ft_lr}"
    )

    model, tok = FastLanguageModel.from_pretrained(
        hf_model, max_seq_length=a.max_seq_len, dtype=None, load_in_4bit=True
    )  # dtype=None：自动（A100 用 bf16）
    eos = tok.eos_token
    texts = [
        ft_text(style, SYSTEM_BASE, user_msg(it["natural_language"]), it["nile"], eos)
        for it in train
    ]

    lens = [len(tok(t)["input_ids"]) for t in texts]  # 检查有没有被 max_seq_len 截断
    n_trunc = sum(l > a.max_seq_len for l in lens)
    print(
        f"训练文本 token 长度：平均 {sum(lens) / len(lens):.0f}，最大 {max(lens)}；超过 max_seq_len={a.max_seq_len} 的有 {n_trunc} 条"
    )
    if n_trunc:
        print("  [警告] 这些样本的结尾（含答案）会被截断，请调大 --max_seq_len")
    print(
        "---- 第 1 条训练文本（请核对格式）----\n"
        + texts[0]
        + "\n--------------------------------------"
    )
    ds = Dataset.from_list([{"text": t} for t in texts])

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=a.ft_seed,
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
        use_rslora=False,
        loftq_config=None,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=a.max_seq_len,
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=a.ft_batch,
            gradient_accumulation_steps=a.ft_accum,
            num_train_epochs=a.ft_epochs,
            learning_rate=a.ft_lr,
            warmup_steps=5,
            logging_steps=5,
            fp16=not bf16,
            bf16=bf16,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=a.ft_seed,
            output_dir=a.out + "_ckpt",
            report_to="none",
        ),
    )
    stats = trainer.train()

    model.save_pretrained(a.out + "_lora")
    tok.save_pretrained(a.out + "_lora")
    model.save_pretrained_gguf(a.out + "_gguf", tok, quantization_method=a.ft_quant)

    # 不同 unsloth 版本的导出目录/文件名略有差异：递归查找，并优先取量化后的那个文件（避免误选 F16）
    ggufs = sorted(glob.glob(f"{a.out}_gguf*/**/*.gguf", recursive=True))
    if not ggufs:
        print("未找到 .gguf，请检查 unsloth 的导出目录。")
        return
    gguf = next(
        (g for g in ggufs if a.ft_quant.upper() in os.path.basename(g).upper()),
        ggufs[0],
    )
    out_dir = os.path.dirname(os.path.abspath(gguf))
    mf = os.path.join(out_dir, "Modelfile.demo")
    with open(mf, "w", encoding="utf-8") as f:
        f.write(MODELFILES[style].format(gguf=os.path.abspath(gguf)))

    meta = {
        "style": style,
        "hf_model": hf_model,
        "quant": a.ft_quant,
        "gguf": os.path.abspath(gguf),
        "prompt_sha1": prompt_fingerprint(),
        "train_file": a.train,
        "n_train": len(train),
        "max_seq_len": a.max_seq_len,
        "n_truncated": n_trunc,
        "mean_tokens": round(sum(lens) / len(lens), 1),
        "lora_r": 16,
        "epochs": a.ft_epochs,
        "lr": a.ft_lr,
        "per_device_batch": a.ft_batch,
        "grad_accum": a.ft_accum,
        "bf16": bf16,
        "seed": a.ft_seed,
        "train_loss": getattr(stats, "training_loss", None),
        "train_runtime_sec": (getattr(stats, "metrics", {}) or {}).get("train_runtime"),
        "hardware": a.hardware,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "ft_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"\n训练完成：loss={meta['train_loss']}  用时={meta['train_runtime_sec']}s  元数据 -> {out_dir}/ft_meta.json"
    )
    print(f"导入 ollama:  ollama create {a.ft_name} -f {mf}")
    print(
        f"评估(纯微调):  python nile_llm.py predict --model {a.ft_name} --mode base --limit 50"
    )
    print(
        f"评估(FT+ICL):  python nile_llm.py predict --model {a.ft_name} --mode icl --shots 9 --limit 50"
    )


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

    # ---- 微调设置（只对 finetune 子命令有用；默认值对齐原论文 finetuning.py） ----
    ap.add_argument(
        "--ft_style", choices=["paper", "llama3"], default="llama3"
    )  # paper=论文做法(基座模型+USER/ASSISTANT 模板)；llama3=Instruct 模型+官方模板
    ap.add_argument(
        "--hf_model", default=None
    )  # 微调的基座(HuggingFace 4bit)；不填则按 --ft_style 选择：paper->Meta-Llama-3.1-8B-bnb-4bit，llama3->...-Instruct-bnb-4bit
    ap.add_argument(
        "--out", default="nile_ft"
    )  # 输出名前缀：生成 nile_ft_lora（LoRA 权重）和 nile_ft_gguf*（导出模型 + Modelfile.demo + ft_meta.json）
    ap.add_argument(
        "--ft_name", default="llama31-nile-ft"
    )  # 导入 ollama 时使用的模型名；两种 style 都训练时请用不同的 --out 和 --ft_name
    ap.add_argument(
        "--max_seq_len", type=int, default=2048
    )  # 微调时单条样本最大 token 长度（训练时会打印是否有样本被截断）
    ap.add_argument("--ft_epochs", type=int, default=3)  # 论文 3
    ap.add_argument("--ft_lr", type=float, default=2e-4)  # 论文 2e-4
    ap.add_argument(
        "--ft_batch", type=int, default=8
    )  # 每卡 batch；A100 上 8 x accum 1 = 论文的 2 x 4（有效 batch 都是 8）
    ap.add_argument(
        "--ft_accum", type=int, default=1
    )  # 梯度累积步数；显存不够就调小 --ft_batch、调大它，保持乘积为 8
    ap.add_argument(
        "--ft_seed", type=int, default=3407
    )  # 论文用 3407（LoRA 初始化 + 训练），与推理用的 --seed 分开
    ap.add_argument(
        "--ft_quant", default="q4_k_m"
    )  # GGUF 量化方式；与 ollama 里 llama3.1:8b 默认的 Q4_K_M 一致

    a = ap.parse_args()
    if a.cmd == "eval":
        print(
            "提示：eval 已更名为 predict；本脚本不再计算指标，请用 evaluation.py metrics"
        )
        a.cmd = "predict"
    {"check": cmd_check, "predict": cmd_predict, "finetune": cmd_finetune}[a.cmd](
        a
    )  # 按子命令名调用对应函数


if __name__ == "__main__":
    main()
