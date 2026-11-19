# -*- coding: utf-8 -*-
"""
evaluation.py —— 对 NILE 生成结果做 L1–L4 离线评估

  L1  NLP 相似度        bleu、rouge_l、cosine_sim（附带 syntax_valid 辅助标记）
  L2  规范化 EM           strict_em、canon_em
  L3  编译后比较(无别名)   NILE -> 原子规则集合（单位换算/大小写与空白/时间补零 + 语义自洽检查），
                        输出规则等价性与规则级 Precision / Recall / F1
  L4  LLM-judge           按槽位/维度给出结论 + 总体语义等价判定，支持多模型/多次投票

用法:
  python evaluation.py metrics --preds preds_*.jsonl [--intersect]
  python evaluation.py judge   --eval eval_base_llama3.1-8b.jsonl --judge_models qwen2.5:32b

依赖: numpy, requests
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import requests

from prompts import judge_fingerprint, judge_messages
from utils import (
    FOR_ONE,
    FROM_TO,
    TIME_PARTS,
    norm_ws,
    parse_args_from_item,
    parse_nile,
)

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST


# ----------------------------------------------------------------------------
# 1. L1：NLP 指标（BLEU / ROUGE-L / Cosine Similarity）
# ----------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.:/-]+", (text or "").lower())


def bleu_score_sentence(pred: str, gold: str, max_n: int = 4) -> float:
    p_toks, g_toks = _tokenize(pred), _tokenize(gold)
    if not p_toks or not g_toks:
        return 0.0
    bp = 1.0 if len(p_toks) >= len(g_toks) else math.exp(1.0 - len(g_toks) / len(p_toks))
    log_precs = []
    for n in range(1, max_n + 1):
        if len(p_toks) < n or len(g_toks) < n:
            log_precs.append(math.log(1e-9))
            continue
        p_ngrams = Counter(tuple(p_toks[i : i + n]) for i in range(len(p_toks) - n + 1))
        g_ngrams = Counter(tuple(g_toks[i : i + n]) for i in range(len(g_toks) - n + 1))
        overlap = sum(min(cnt, g_ngrams[ng]) for ng, cnt in p_ngrams.items())
        total = max(sum(p_ngrams.values()), 1)
        prec = (overlap + 1e-6) / (total + 1e-6) if overlap == 0 else overlap / total
        log_precs.append(math.log(max(prec, 1e-9)))
    return float(bp * math.exp(sum(log_precs) / max_n))


def rouge_l_f1(pred: str, gold: str) -> float:
    a, b = _tokenize(pred), _tokenize(gold)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec = lcs / len(a)
    rec = lcs / len(b)
    return float(2 * prec * rec / (prec + rec))


def cosine_sim_bow(pred: str, gold: str) -> float:
    ca, cb = Counter(_tokenize(pred)), Counter(_tokenize(gold))
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[k] * cb[k] for k in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


def l1_nlp(pred: str, gold: str) -> dict:
    p, err = parse_nile(pred)
    return {
        "syntax_valid": p is not None,
        "syntax_error": err or "",
        "bleu": bleu_score_sentence(pred, gold),
        "rouge_l": rouge_l_f1(pred, gold),
        "cosine_sim": cosine_sim_bow(pred, gold),
    }


# ----------------------------------------------------------------------------
# 2. L2：规范化 Exact Match
# ----------------------------------------------------------------------------
def _items_norm(action: str, items: list[str]) -> list[str]:
    out = []
    for it in items:
        name, args = parse_args_from_item(it)
        if name in ("bandwidth", "quota"):
            num, unit = str(int(args[1])), args[2]
            if action == "unset":
                num = "*"
                unit = "*" if name == "bandwidth" else "*/" + unit.split("/")[-1]
            args = [args[0], num, unit]
        out.append(f"{name}({','.join(args)})")
    return out


def canon(s: str):
    p, _ = parse_nile(s)
    if p is None:
        return None
    return (
        norm_ws(p["scope"]),
        p["action"],
        tuple(sorted(_items_norm(p["action"], p["items"]))),
        norm_ws(p["time"]),
    )


def l2_norm_em(pred: str, gold: str) -> dict:
    valid = parse_nile(pred)[0] is not None
    return {
        "strict_em": norm_ws(pred) == norm_ws(gold),
        "canon_em": valid and (canon(pred) == canon(gold)),
    }


# ----------------------------------------------------------------------------
# 3. L3：编译后比较（无别名表）
#    NILE -> 原子规则集合 {(scope, action, obj, time)}
#    保留：单位换算、大小写/空白归一化、时间补零、语义自洽检查、规则级 P/R/F1
# ----------------------------------------------------------------------------
BW = {"bps": 1, "kbps": 10**3, "mbps": 10**6, "gbps": 10**9}
QMB = {"mb": 1, "gb": 1000}


def _hour(v: str) -> str:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", v or "")
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else (v or "").strip()


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return p, r, f


def compile_nile_rules(s: str):
    """返回 (规则集合 | None, 警告列表, 错误信息 | None)。"""
    p, err = parse_nile(s)
    if p is None:
        return None, [], err

    warns = []
    m = FROM_TO.match(p["scope"])
    if m:
        scope = ("from_to", norm_ws(m.group(1)).lower(), norm_ws(m.group(2)).lower())
    else:
        k, v = FOR_ONE.match(p["scope"]).groups()
        scope = ("for", k.lower(), norm_ws(v).lower())

    tm = ()
    if p["time"]:
        kind, a, b = TIME_PARTS.search(p["time"]).groups()
        if kind == "hour":
            a, b = _hour(a), _hour(b)
        else:
            a, b = a.strip(), b.strip()
            if a > b:
                warns.append("date_end_before_start")
        tm = (kind, a, b)

    rules, bw = set(), {}
    unset = p["action"] == "unset"
    for it in p["items"]:
        name, args = parse_args_from_item(it)
        lname = name.lower()
        if lname == "bandwidth":
            val = "*" if unset else int(args[1]) * BW[args[2].lower()]
            obj = ("bandwidth", args[0].lower(), val)
            if not unset:
                bw[args[0].lower()] = val
        elif lname == "quota":
            amt, per = args[2].lower().split("/")
            obj = ("quota", args[0].lower(), "*" if unset else int(args[1]) * QMB[amt], per)
        else:
            obj = (lname, norm_ws(args[0]).lower())
        rules.add((scope, p["action"].lower(), obj, tm))

    if "min" in bw and "max" in bw and bw["min"] > bw["max"]:
        warns.append("bandwidth_min_gt_max")

    return rules, warns, None


def l3_rules(pred: str, gold: str) -> dict:
    gr, _, _ = compile_nile_rules(gold)
    pr, warns, err = compile_nile_rules(pred)
    if pr is None or gr is None:
        return {
            "compile_ok": False,
            "compile_error": err or "",
            "warnings": [],
            "equiv": False,
            "rule_tp": 0,
            "rule_fp": 0,
            "rule_fn": len(gr or []),
            "rule_precision": 0.0,
            "rule_recall": 0.0,
            "rule_f1": 0.0,
        }
    tp = len(pr & gr)
    fp = len(pr - gr)
    fn = len(gr - pr)
    p, r, f = prf(tp, fp, fn)
    return {
        "compile_ok": len(warns) == 0,
        "compile_error": "",
        "warnings": warns,
        "equiv": pr == gr,
        "rule_tp": tp,
        "rule_fp": fp,
        "rule_fn": fn,
        "rule_precision": p,
        "rule_recall": r,
        "rule_f1": f,
    }


# ----------------------------------------------------------------------------
# 4. 汇总与指标计算
# ----------------------------------------------------------------------------
def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (0.0, 0.0)
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(rows, tag):
    rows = [r for r in rows if not r["gold_invalid"]]
    n = len(rows)
    if n == 0:
        return {"tag": tag, "n": 0}
    col = lambda f: [f(r) for r in rows]

    S = {
        "tag": tag,
        "n": n,
        "L1_syntax_valid": float(np.mean(col(lambda r: r["l1"]["syntax_valid"]))),
        "L1_bleu": float(np.mean(col(lambda r: r["l1"]["bleu"]))),
        "L1_rouge_l": float(np.mean(col(lambda r: r["l1"]["rouge_l"]))),
        "L1_cosine_sim": float(np.mean(col(lambda r: r["l1"]["cosine_sim"]))),
        "L2_strict_em": float(np.mean(col(lambda r: r["l2"]["strict_em"]))),
        "L2_canon_em": float(np.mean(col(lambda r: r["l2"]["canon_em"]))),
        "L3_compile_ok": float(np.mean(col(lambda r: r["l3"]["compile_ok"]))),
        "L3_equiv": float(np.mean(col(lambda r: r["l3"]["equiv"]))),
        "L3_rescued_vs_L2": int(sum(r["l3"]["equiv"] and not r["l2"]["canon_em"] for r in rows)),
        "L3_lost_vs_L2": int(sum(r["l2"]["canon_em"] and not r["l3"]["equiv"] for r in rows)),
    }
    S["L2_canon_em_ci95"] = boot_ci(col(lambda r: r["l2"]["canon_em"]))
    S["L3_equiv_ci95"] = boot_ci(col(lambda r: r["l3"]["equiv"]))

    for k in ("sec", "prompt_tokens", "prompt_chars", "gen_tokens"):
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            S[f"mean_{k}"] = float(np.mean(v))

    rtp = sum(r["l3"]["rule_tp"] for r in rows)
    rfp = sum(r["l3"]["rule_fp"] for r in rows)
    rfn = sum(r["l3"]["rule_fn"] for r in rows)
    S["L3_rule_P"], S["L3_rule_R"], S["L3_rule_F1"] = prf(rtp, rfp, rfn)

    S["by_gold_action"] = {}
    for act in sorted({r["gold_action"] for r in rows}):
        sub = [r for r in rows if r["gold_action"] == act]
        S["by_gold_action"][act] = {
            "n": len(sub),
            "canon_em": float(np.mean([r["l2"]["canon_em"] for r in sub])),
            "l3_equiv": float(np.mean([r["l3"]["equiv"] for r in sub])),
            "rule_f1": float(np.mean([r["l3"]["rule_f1"] for r in sub])),
        }

    warn_counter = Counter(w for r in rows for w in r["l3"].get("warnings", []))
    S["l3_warnings"] = dict(warn_counter.most_common())
    return S


def cmd_metrics(a):
    files = sorted({f for pat in a.preds for f in glob.glob(pat)})
    raws = {fp_: [json.loads(l) for l in open(fp_, encoding="utf-8") if l.strip()] for fp_ in files}
    common = None
    if a.intersect and raws:      # 不同方法有 [FAIL] 时 n 会不同；取各文件共有的 id，保证对比在同一批样本上
        common = set.intersection(*[{r["id"] for r in v} for v in raws.values()])
        print(f"--intersect：{len(raws)} 个文件的共有样本 {len(common)} 条（各文件原有 "
              f"{', '.join(str(len(v)) for v in raws.values())} 条）")
    table = []
    for fp_ in files:
        tag = re.sub(r"^preds_", "", os.path.splitext(os.path.basename(fp_))[0])
        raw = raws[fp_]
        n_raw = len(raw)
        if common is not None:
            raw = [r for r in raw if r["id"] in common]
        rows, n_bad = [], 0
        for r in raw:
            gp, _ = parse_nile(r["gold"])
            gold_invalid = gp is None
            n_bad += gold_invalid
            row = {
                "id": r.get("id"),
                "nl": r.get("nl", ""),
                "gold": r.get("gold", ""),
                "pred": r.get("pred", ""),
                "gold_invalid": gold_invalid,
                "gold_action": gp["action"] if gp else "?",
            }
            for k in ("sec", "prompt_tokens", "prompt_chars", "gen_tokens"):
                if r.get(k) is not None:
                    row[k] = r[k]
            if not gold_invalid:
                row["l1"] = l1_nlp(row["pred"], row["gold"])
                row["l2"] = l2_norm_em(row["pred"], row["gold"])
                row["l3"] = l3_rules(row["pred"], row["gold"])
            else:
                row["l1"], row["l2"], row["l3"] = {}, {}, {}
            rows.append(row)

        S = summarize(rows, tag)
        S["gold_invalid_excluded"] = n_bad
        run = os.path.join(os.path.dirname(fp_), f"run_{tag}.json")
        if os.path.exists(run):
            meta = json.load(open(run, encoding="utf-8"))
            S["run"] = {
                k: meta.get(k)
                for k in ("model", "mode", "shots", "split", "seed", "hardware", "num_ctx", "workers")
            }
            S["run"].update({f"model_{k}": v for k, v in (meta.get("model_info") or {}).items()})
            S["n_requested"] = meta.get("n_requested")
        S["n_pred_rows"] = n_raw
        nreq = S.get("n_requested")
        S["coverage"] = (n_raw / nreq) if nreq else None

        with open(f"eval_{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(f"metrics_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(S, f, ensure_ascii=False, indent=2)

        print(f"\n=== {tag}  (n={S['n']}, gold 不可解析被排除 {n_bad}) ===")
        if S.get("n_requested") and n_raw < S["n_requested"]:
            print(f"  [警告] 只有 {n_raw}/{S['n_requested']} 条预测（有 [FAIL] 未补跑）；"
                  f"与其他方法比较前请补跑，或使用 --intersect")
        for k in (
            "L1_bleu",
            "L1_rouge_l",
            "L1_cosine_sim",
            "L1_syntax_valid",
            "L2_strict_em",
            "L2_canon_em",
            "L3_compile_ok",
            "L3_equiv",
            "L3_rule_P",
            "L3_rule_R",
            "L3_rule_F1",
            "L3_rescued_vs_L2",
            "L3_lost_vs_L2",
        ):
            v = S.get(k)
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(
            "  canon_em 95%CI:",
            [round(x, 4) for x in S["L2_canon_em_ci95"]],
            " l3_equiv 95%CI:",
            [round(x, 4) for x in S["L3_equiv_ci95"]],
        )
        print("  l3_warnings:", S["l3_warnings"])
        table.append(S)

    if table:
        flat = [{**S, **(S.get("run") or {})} for S in table]
        keys = [
            "tag",
            "model",
            "mode",
            "shots",
            "hardware",
            "model_quantization_level",
            "n",
            "n_requested",
            "coverage",
            "L1_syntax_valid",
            "L1_bleu",
            "L1_rouge_l",
            "L1_cosine_sim",
            "L2_strict_em",
            "L2_canon_em",
            "L3_equiv",
            "L3_rule_F1",
            "mean_sec",
            "mean_prompt_tokens",
            "mean_prompt_chars",
            "mean_gen_tokens",
        ]
        with open("metrics_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for S in flat:
                w.writerow([round(S[k], 4) if isinstance(S.get(k), float) else S.get(k, "") for k in keys])
        print("\n汇总表: metrics_table.csv")


# ----------------------------------------------------------------------------
# 5. L4：LLM-judge
# ----------------------------------------------------------------------------
def call_judge(backend, model, messages, temperature, seed):
    if backend == "ollama":
        r = requests.post(
            f"{HOST}/api/chat",
            timeout=600,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_predict": 300,
                    "num_ctx": 4096,
                },
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    base = os.environ["JUDGE_BASE_URL"].rstrip("/")
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 300,
    }
    if not os.environ.get("JUDGE_NO_JSON_MODE"):
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        f"{base}/chat/completions",
        timeout=600,
        json=body,
        headers={"Authorization": "Bearer " + os.environ.get("JUDGE_API_KEY", "")},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_judge(text):
    text = text or ""
    cands = [text]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        cands.append(m.group(0))
    for cand in cands:
        try:
            j = json.loads(cand)
            if isinstance(j, dict) and "equivalent" in j:
                eq = j["equivalent"]
                if isinstance(eq, str):
                    eq = eq.strip().lower() == "true"
                j["equivalent"] = bool(eq)
                return j
        except Exception:
            pass
    return None


def judge_row(row, models, backend, mode, repeats):
    votes = []
    for m in models:
        for k in range(repeats):
            temp = 0.0 if repeats == 1 else 0.3
            j = None
            for attempt in range(3):
                try:
                    j = parse_judge(call_judge(backend, m, judge_messages(row, mode), temp, 1000 + k))
                    break
                except Exception:
                    j = None
                    time.sleep(2 * (attempt + 1))
            if j:
                j["model"] = m
                votes.append(j)
    if not votes:
        return {"verdict": None, "n_votes": 0}
    pos = sum(v["equivalent"] for v in votes)
    verdict = pos * 2 > len(votes)
    maj = [v for v in votes if v["equivalent"] == verdict] or votes
    return {
        "verdict": verdict,
        "n_votes": len(votes),
        "agree": max(pos, len(votes) - pos) / len(votes),
        "slots": {k: maj[0].get(k) for k in ("scope", "action", "items", "time")},
        "error_type": maj[0].get("error_type"),
        "rationale": maj[0].get("rationale", ""),
    }


def _dump(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def judge_summary(rows, cfg):
    rows = [r for r in rows if not r["gold_invalid"]]
    n = len(rows)
    judged = [r for r in rows if r.get("judge", {}).get("verdict") is not None and r["judge"].get("n_votes", 0) > 0]
    final = []
    for r in rows:
        j = r.get("judge", {})
        if j.get("verdict") is not None:
            final.append(bool(j["verdict"]))
        else:
            final.append(bool(r.get("l3", {}).get("equiv")) and bool(r.get("l1", {}).get("syntax_valid")))
    rescue = [r for r in judged if r["judge"]["verdict"] and not r.get("l3", {}).get("equiv")]
    veto = [r for r in judged if (not r["judge"]["verdict"]) and r.get("l3", {}).get("equiv")]
    nonequiv_judged = [r for r in judged if not r.get("l3", {}).get("equiv")]
    return {
        "cfg": cfg,
        "n": n,
        "judged": len(judged),
        "judge_parse_fail": sum(
            1 for r in rows if r.get("judge", {}).get("n_votes", 1) == 0 and "auto" not in r.get("judge", {})
        ),
        "L4_accuracy": float(np.mean(final)) if final else 0.0,
        "L4_accuracy_ci95": boot_ci([float(x) for x in final]),
        "judge_rescue_count": len(rescue),
        "judge_rescue_rate_of_nonequiv": len(rescue) / len(nonequiv_judged) if nonequiv_judged else 0.0,
        "judge_veto_count": len(veto),
        "judge_mean_vote_agreement": float(np.mean([r["judge"]["agree"] for r in judged])) if judged else 0.0,
        "error_types_judged_wrong": dict(
            Counter(r["judge"].get("error_type") for r in judged if not r["judge"]["verdict"]).most_common()
        ),
    }


def cmd_judge(a):
    models = [m.strip() for m in a.judge_models.split(",") if m.strip()]
    cfg = (
        f"{a.judge_backend}|{','.join(models)}|{a.judge_mode}|r{a.judge_repeats}|"
        f"{a.judge_scope}|{a.judge_equiv_sample}|{judge_fingerprint()}"
    )
    rng = random.Random(a.seed)
    for path in sorted({f for pat in a.eval for f in glob.glob(pat)}):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        todo = []
        for r in rows:
            if r["gold_invalid"] or not r.get("l1", {}).get("syntax_valid"):
                r["judge"] = {"verdict": False, "n_votes": 0, "auto": "syntax_invalid", "cfg": cfg}
                continue
            if r.get("judge", {}).get("cfg") == cfg:
                continue
            if a.judge_scope == "nonequiv" and r.get("l3", {}).get("equiv") and rng.random() >= a.judge_equiv_sample:
                r["judge"] = {"verdict": None, "n_votes": 0, "auto": "skipped_l3_equiv", "cfg": cfg}
                continue
            todo.append(r)
        print(f"\n{path}: 需要 judge {len(todo)} 条 / 共 {len(rows)} 条  (cfg={cfg})")
        for i, r in enumerate(todo, 1):
            r["judge"] = judge_row(r, models, a.judge_backend, a.judge_mode, a.judge_repeats)
            r["judge"]["cfg"] = cfg
            if i % 25 == 0 or i == len(todo):
                _dump(path, rows)
                print(f"  [{i}/{len(todo)}]", flush=True)
        _dump(path, rows)
        S = judge_summary(rows, cfg)
        tag = re.sub(r"^eval_", "", os.path.splitext(os.path.basename(path))[0])
        json.dump(S, open(f"judge_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"=== L4 {tag} ===")
        for k, v in S.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    m = sp.add_parser("metrics")
    m.add_argument("--preds", nargs="+", required=True)
    m.add_argument("--intersect", action="store_true",
                   help="只在各预测文件共有的 id 上计算，保证不同方法的 n 一致（有 FAIL 样本时用）")

    j = sp.add_parser("judge")
    j.add_argument("--eval", nargs="+", required=True)
    j.add_argument("--judge_models", required=True)
    j.add_argument("--judge_backend", choices=["ollama", "openai"], default="ollama")
    j.add_argument("--judge_mode", choices=["ref", "noref"], default="ref")
    j.add_argument("--judge_repeats", type=int, default=1)
    j.add_argument("--judge_scope", choices=["all", "nonequiv"], default="all")
    j.add_argument("--judge_equiv_sample", type=float, default=0.1)
    j.add_argument("--seed", type=int, default=42)

    a = ap.parse_args()
    {"metrics": cmd_metrics, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    main()