# -*- coding: utf-8 -*-
"""
nile_prompts.py —— 本项目所有提示词的唯一来源（改提示词只改这个文件）

  推理侧   SYSTEM_BASE / user_msg / build_system   被 nile_llm_demo.py 使用（predict 与 finetune 共用同一份）
  评估侧   JUDGE_SYSTEM / MODE_NOTE / judge_messages 被 nile_eval.py 的 L4 LLM-judge 使用

注意：
  - 微调(finetune)用的就是这里的 SYSTEM_BASE。改了它之后，之前微调出来的模型和旧的预测都与新提示不一致，
    需要重新微调 / 重新推理，否则对比不公平。run_*.json 里会记录 prompt_sha1，可用来核对。
  - judge 提示词改了之后，旧的 judge 结果会因 cfg 里的 judge_sha1 不同而自动重判。
"""
import hashlib

# ----------------------------------------------------------------------------
# 推理侧（论文 3.2：输出模板 + 示例 + 用户意图 三段）
# ----------------------------------------------------------------------------
SYSTEM_BASE = """You translate network intents written in natural language into NILE (Network Intent LanguagE) statements.
Output ONLY one NILE statement on a single line. No explanation, no markdown.

Template:
define intent uniIntent: <scope> <action> <items> [<time window>]

<scope> is one of:
  for endpoint('X') | for group('X') | for service('X') | for traffic('X')
  from endpoint('X') to endpoint('Y')
<action> <items>:
  allow|block  service('..'), traffic('..'), protocol('..')   (comma-separated, any order)
  set|unset    bandwidth('max'|'min','<number>','<bps|kbps|mbps|gbps>'),
               quota('download'|'upload','<number>','<mb/d|gb/d|mb/wk|gb/wk|mb/mth|gb/mth>')
  add|remove   middlebox('..')
<time window> (optional): start hour('HH:MM') end hour('HH:MM')  or  start date('YYYYMMDD') end date('YYYYMMDD')
Use 'any' when the intent applies to everything. Keep names exactly as in the NILE vocabulary
(e.g. 'peer2peer' for peer-to-peer, 'streaming movies', 'Sorenson Videophone 200', 'dorms' for ResNet)."""


def user_msg(nl):
    return f"Intent: {nl}\nNILE:"


def build_system(examples=None):
    """system 提示 = 输出模板 [+ few-shot 示例]。examples 为 [{'natural_language':..., 'nile':...}, ...]"""
    system = SYSTEM_BASE
    if examples:
        system += "\n\nExamples:\n" + "\n\n".join(
            f"Intent: {e['natural_language']}\nNILE: {e['nile']}" for e in examples)
    return system


def prompt_fingerprint():
    return hashlib.sha1(SYSTEM_BASE.encode("utf-8")).hexdigest()[:10]


# ----------------------------------------------------------------------------
# 评估侧：L4 LLM-judge
# ----------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a strict evaluator of network-intent translations.
A model translated a natural-language network intent into a NILE statement.

NILE format: define intent uniIntent: <scope> <action> <items> [<time window>]
  scope : for endpoint|group|service|traffic('X')  or  from endpoint('X') to endpoint('Y')
  action: allow|block (service/traffic/protocol), set|unset (bandwidth/quota), add|remove (middlebox)
  time  : start hour('HH:MM') end hour('HH:MM')  or  start date('YYYYMMDD') end date('YYYYMMDD')

Equivalence rules (apply them exactly):
1. The order of items inside one list does not matter.
2. Case, whitespace, leading zeros and unit conversions (e.g. 1 gbps = 1000 mbps) do not matter.
3. For 'unset', the numbers/units of bandwidth and quota cannot be recovered from the intent: ignore them.
4. NOT equivalent: a different entity / service / traffic / protocol / middlebox name; a different action;
   swapped source and destination; a missing or extra item; a missing or extra time window; a different scope type.
{mode_note}
Judge each component, then give an overall verdict. Answer with ONE JSON object and nothing else:
{{"scope":"correct|wrong", "action":"correct|wrong", "items":"correct|wrong|missing|extra",
 "time":"correct|wrong|missing|extra", "equivalent":true|false,
 "error_type":"none|wrong_entity|wrong_action|wrong_scope|omission|hallucination|wrong_value|wrong_time|other",
 "rationale":"<=25 words"}}"""
MODE_NOTE = {"ref": "A reference NILE is given: the candidate is correct iff it is equivalent to the reference.",
             "noref": "No reference is given: the candidate is correct iff it faithfully and completely expresses the intent."}


def judge_messages(row, mode):
    sys_ = JUDGE_SYSTEM.format(mode_note=MODE_NOTE[mode])
    if mode == "ref":
        u = f"Intent: {row['nl']}\nReference NILE: {row['gold']}\nCandidate NILE: {row['pred']}"
    else:
        u = f"Intent: {row['nl']}\nCandidate NILE: {row['pred']}"
    return [{"role": "system", "content": sys_}, {"role": "user", "content": u}]


def judge_fingerprint():
    return hashlib.sha1((JUDGE_SYSTEM + str(sorted(MODE_NOTE.items()))).encode("utf-8")).hexdigest()[:10]
