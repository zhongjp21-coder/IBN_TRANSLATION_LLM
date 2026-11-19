"""两个用例的任务定义：输出模板、提示词构造、系统验证(system verification)、答案比较。

对应论文 3.2 Prompt Design（output template / examples / user intent 三段式）
和 3.3 的 system verification。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

# ----------------------------------------------------------------------------
# 输出模板
# ----------------------------------------------------------------------------
# 注意：论文只给出了 create_sfc 的完整样例(Figure 2)。其余 5 个动作
# (create_vnf / update_* / destroy_*) 的真实模板来自作者的 NI-testbed，论文未公开，
# 这里按 Figure 2 的字段风格补全，请按你自己的测试床/模板修改。
NFV_ACTIONS = [
    "create_vnf", "update_vnf", "destroy_vnf",
    "create_sfc", "update_sfc", "destroy_sfc",
]

NFV_TEMPLATE = """\
{
  "action": "<create_vnf | update_vnf | destroy_vnf | create_sfc | update_sfc | destroy_sfc>",
  "<the action name>": { ...fields below... }
}
Fields per action:
- create_vnf:  {"name": str, "vnf_type": "fw|nat|dpi|ids|lb",
                "flavor": {"cpu": int, "ram_gb": int, "disk_gb": int}}
- update_vnf:  {"name": str, "flavor": {only the fields to change among cpu, ram_gb, disk_gb}}
- destroy_vnf: {"name": str}
- create_sfc:  {"name": str, "max_bandwidth_mbps": int, "delay_requirement": "normal|low",
                "flow_classifiers": ["<src_ip/mask -> dst_ip/mask:port_min-port_max>"],
                "vnfs": [{"vnf_type": "fw|nat|dpi|ids|lb",
                          "flavor": {"cpu": int, "ram_gb": int, "disk_gb": int},
                          "num_instances": int}]}
- update_sfc:  {"name": str, plus only the changed fields among max_bandwidth_mbps, delay_requirement}
- destroy_sfc: {"name": str}
Defaults when not stated: SFC VNF flavor = cpu 1, ram_gb 2, disk_gb 10; delay_requirement = "normal";
match-all address = 0.0.0.0/0; "all ports" = 0-65535; 1Gbps = 1000 Mbps."""

FORMAL_TEMPLATE = """\
{
  "reachability": {"<source host>": ["<destination subnet>", ...]},
  "waypoint": {"(<source host>,<destination subnet>)": ["<switch>", ...]},
  "loadbalancing": {"(<source host>,<destination subnet>)": [["<path switch>", ...], ...]}
}"""

# ----------------------------------------------------------------------------
# JSON 解析 + 验证
# ----------------------------------------------------------------------------
Verify = Callable[[str], tuple[bool, str, object]]


def parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    return json.loads(t)


def verify_nfv(text: str):
    try:
        obj = parse_json(text)
    except Exception as e:  # 语法错误
        return False, f"Invalid JSON: {e}", None
    if not isinstance(obj, dict):
        return False, "Top-level must be a JSON object.", None
    action = obj.get("action")
    if action not in NFV_ACTIONS:
        return False, f"'action' must be one of {NFV_ACTIONS}, got {action!r}.", None
    body = obj.get(action)
    if not isinstance(body, dict):
        return False, f"Missing object field '{action}'.", None
    if action == "create_sfc":
        need = ["name", "max_bandwidth_mbps", "delay_requirement", "flow_classifiers", "vnfs"]
        miss = [k for k in need if k not in body]
        if miss:
            return False, f"create_sfc is missing fields: {miss}", None
        for i, v in enumerate(body["vnfs"]):
            for k in ("vnf_type", "flavor", "num_instances"):
                if k not in v:
                    return False, f"vnfs[{i}] is missing '{k}'.", None
            for k in ("cpu", "ram_gb", "disk_gb"):
                if k not in v["flavor"]:
                    return False, f"vnfs[{i}].flavor is missing '{k}'.", None
    return True, "", obj


def verify_formal(text: str):
    try:
        obj = parse_json(text)
    except Exception as e:
        return False, f"Invalid JSON: {e}", None
    if not isinstance(obj, dict):
        return False, "Top-level must be a JSON object.", None
    for k in ("reachability", "waypoint", "loadbalancing"):
        if k not in obj or not isinstance(obj[k], dict):
            return False, f"Missing object field '{k}'.", None
    return True, "", obj


# ----------------------------------------------------------------------------
# 答案比较（模拟“人工验证”：与标准答案比对）
# ----------------------------------------------------------------------------
def canonical(obj, parent_key: str = ""):
    """递归规范化：字典忽略键序；列表默认忽略顺序，但 waypoint 的路径顺序有意义，需保留。"""
    if isinstance(obj, dict):
        return {k: canonical(v, k) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        items = [canonical(x, parent_key) for x in obj]
        if parent_key == "waypoint":
            return items
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
    return obj


def is_correct(pred, gold) -> bool:
    if pred is None:
        return False
    return canonical(pred) == canonical(gold)


# ----------------------------------------------------------------------------
# Task 定义与 prompt 构造
# ----------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    instruction: str
    template: str
    verify: Verify


NFV_TASK = Task(
    name="nfv",
    instruction=(
        "You are a network configuration assistant. Translate the user's natural-language "
        "intent into a JSON configuration for virtual network functions (VNFs) and service "
        "function chains (SFCs). Output ONLY valid JSON that follows the template."
    ),
    template=NFV_TEMPLATE,
    verify=verify_nfv,
)

FORMAL_TASK = Task(
    name="formal",
    instruction=(
        "You are a network configuration assistant. Translate the user's natural-language "
        "network requirements into a JSON formal specification with reachability, waypoint "
        "and loadbalancing. Output ONLY valid JSON that follows the template."
    ),
    template=FORMAL_TEMPLATE,
    verify=verify_formal,
)

TASKS = {"nfv": NFV_TASK, "formal": FORMAL_TASK}


def build_messages(task: Task, intent: str, examples: list[tuple[str, object]]) -> list[dict]:
    """prompt = output template + examples + user intent (论文 3.2)"""
    system = f"{task.instruction}\n\nOutput template:\n{task.template}"
    parts = []
    if examples:
        parts.append("Here are some examples:")
        for i, (ex_intent, ex_out) in enumerate(examples, 1):
            parts.append(f"### Example {i}\nIntent: {ex_intent}\nOutput:\n{json.dumps(ex_out, ensure_ascii=False)}")
        parts.append("Now translate the following intent.")
    parts.append(f"Intent: {intent}\nOutput:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
