# -*- coding: utf-8 -*-
"""
utils.py —— NILE 语法定义与基础解析工具
（与 nile_llm_demo.py 保持一致；若修改语法，两处同步）
"""
import re

SCOPE = (
    r"(?:for\s+(?:endpoint|group|service|traffic)\('[^']*'\)"
    r"|from\s+endpoint\('[^']*'\)\s+to\s+endpoint\('[^']*'\))"
)
ITEM = r"[a-z]+\('[^']*'(?:\s*,\s*'[^']*')*\)"
TIME = r"start\s+(?P<tk>hour|date)\('[^']*'\)\s+end\s+(?P=tk)\('[^']*'\)"

NILE_RE = re.compile(
    rf"^define intent uniIntent:\s*(?P<scope>{SCOPE})\s+"
    rf"(?P<action>allow|block|set|unset|add|remove)\s+"
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

FROM_TO = re.compile(r"from\s+endpoint\('([^']*)'\)\s+to\s+endpoint\('([^']*)'\)")
FOR_ONE = re.compile(r"for\s+(\w+)\('([^']*)'\)")
TIME_PARTS = re.compile(r"start\s+(hour|date)\('([^']*)'\)\s+end\s+(?:hour|date)\('([^']*)'\)")


def _nc(t: str) -> str:
    return re.sub(r"\s*,\s*", ",", (t or "").strip())


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def parse_args_from_item(it: str) -> tuple[str, list[str]]:
    name = it.split("(", 1)[0]
    args = re.findall(r"'([^']*)'", it)
    return name, args


def parse_nile(s: str) -> tuple[dict | None, str | None]:
    """返回 (解析字典, None) 或 (None, 错误信息)。"""
    m = NILE_RE.match((s or "").strip())
    if not m:
        return None, "does not match the NILE template (scope / action / items / optional time window)"
    action, body = m.group("action"), m.group("body")
    items = re.findall(ITEM, body)
    if not items or _nc(body) != _nc(", ".join(items)):
        return None, "items must be a comma-separated list of name('arg') entries"
    for it in items:
        name, args = parse_args_from_item(it)
        if name not in ALLOWED[action]:
            return None, f"item '{name}' is not allowed after '{action}'"
        if name == "bandwidth":
            if not (len(args) == 3 and args[0] in ("max", "min") and args[1].isdigit() and args[2] in BW_UNITS):
                return None, "bad bandwidth(...)"
        elif name == "quota":
            if not (len(args) == 3 and args[0] in ("download", "upload") and args[1].isdigit() and args[2] in Q_UNITS):
                return None, "bad quota(...)"
        elif len(args) != 1:
            return None, f"{name}(...) takes exactly one argument"
    return {
        "scope": m.group("scope"),
        "action": action,
        "items": items,
        "time": m.group("time") or "",
    }, None