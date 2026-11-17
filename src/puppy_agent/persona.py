from datetime import datetime
from typing import Any, Iterable

from .config import AppSettings


CATEGORY_LABELS = {
    "preference": "喜好",
    "hobby": "爱好",
    "habit": "习惯",
    "goal": "计划",
    "identity": "身份",
    "work": "工作",
    "study": "学习",
    "pet": "宠物",
    "important_date": "重要日子",
    "health": "身体",
    "family_conflict": "家里",
    "relationship_conflict": "感情",
    "address": "住处",
    "contact": "联系方式",
    "account": "账号",
    "crisis": "心情",
}


def memory_text(memory: Any) -> str:
    """Return only the remembered sentence.

    Storage rows, candidate objects and plain strings all reach this function.
    Building the prompt from ``str(memory)`` used to inject the whole dataclass
    representation (ids, timestamps, flags) into the model's context, which
    wasted tokens and buried the actual fact.
    """
    if isinstance(memory, str):
        return memory.strip()
    if isinstance(memory, dict):
        return str(memory.get("content") or "").strip()
    return str(getattr(memory, "content", "") or "").strip()


def memory_category(memory: Any) -> str:
    if isinstance(memory, dict):
        return str(memory.get("category") or "").strip()
    return str(getattr(memory, "category", "") or "").strip()


def _memory_line_parts(memory: Any):
    content = memory_text(memory)
    if not content:
        return "", ""
    label = CATEGORY_LABELS.get(memory_category(memory))
    prefix = "- [{}] ".format(label) if label else "- "
    return prefix, content


def memory_line(memory: Any) -> str:
    """One prompt line for a memory; empty when there is nothing to say."""
    prefix, content = _memory_line_parts(memory)
    return prefix + content


def format_memories(memories: Iterable[Any], *, offsets=None, start=0) -> str:
    lines = []
    for memory in memories:
        prefix, content = _memory_line_parts(memory)
        if not content:
            continue
        lines.append(prefix + content)
        if offsets is not None:
            content_start = start + len(prefix)
            offsets.append((getattr(memory, "id", None), content_start))
            start = content_start + len(content) + 1  # separator after this line
    return "\n".join(lines) or "- 暂无"


def build_system_prompt(settings: AppSettings, memories: Iterable[Any], now: datetime, *, memory_offsets=None) -> str:
    """Neutral public prompt; memory formatting and offset accounting are retained."""
    intro = (
        f"你是{settings.pet_name}，一个对话助手。\n"
        f"当前时间：{now:%Y-%m-%d %H:%M}\n"
        "只使用以下已提供的记忆，不要编造用户信息。\n"
        "记忆：\n"
    )
    remembered = format_memories(memories, offsets=memory_offsets, start=len(intro))
    return intro + remembered + "\n缺少信息时说明不确定。"
