"""Deterministic ADD / UPDATE / NOOP decisions for extracted facts.

The model only proposes facts.  It never decides whether an existing memory
should be replaced: overwriting is a data-loss operation, so the rules live in
code where they can be tested and reviewed.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


ADD = "add"
UPDATE = "update"
NOOP = "noop"

# An explicit correction is the only signal that lets a new value retire an old
# one.  "喜欢咖啡" plus "喜欢茶" must coexist; "不再喝咖啡，改喝茶" must not.
CORRECTION_MARKERS = (
    "不再", "不喝", "不吃", "不去", "不喜欢", "不想要", "戒了", "戒掉",
    "改成", "改为", "改喝", "改吃", "改去", "改叫", "改到", "换成了", "换成",
    "现在改", "最近改", "已经改", "现在变成", "改成", "而不是", "反而不",
    "现在已经", "现在住", "现在在", "搬家", "搬去", "搬到", "分手", "离职",
    "换工作", "换了工作", "改了",
    "别推荐", "别再", "不要再", "不推荐",
)

HIGH_SIMILARITY = 0.95


def normalize(text) -> str:
    return " ".join(str(text or "").split()).casefold()


def correction_intent(turn_text: str) -> bool:
    """True when the owner is correcting an earlier statement."""
    text = str(turn_text or "")
    return any(marker in text for marker in CORRECTION_MARKERS)


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    memory_id: Optional[int] = None
    supersede: tuple = ()

    @property
    def writes(self) -> bool:
        return self.action in (ADD, UPDATE)


def _rows_with_same_value(existing: Sequence, value: str):
    key = normalize(value)
    return [row for row in existing if normalize(getattr(row, "value", None) or row.content) == key]


def decide(candidate, existing: Iterable = (), turn_text: str = "",
           similarity: Optional[float] = None) -> Decision:
    """Choose what to do with one candidate fact.

    ``existing`` holds the active rows for the same (user, subject, predicate)
    key.  When the extractor could not name a predicate the caller passes an
    empty sequence, because an unnamed predicate cannot safely identify a
    previous value.
    """
    rows = list(existing)
    value = getattr(candidate, "value", None)
    if value is None:
        value = getattr(candidate, "content", "")
    if not normalize(value):
        return Decision(NOOP, "empty value")

    same_value = _rows_with_same_value(rows, value)
    if same_value:
        return Decision(NOOP, "same value already active", same_value[0].id)
    if similarity is not None and similarity >= HIGH_SIMILARITY:
        return Decision(NOOP, "near-duplicate of an active memory",
                        rows[0].id if rows else None)

    if rows and correction_intent(turn_text):
        # Retire every active value for this predicate, newest first, so the
        # stored history still shows what used to be true.
        ordered = sorted(rows, key=lambda row: row.id, reverse=True)
        return Decision(UPDATE, "explicit correction", ordered[0].id,
                        tuple(row.id for row in ordered))
    if rows:
        return Decision(ADD, "coexisting value for the same predicate")
    return Decision(ADD, "new fact")
