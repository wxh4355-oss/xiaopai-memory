from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Sensitivity(str, Enum):
    LOW = "low"
    SENSITIVE = "sensitive"


@dataclass(frozen=True)
class MemoryCandidate:
    """One proposed fact, before the ADD/UPDATE/NOOP decision runs."""

    category: str
    content: str
    sensitivity: Sensitivity
    source: str
    subject: str = "用户"
    predicate: Optional[str] = None
    value: Optional[str] = None
    source_message_id: Optional[int] = None
    importance: float = 0.5
    user_id: str = "user_001"
    # The sentence this fact came from, kept in memory only: it lets the
    # ADD/UPDATE/NOOP decision see an explicit correction ("改喝茶了").
    turn_text: str = ""
    scene: str = "default"
    valid_to: Optional[str] = None

    def __post_init__(self):
        if self.predicate is not None:
            object.__setattr__(self, "predicate", str(self.predicate).strip() or None)
        if self.value is None:
            object.__setattr__(self, "value", self.content)
        object.__setattr__(self, "importance",
                           max(0.0, min(1.0, float(self.importance or 0.5))))
