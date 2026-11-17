"""Public example settings. All paths must be supplied by the caller."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppSettings:
    user_name: str = "示例用户"
    giver_name: str = "示例用户"
    pet_name: str = "小拍"
    retention_days: int = 30
    context_char_budget: int = 4000
    context_message_limit: int = 12
    context_summary_limit: int = 3
    user_id: str = "user_001"
    retrieval_enabled: bool = True
    retrieval_limit: int = 20
    memory_budget_chars: int = 1200
    memory_max_items: int = 12


class AppPaths:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_file = self.root / "memory.db"
        self.index_dir = self.root / "memory_index"
