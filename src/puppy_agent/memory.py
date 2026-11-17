import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from . import facts
from .config import AppSettings
from .models import MemoryCandidate, Sensitivity
from .memory_context import (SCENES, driving_audio, midnight_after, scene_for,
                             temporary_intent, timestamp)
from .retrieval import lexical_similarity


SENSITIVE_CATEGORIES = frozenset({
    "health", "family_conflict", "relationship_conflict", "address",
    "contact", "account", "crisis",
})
ALLOWED_CATEGORIES = frozenset({
    "preference", "hobby", "habit", "goal", "identity", "work", "study",
    "pet", "important_date", "health", "family_conflict",
    "relationship_conflict", "address", "contact", "account", "crisis",
})
_SECRET = re.compile(
    r"(?i)(?:api[_ -]?key|access[_ -]?token|secret|password|密码|口令|验证码|私钥)\s*[:=：]?\s*\S+"
)
_REMEMBER = re.compile(r"^\s*(?:请)?记住[：:，,\s]*(.+?)\s*$")

EXTRACTION_PROMPT = (
    "你是事实提取器。从用户这一句话里提取值得长期记住的用户事实，只输出 JSON：\n"
    "{\"memories\":[{\"category\":\"preference\",\"subject\":\"用户\","
    "\"predicate\":\"饮品偏好\",\"value\":\"无糖拿铁\",\"content\":\"喜欢喝无糖拿铁\","
    "\"importance\":0.7,\"sensitivity\":\"low\",\"scene\":\"default\",\"temporary\":false}]}\n"
    "规则：\n"
    "- 只提取长期有效的事实：喜好、习惯、计划、身份、重要日期、经历；"
    "一次性的情绪、天气、闲聊不要提取。\n"
    "- 对明确的偏好，按适用场景分别提取。scene 可为 default、daytime（白天）、"
    "evening_return（晚间或下班返程）。一条话里有两个场景时拆成两条事实。\n"
    "- 用户明确指定仅今天或这趟适用的偏好例外时，temporary 为 true；"
    "例如「今天头疼，回去别放音乐」只提取返程不要放音乐，不提取头疼。\n"
    "- 开车时听什么、关掉音乐等归为同一个属性「驾车音频偏好」。\n"
    "- predicate 是属性名，描述同一件事时用同一个词，例如 饮品偏好、居住城市、工作单位、生日。\n"
    "- 如果这条事实属于下面列出的已有属性，predicate 必须原样复用那个词。\n"
    "- 如果这句话是在更正之前的说法，只输出更正后的事实；不要输出“变化过程”、"
    "“从A改为B”这类描述，也不要新造带“变化”的属性名。\n"
    "- value 只写事实本身，例如 无糖拿铁、杭州。content 写成一句完整的话。\n"
    "- category 只能从这里选：preference, hobby, habit, goal, identity, work, study, "
    "pet, important_date, health, family_conflict, relationship_conflict, address, "
    "contact, account, crisis。\n"
    "- importance 取 0~1，越长期稳定越高。\n"
    "- 绝不提取密码、密钥、验证码、银行卡号。\n"
    "- 没有值得记住的内容就返回 {\"memories\":[]}。"
)
_CHANGE_PREDICATE = re.compile(r"(变化|变更|改变|变动)$")


def classify_candidate(category: str, content: str, source: str,
                       predicate: Optional[str] = None,
                       value: Optional[str] = None,
                       subject: str = "用户",
                       importance: float = 0.5,
                       source_message_id: Optional[int] = None,
                       user_id: str = "user_001",
                       turn_text: str = "",
                       scene: str = "default",
                       valid_to: Optional[str] = None) -> MemoryCandidate:
    """Apply local privacy rules; a model can never downgrade sensitivity."""
    normalized = str(category).strip().lower()
    sensitivity = (
        Sensitivity.SENSITIVE if normalized in SENSITIVE_CATEGORIES else Sensitivity.LOW
    )
    return MemoryCandidate(normalized, str(content).strip(), sensitivity, str(source),
                           subject=subject, predicate=predicate, value=value,
                           source_message_id=source_message_id,
                           importance=importance, user_id=user_id,
                           turn_text=turn_text, scene=scene, valid_to=valid_to)


class MemoryService:
    def __init__(self, storage, kimi, settings: AppSettings):
        self.storage = storage
        self.kimi = kimi
        self.settings = settings
        # Extraction used to fail completely silently: a rejected request just
        # meant "no memories", which looked identical to "nothing to remember".
        self.last_error: Optional[str] = None
        self.failure_count = 0
        self.extraction_count = 0
        self.last_decision: Optional[facts.Decision] = None

    @property
    def user_id(self) -> str:
        return str(getattr(self.settings, "user_id", "user_001") or "user_001")

    def _known_predicates(self) -> List[str]:
        """Existing property names, offered to the extractor so it reuses them."""
        try:
            rows = self.storage.list_memories(active_only=True)
        except Exception:
            return []
        seen = []
        for row in rows:
            if row.user_id != self.user_id:
                continue
            predicate = (row.predicate or "").strip()
            if predicate and predicate not in seen:
                seen.append(predicate)
        return seen[:20]

    def _extraction_system_prompt(self) -> str:
        known = self._known_predicates()
        hint = "、".join(known) if known else "（暂无）"
        return EXTRACTION_PROMPT + "\n- 已有的属性名：" + hint + "。"

    def candidates_for_turn(self, text: str,
                            message_id: Optional[int] = None) -> List[MemoryCandidate]:
        explicit = self._explicit_candidate(text, message_id)
        prompt = [
            {"role": "system", "content": self._extraction_system_prompt()},
            {"role": "user", "content": text},
        ]
        self.extraction_count += 1
        try:
            result = self.kimi.complete_json(prompt)
            candidates = self._validate_candidates(result, message_id, text)
            self.last_error = None
        except Exception as exc:
            self.last_error = "{}: {}".format(type(exc).__name__, exc)
            self.failure_count += 1
            candidates = []
        if explicit and not any(
            item.category == explicit.category and (
                item.content == explicit.content or (
                    explicit.predicate is not None and item.predicate == explicit.predicate
                    and item.scene == explicit.scene and item.valid_to == explicit.valid_to
                )
            )
            for item in candidates
        ):
            candidates.insert(0, explicit)
        return candidates[:5]

    def _explicit_candidate(self, text: str,
                            message_id: Optional[int] = None) -> Optional[MemoryCandidate]:
        match = _REMEMBER.match(text) if isinstance(text, str) else None
        if not match:
            return None
        content = match.group(1).strip()
        if not content or len(content) > 300 or _SECRET.search(content):
            return None
        if re.search(r"(?:轻生|自杀|不想活|伤害自己|活不下去|危机)", content):
            category = "crisis"
        elif re.search(r"(?:账号|账户|银行卡|身份证|支付账户|社保号)", content):
            category = "account"
        elif re.search(r"(?:电话|手机|邮箱|微信|联系方式)", content):
            category = "contact"
        elif re.search(r"(?:生病|失眠|诊断|药物|病史|高血压|抑郁|焦虑症)", content):
            category = "health"
        elif re.search(r"(?:家里人|家人|父母|爸爸|妈妈|兄弟|姐妹|婆媳).*(?:吵架|冲突|冷战|矛盾)", content):
            category = "family_conflict"
        elif re.search(r"(?:男朋友|女朋友|伴侣|丈夫|妻子|老公|老婆|对象|恋人).*(?:吵架|冲突|冷战|分手|矛盾)", content):
            category = "relationship_conflict"
        elif re.search(r"(?:住在|地址|(?:路|街|巷|弄)\s*\d*\s*号)", content):
            category = "address"
        else:
            category = "preference"
        predicate, scope, expiry = (None, "default", None)
        if category == "preference":
            predicate, scope, expiry = self._preference_scope(content, text)
        return classify_candidate(category, content, "explicit",
                                  predicate=predicate, scene=scope, valid_to=expiry,
                                  source_message_id=message_id,
                                  importance=0.8,
                                  user_id=self.user_id,
                                  turn_text=text)

    def _preference_scope(self, content, turn_text, predicate=None, scene="default"):
        now = self.storage.clock()
        if driving_audio(turn_text):
            if scene == "default":
                scene = scene_for(content, now)
                if scene == "default":
                    scene = scene_for(turn_text, now)
            if any(word in content for word in ("听", "音乐", "播客", "安静", "播放", "电台")):
                predicate = "驾车音频偏好"
        expiry = midnight_after(now).isoformat() if temporary_intent(turn_text) else None
        return predicate, scene, expiry

    def _validate_candidates(self, result, message_id: Optional[int] = None,
                             turn_text: str = "") -> List[MemoryCandidate]:
        if not isinstance(result, dict) or not isinstance(result.get("memories"), list):
            return []
        candidates = []
        for entry in result["memories"]:
            if len(candidates) == 5:
                break
            if not isinstance(entry, dict):
                continue
            category = entry.get("category")
            content = entry.get("content")
            sensitivity = entry.get("sensitivity")
            if not isinstance(category, str) or category.strip().lower() not in ALLOWED_CATEGORIES:
                continue
            if not isinstance(content, str) or not content.strip() or len(content.strip()) > 300:
                continue
            if sensitivity not in (Sensitivity.LOW.value, Sensitivity.SENSITIVE.value):
                continue
            if _SECRET.search(content):
                continue
            predicate = entry.get("predicate")
            if not isinstance(predicate, str) or not predicate.strip() or len(predicate.strip()) > 40:
                predicate = None
            if predicate and _CHANGE_PREDICATE.search(predicate):
                # "饮品习惯变化" describes history, not a durable property; only
                # the corrected fact belongs in the memory table.
                continue
            value = entry.get("value")
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
                value = content.strip()
            importance = entry.get("importance")
            try:
                importance = float(importance)
            except (TypeError, ValueError):
                importance = 0.5
            subject = entry.get("subject")
            if not isinstance(subject, str) or not subject.strip() or len(subject.strip()) > 40:
                subject = "用户"
            scope = entry.get("scene")
            scope = scope if isinstance(scope, str) and scope in SCENES else "default"
            expiry = None
            if category.strip().lower() == "preference":
                predicate, scope, expiry = self._preference_scope(content, turn_text, predicate, scope)
                if entry.get("temporary") is True and expiry is None:
                    continue
            else:
                scope = "default"
            candidate = classify_candidate(
                category, content, "auto", predicate=predicate, value=value.strip(),
                subject=subject.strip(), importance=importance,
                source_message_id=message_id, user_id=self.user_id, turn_text=turn_text,
                scene=scope, valid_to=expiry)
            if sensitivity == Sensitivity.SENSITIVE.value:
                candidate = replace(candidate, sensitivity=Sensitivity.SENSITIVE)
            candidates.append(candidate)
        return candidates

    def accept(self, candidate: MemoryCandidate, turn_text: str = "") -> int:
        """Apply the deterministic decision, then write through SQLite."""
        checked = self._checked_candidate(candidate, turn_text)
        with self.storage.transaction(immediate=True):
            memory_id, decision = self._accept_checked(checked)
        self.last_decision = decision
        return memory_id

    def _checked_candidate(self, candidate, turn_text=""):
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError("candidate must be a MemoryCandidate")
        checked = classify_candidate(
            candidate.category, candidate.content, candidate.source,
            predicate=candidate.predicate, value=candidate.value,
            subject=candidate.subject, importance=candidate.importance,
            source_message_id=candidate.source_message_id,
            user_id=candidate.user_id or self.user_id,
            turn_text=candidate.turn_text or turn_text,
            scene=candidate.scene, valid_to=candidate.valid_to)
        if not checked.content or len(checked.content) > 300 or checked.category not in ALLOWED_CATEGORIES:
            raise ValueError("invalid memory candidate")
        if _SECRET.search(checked.content):
            raise ValueError("secrets cannot be remembered")
        if checked.scene not in SCENES:
            raise ValueError("unsupported memory scene")
        if checked.valid_to is not None and timestamp(checked.valid_to) <= timestamp(self.storage.clock()):
            raise ValueError("temporary preference must end in the future")
        if candidate.sensitivity == Sensitivity.SENSITIVE:
            checked = replace(checked, sensitivity=Sensitivity.SENSITIVE)
        return checked

    def _accept_checked(self, checked, confirmed=False):
        existing = self._existing_facts_for(checked, checked.turn_text)
        similarity = max(
            (lexical_similarity(checked.value, row.value or row.content) for row in existing),
            default=None)
        decision = facts.decide(checked, existing, checked.turn_text, similarity)
        if confirmed:
            decision = facts.Decision(facts.UPDATE, "confirmed correction",
                                      supersede=tuple(row.id for row in existing))
        if decision.action == facts.NOOP:
            return decision.memory_id or 0, decision
        memory_id = self.storage.save_memory(checked, commit=False)
        if decision.action == facts.UPDATE:
            replacement = self.storage.memory_by_id(memory_id)
            for old_id in decision.supersede:
                if old_id != memory_id:
                    self.storage.supersede_memory(
                        old_id, memory_id, when=replacement.valid_from, action="updated", commit=False)
        return memory_id, decision

    def _existing_facts_for(self, candidate, turn_text: str):
        """Match an explicit attribute, allowing habit/preference suffix drift."""
        if not candidate.predicate:
            return []
        predicate = facts.normalize(candidate.predicate)
        stem = re.sub(r"(?:偏好|习惯)$", "", predicate)
        correction = facts.correction_intent(turn_text)
        return [
            row for row in self.storage.active_facts(candidate.user_id, candidate.subject)
            if row.scene == candidate.scene and row.valid_to == candidate.valid_to
            and (facts.normalize(row.predicate) == predicate or (
                correction and stem and row.category == candidate.category
                and re.sub(r"(?:偏好|习惯)$", "", facts.normalize(row.predicate)) == stem
            ))
        ]

    def record_feedback(self, trace, memory_id, replacement=None):
        """Record one turn's feedback, and apply an explicitly confirmed correction."""
        recall_id, scene = trace.get("recall_id"), trace.get("scene", "default")
        if (not isinstance(recall_id, str) or not recall_id or scene not in SCENES
                or type(memory_id) is not int or memory_id not in trace.get("selected", ())):
            raise ValueError("feedback must reference a selected memory")
        with self.storage.transaction(immediate=True):
            row = self.storage.memory_by_id(memory_id)
            if row is None or row.user_id != self.user_id:
                raise ValueError("feedback must reference a memory owned by the current user")
            if replacement is None:
                return self.storage.record_memory_feedback(self.user_id, memory_id, recall_id, scene)
            previous = self.storage.memory_feedback(self.user_id, memory_id, recall_id, "confirmed_correction")
            if previous is not None:
                return previous
            if not row.active or row.temporary or not row.predicate:
                raise ValueError("confirmed correction requires an active, identified long-term fact")
            checked = self._checked_candidate(replace(
                replacement, user_id=row.user_id, subject=row.subject,
                category=row.category, predicate=row.predicate, scene=scene,
                valid_to=None, source="confirmed_feedback",
                sensitivity=Sensitivity(row.sensitivity)))
            replacement_id, _ = self._accept_checked(checked, confirmed=True)
            return self.storage.record_memory_feedback(
                self.user_id, memory_id, recall_id, scene, replacement_id=replacement_id)

    def reject(self, candidate: MemoryCandidate) -> None:
        # Candidates are transient until accepted, so rejection deliberately writes nothing.
        return None

    def summarize_expired_messages(self, retention_days=None,
                                   now: Optional[str] = None) -> int:
        retention_days = (self.settings.retention_days if retention_days is None
                          else int(retention_days))
        if retention_days < 0:
            raise ValueError("retention_days must not be negative")
        current = datetime.fromisoformat(now) if now else datetime.now(timezone.utc).astimezone()
        cutoff = (current - timedelta(days=retention_days)).isoformat()
        with self.storage.locked() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.session_id, m.role, m.content
                FROM messages m
                WHERE datetime(m.created_at) < datetime(?)
                  AND NOT EXISTS (
                    SELECT 1 FROM summaries s
                    WHERE s.session_id = m.session_id
                      AND m.id BETWEEN s.first_message_id AND s.last_message_id
                  )
                ORDER BY m.session_id, m.id
                """,
                (cutoff,),
            ).fetchall()
        sessions = {}
        for row in rows:
            sessions.setdefault(row["session_id"], []).append(row)
        removed = 0
        for session_id, messages in sessions.items():
            groups = []
            for message in messages:
                if not groups or message["id"] != groups[-1][-1]["id"] + 1:
                    groups.append([])
                groups[-1].append(message)
            for group in groups:
                removed += self._summarize_group(session_id, group)
        return removed

    def _summarize_group(self, session_id, messages) -> int:
            transcript = "\n".join("{}: {}".format(r["role"], r["content"]) for r in messages)
            try:
                result = self.kimi.complete_json([
                    {"role": "system", "content": "Return concise factual JSON {\"summary\":\"...\"}."},
                    {"role": "user", "content": transcript},
                ])
                summary = result.get("summary") if isinstance(result, dict) else None
                if not isinstance(summary, str) or not summary.strip():
                    return 0
            except Exception:
                return 0
            first_id, last_id = messages[0]["id"], messages[-1]["id"]
            with self.storage.transaction() as connection:
                connection.execute(
                    """INSERT INTO summaries(session_id, content, first_message_id,
                                               last_message_id, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, summary.strip(), first_id, last_id, self.storage._now()),
                )
                deleted = self.storage._delete_messages_by_ids([row["id"] for row in messages])
                return deleted
