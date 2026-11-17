"""Supported preference scopes and their local-time boundaries."""

import re
from datetime import datetime, time, timedelta, timezone


SCENES = frozenset({"default", "daytime", "evening_return"})
_DRIVE = re.compile(r"开车|驾车|行车|通勤|车里|车上|回家|返程|回去|下班|路上")
_AUDIO = re.compile(r"听点|听些|听什|音乐|播客|电台|播放|放歌|放点|安静|音频|歌单|节目")
_DAY = re.compile(r"白天|日间|上午|中午|下午|早上|早晨|早班")
_EVENING = re.compile(r"今晚|当晚|夜间|晚上|晚间|下班|夜班")
_TEMPORARY = re.compile(r"今天|今晚|当晚|这次|这一趟|这趟")
_LONG_TERM = re.compile(r"以后|今后|每次|从今往后|一直")


def scene_for(text, now, explicit=None):
    if explicit is not None:
        if explicit not in SCENES:
            raise ValueError("unsupported memory scene")
        return explicit
    if _DAY.search(text):
        return "daytime"
    if _EVENING.search(text):
        return "evening_return"
    if _DRIVE.search(text):
        return "evening_return" if now.hour >= 17 or now.hour < 5 else "daytime"
    return "default"


def driving_audio(text):
    return bool(_DRIVE.search(text) and _AUDIO.search(text))


def temporary_intent(text):
    return bool(_TEMPORARY.search(text) and not _LONG_TERM.search(text))


def midnight_after(now):
    return datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=now.tzinfo)


def timestamp(value):
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def validity_reason(row, now):
    if row.status not in ("active", "superseded"):
        return "superseded"
    try:
        if row.valid_from and timestamp(row.valid_from) > timestamp(now):
            return "not_yet_valid"
        if row.valid_to and timestamp(row.valid_to) <= timestamp(now):
            return "superseded" if row.status == "superseded" else "expired"
    except ValueError:
        return "expired"
    return None
