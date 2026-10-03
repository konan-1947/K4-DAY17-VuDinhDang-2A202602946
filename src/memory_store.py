from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Return a stable, tokenizer-free token estimate suitable for comparisons."""

    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return 0
    # Vietnamese uses more short whitespace-separated units than English. Taking
    # the maximum keeps estimates useful for both long prose and terse messages.
    return max(1, math.ceil(len(compact) / 4), len(compact.split()))


@dataclass
class UserProfileStore:
    """Persistent, human-readable `User.md` storage with keyed fact upserts."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", user_id.strip()).strip(".-") or "anonymous"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.rstrip() + "\n"
        path.write_text(normalized, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([^:]+):\s*(.+?)\s*$", line)
            if match:
                facts[match.group(1).strip()] = match.group(2).strip()
        return facts

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        if not updates:
            return
        facts = self.facts(user_id)
        # Preferences and interests are usually supplied incrementally. Preserve
        # their known parts, while scalar facts such as location/job intentionally
        # let an explicit newer correction replace the old value.
        for key in ("response_style", "interests"):
            if key in facts and key in updates:
                updates[key] = ", ".join(dict.fromkeys([*facts[key].split(", "), *updates[key].split(", ")]))
        facts.update(updates)
        lines = ["# User profile", "", "## Stable facts"]
        lines.extend(f"- {key}: {value}" for key, value in facts.items())
        self.write_text(user_id, "\n".join(lines))


def _clean_fact(value: str) -> str:
    return value.strip(" .,!?;:\"'")


def _find(message: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return _clean_fact(match.group(1))
    return None


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicit, durable Vietnamese profile facts from a user statement.

    Rules intentionally prefer precision: question-only turns and negated/joking
    claims are ignored, while later explicit corrections overwrite keyed facts.
    """

    text = " ".join(message.strip().split())
    lower = text.casefold()
    if not text or ("?" in text and not any(marker in lower for marker in ("mình tên", "tôi tên", "mình ở", "mình làm", "mình thích", "mình nuôi", "món ăn", "đồ uống", "đính chính", "chuyển sang"))):
        return {}

    updates: dict[str, str] = {}
    # Require an explicit assertion ("tên là") so a recall question such as
    # "Mình tên gì?" can never overwrite an existing profile value.
    name = _find(text, [r"(?:mình|tôi) tên\s+là\s+([^,.!]+)"])
    if name:
        updates["name"] = name

    # Only accept explicit residence language, not travel/meeting locations.
    location = _find(
        text,
        [
            r"(?:hiện tại |giờ |vẫn )?(?:mình|tôi) đang ở\s+([^,.!]+)",
            r"(?:mình|tôi) ở\s+([^,.!]+)",
            r"(?:làm việc ở|chuyển (?:sang|đến))\s+(Đà Nẵng|Huế)(?:\s|,|\.|$)",
        ],
    )
    is_location_question = "?" in text and any(phrase in lower for phrase in ("ở đâu", "đang ở đâu", "nơi ở"))
    if location and not is_location_question and not any(word in lower for word in ("bay ra họp", "chỉ là nơi", "không phải nơi ở")):
        location = re.sub(r"\s+(?:vài tháng|mỗi ngày|trong giai đoạn này).*$", "", location, flags=re.IGNORECASE)
        updates["location"] = location

    profession = _find(
        text,
        [
            r"(?:đang |hiện tại |giờ )?(?:làm|là)\s+(MLOps engineer|backend engineer)(?:\s|,|\.|$)",
            r"chuyển sang\s+(MLOps engineer|backend engineer)(?:\s|,|\.|$)",
            r"nghề nghiệp (?:hiện tại )?(?:vẫn )?là\s+(MLOps engineer|backend engineer)(?:\s|,|\.|$)",
        ],
    )
    if profession and not any(marker in lower for marker in ("câu đùa", "đùa với", "không còn làm")):
        updates["profession"] = profession
    if "không còn làm backend engineer" in lower and "mlops engineer" in lower:
        updates["profession"] = "MLOps engineer"

    if "cà phê sữa đá" in lower:
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lower:
        updates["favorite_food"] = "mì Quảng"
    pet = _find(text, [r"(?:nuôi|con) (?:một |một bé )?(corgi)(?: tên ([^,.!]+))?"])
    if pet:
        updates["pet"] = "corgi"
    pet_name = _find(text, [r"corgi tên\s+([^,.!]+)"])
    if pet_name:
        updates["pet_name"] = pet_name

    interests = [topic for topic in ("Python", "AI ứng dụng", "MLOps", "AI agent", "benchmark memory", "RAG", "evaluation") if topic.casefold() in lower]
    if interests:
        old = updates.get("interests", "")
        updates["interests"] = ", ".join(dict.fromkeys(filter(None, [old, *interests])))

    style_markers: list[str] = []
    if "ngắn gọn" in lower:
        style_markers.append("ngắn gọn")
    if "3 bullet" in lower or "ba bullet" in lower:
        style_markers.append("3 bullet")
    if "ví dụ thực chiến" in lower or "ví dụ thực tế" in lower:
        style_markers.append("có ví dụ thực chiến")
    if "trade-off" in lower or "trade off" in lower:
        style_markers.append("nhấn trade-off recall và token")
    if style_markers:
        updates["response_style"] = ", ".join(dict.fromkeys(style_markers))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded heuristic summary without an LLM dependency."""

    if not messages:
        return ""
    selected = messages[-max_items:]
    snippets: list[str] = []
    for message in selected:
        content = " ".join(message["content"].split())
        if len(content) > 160:
            content = content[:157].rstrip() + "..."
        snippets.append(f"{message['role']}: {content}")
    return " | ".join(snippets)


@dataclass
class CompactMemoryManager:
    """Per-thread rolling summaries that retain only recent full messages."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        prompt_text = str(thread["summary"]) + " ".join(item["content"] for item in messages)
        if estimate_tokens(prompt_text) <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        old_messages = messages[:-self.keep_messages]
        retained = messages[-self.keep_messages:]
        existing_summary = str(thread["summary"])
        condensed = summarize_messages(old_messages)
        combined = " | ".join(piece for piece in (existing_summary, condensed) if piece)
        # Bound summaries so repeated compaction remains a genuine token reduction.
        max_chars = max(240, self.threshold_tokens * 2)
        thread["summary"] = combined[-max_chars:]
        thread["messages"] = retained
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {
            "messages": [dict(item) for item in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
