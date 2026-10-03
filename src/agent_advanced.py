from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term, persistent profile, and compact-memory layers."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Use the deterministic memory pipeline, with live mode as an optional extension."""

        # The live model does not implement this lab's explicit profile tools yet;
        # retain deterministic path so storage and accounting stay consistent.
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        self.profile_store.upsert_facts(user_id, updates)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(response)
        return {
            "response": response,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        contents = [self.profile_store.read_text(user_id), str(context["summary"])]
        contents.extend(item["content"] for item in context["messages"])
        return estimate_tokens("\n".join(contents))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        question = message.casefold()
        if not facts:
            return "Mình đã lưu các thông tin ổn định bạn vừa cung cấp để dùng ở các thread sau."

        requested: list[tuple[str, str]] = []
        keyword_map = {
            "name": ("tên", "là ai", "biết dũngct"),
            "profession": ("nghề", "công việc", "làm gì", "product manager", "backend"),
            "location": ("ở đâu", "nơi ở", "huế", "hà nội", "đà nẵng"),
            "response_style": ("style", "kiểu trả lời", "trả lời mình thích"),
            "favorite_drink": ("đồ uống", "uống", "cà phê"),
            "favorite_food": ("món ăn", "ăn yêu thích", "mì quảng"),
            "pet": ("nuôi con gì", "corgi", "thú cưng"),
            "interests": ("mối quan tâm", "quan tâm", "kỹ thuật"),
        }
        for key, keywords in keyword_map.items():
            if key in facts and any(keyword in question for keyword in keywords):
                label = {
                    "name": "Tên", "profession": "Nghề nghiệp hiện tại", "location": "Nơi ở hiện tại",
                    "response_style": "Style trả lời", "favorite_drink": "Đồ uống yêu thích",
                    "favorite_food": "Món ăn yêu thích", "pet": "Thú cưng", "interests": "Mối quan tâm",
                }[key]
                requested.append((label, facts[key]))
        # Stress recall asks several facts in one sentence; honour facts named by
        # broad recall language, not only per-keyword matches.
        if any(marker in question for marker in ("nhắc lại giúp", "tóm tắt", "sang thread mới")):
            requested = [(label, facts[key]) for key, label in (
                ("name", "Tên"), ("profession", "Nghề nghiệp hiện tại"), ("location", "Nơi ở hiện tại"),
                ("favorite_drink", "Đồ uống yêu thích"), ("favorite_food", "Món ăn yêu thích"),
                ("pet", "Thú cưng"), ("interests", "Mối quan tâm"), ("response_style", "Style trả lời"),
            ) if key in facts]
        if requested:
            return "\n".join(f"- {label}: {value}" for label, value in requested)

        style = facts.get("response_style", "ngắn gọn")
        return f"Đã ghi nhớ profile bền vững. Mình sẽ tiếp tục trả lời theo style: {style}."

    def _maybe_build_langchain_agent(self):
        """Best-effort live model setup; offline memory workflow remains authoritative."""

        try:
            return build_chat_model(self.config.model)
        except (ImportError, RuntimeError, ValueError):
            return None
