from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: deterministic short-term memory scoped strictly to one thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a response and cumulative accounting for this thread."""

        # A real model is deliberately optional. Offline fallback preserves a
        # repeatable benchmark even with live credentials configured incorrectly.
        if self.langchain_agent is not None:
            try:
                response = self.langchain_agent.invoke(message)
                text = getattr(response, "content", str(response))
                state = self.sessions.setdefault(thread_id, SessionState())
                state.messages.extend(({"role": "user", "content": message}, {"role": "assistant", "content": text}))
                state.prompt_tokens_processed += sum(estimate_tokens(item["content"]) for item in state.messages)
                state.token_usage += estimate_tokens(text)
                return {"response": text, "token_usage": state.token_usage, "prompt_tokens_processed": state.prompt_tokens_processed}
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        lower = message.casefold()
        if any(phrase in lower for phrase in ("mình tên gì", "nhắc lại", "bạn có biết", "hiện tại mình")) and len(state.messages) == 1:
            response = "Mình không có thông tin từ các thread trước, nên chưa thể nhắc lại chính xác."
        else:
            response = "Đã nhận trong thread hiện tại. Baseline chỉ dùng ngữ cảnh của thread này."
        state.messages.append({"role": "assistant", "content": response})
        state.prompt_tokens_processed += sum(estimate_tokens(item["content"]) for item in state.messages)
        state.token_usage += estimate_tokens(response)
        return {
            "response": response,
            "token_usage": state.token_usage,
            "prompt_tokens_processed": state.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Build a raw chat model when optional live dependencies are available."""

        try:
            return build_chat_model(self.config.model)
        except (ImportError, RuntimeError, ValueError):
            return None
