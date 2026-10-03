from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated, low-threshold configuration for deterministic tests."""

    config = load_config(tmp_path)
    config.compact_threshold_tokens = 40
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("dungct", "# User profile\n\n- name: DũngCT\n")

    assert path.name == "User.md"
    assert "DũngCT" in store.read_text("dungct")
    assert store.edit_text("dungct", "DũngCT", "DũngCT Updated")
    assert "Updated" in store.read_text("dungct")
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    for index in range(8):
        agent.reply("dungct", "long-thread", f"Turn {index}: " + "ngữ cảnh dài " * 20)

    assert agent.compaction_count("long-thread") > 0
    context = agent.compact_memory.context("long-thread")
    assert context["summary"]
    assert len(context["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    statement = "Mình tên là DũngCT, hiện tại đang ở Huế và làm MLOps engineer."

    baseline.reply("dungct", "old-thread", statement)
    advanced.reply("dungct", "old-thread", statement)
    baseline_answer = baseline.reply("dungct", "new-thread", "Mình tên gì và hiện tại làm nghề gì?")["response"]
    advanced_answer = advanced.reply("dungct", "new-thread", "Mình tên gì và hiện tại làm nghề gì?")["response"]

    assert "DũngCT" not in baseline_answer
    assert "MLOps engineer" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "MLOps engineer" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    for index in range(12):
        message = f"Turn {index}: " + "đây là ngữ cảnh benchmark dài cần được nén " * 25
        baseline.reply("dungct", "long-thread", message)
        advanced.reply("dungct", "long-thread", message)

    assert advanced.compaction_count("long-thread") > 0
    assert advanced.prompt_token_usage("long-thread") < baseline.prompt_token_usage("long-thread")
