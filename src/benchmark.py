from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load and validate a benchmark JSON array."""

    with path.open(encoding="utf-8") as file:
        conversations = json.load(file)
    if not isinstance(conversations, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    """Score recall as 0, 0.5, or 1 based on expected fact coverage."""

    if not expected:
        return 1.0
    normalized = answer.casefold()
    matched = sum(item.casefold() in normalized for item in expected)
    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight deterministic response-quality proxy for offline comparison."""

    recall = recall_points(answer, expected)
    structured_bonus = 0.1 if "- " in answer else 0.0
    concise_bonus = 0.1 if 0 < len(answer) <= 500 else 0.0
    return min(1.0, round(recall * 0.8 + structured_bonus + concise_bonus, 2))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Run turns in their source threads, then test recall in fresh threads."""

    source_threads: list[str] = []
    recall_answers: list[tuple[str, list[str]]] = []
    memory_before: dict[str, int] = {}

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        source_threads.append(thread_id)
        if hasattr(agent, "memory_file_size"):
            memory_before.setdefault(user_id, agent.memory_file_size(user_id))
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, turn)
        for index, question in enumerate(conversation.get("recall_questions", [])):
            fresh_thread = f"{thread_id}-recall-{index}"
            answer = agent.reply(user_id, fresh_thread, question["question"])["response"]
            recall_answers.append((answer, question["expected_contains"]))

    all_threads = [*source_threads, *(f"{item['id']}-recall-{index}" for item in conversations for index, _ in enumerate(item.get("recall_questions", [])))]
    agent_tokens = sum(agent.token_usage(thread_id) for thread_id in all_threads)
    prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in all_threads)
    compactions = sum(agent.compaction_count(thread_id) for thread_id in source_threads)
    recall = sum(recall_points(answer, expected) for answer, expected in recall_answers) / max(1, len(recall_answers))
    quality = sum(heuristic_quality(answer, expected) for answer, expected in recall_answers) / max(1, len(recall_answers))
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(user) - initial for user, initial in memory_before.items())

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(recall, 2),
        response_quality=round(quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format required metrics as a dependency-free Markdown table."""

    headers = [
        "Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
        "Response quality", "Memory growth (bytes)", "Compactions",
    ]
    values = [
        [
            row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}", f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes), str(row.compactions),
        ]
        for row in rows
    ]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in values),
    ])


def _run_suite(title: str, conversations: list[dict[str, Any]], config) -> None:
    # A fresh state directory makes memory-growth metrics reproducible across
    # repeated benchmark runs and prevents the standard suite leaking into stress.
    with tempfile.TemporaryDirectory(prefix="memory-benchmark-") as directory:
        suite_config = replace(config, state_dir=Path(directory))
        baseline = BaselineAgent(config=suite_config, force_offline=True)
        advanced = AdvancedAgent(config=suite_config, force_offline=True)
        rows = [
            run_agent_benchmark("Baseline", baseline, conversations, suite_config),
            run_agent_benchmark("Advanced", advanced, conversations, suite_config),
        ]
    print(f"\n## {title}")
    print(format_rows(rows))


def main() -> None:
    """Run the standard and long-context offline benchmark suites."""

    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    _run_suite("Standard Benchmark", standard, config)
    _run_suite("Long-Context Stress Benchmark", stress, config)


if __name__ == "__main__":
    main()
