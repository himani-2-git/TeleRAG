"""Health checks for the /status command."""

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class HealthCheck:
    """Result of checking one part of the bot."""

    name: str
    ok: bool
    detail: str


def _run_check(name: str, check: Callable[[], tuple[bool, str]]) -> HealthCheck:
    """Run one check and time it. An exception becomes a failed check, never a crash.

    Only the exception type is reported, not its message: messages can contain file paths.
    """
    started = time.perf_counter()
    try:
        ok, detail = check()
    except Exception as e:
        return HealthCheck(name, False, f"failed ({type(e).__name__})")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return HealthCheck(name, ok, f"{detail} ({elapsed_ms} ms)")


def run_health_checks(document_store, embedding_generator, llm_client) -> list[HealthCheck]:
    """Check the database, the embedding model and the LLM server."""

    def check_database() -> tuple[bool, str]:
        return bool(document_store.db.ping()), "SQLite answers"

    def check_embeddings() -> tuple[bool, str]:
        vector = embedding_generator.generate_query_embedding("health check")
        return len(vector) > 0, f"model works, vectors have {len(vector)} numbers"

    def check_llm() -> tuple[bool, str]:
        if llm_client is None:
            return False, "no LLM client configured"
        return llm_client.check_server()

    return [
        _run_check("Database", check_database),
        _run_check("Embeddings", check_embeddings),
        _run_check("LLM", check_llm),
    ]


def format_duration(seconds: float) -> str:
    """Format seconds as for example '2 h 5 min' or '45 s'."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def format_health_report(
    checks: list[HealthCheck], uptime_seconds: float, active_requests: int
) -> str:
    """Build the text message the admin sees."""
    lines = ["🩺 Bot status", ""]
    for check in checks:
        icon = "✅" if check.ok else "❌"
        lines.append(f"{icon} {check.name}: {check.detail}")
    lines.append("")
    lines.append(f"Uptime: {format_duration(uptime_seconds)}")
    lines.append(f"Questions in progress or waiting: {active_requests}")
    return "\n".join(lines)
