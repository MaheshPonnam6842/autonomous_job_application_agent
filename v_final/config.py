"""Centralized, environment-driven configuration for the v_final agent.

Everything tunable — model names, the Ollama host, scoring weights, decision
thresholds, and artifact paths — lives here. Nodes read from a single frozen
``settings`` object so runs are reproducible and the behavior can be changed
without touching node logic. Override any value via environment variables
(see ``.env.example``); sensible defaults keep the agent runnable out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    """How to reach the local LLM. All Ollama, all local."""

    provider: str = _env_str("JOBAGENT_LLM_PROVIDER", "ollama")
    host: str = _env_str("OLLAMA_HOST", "http://localhost:11434")
    # One model family by default; override the rewrite model if you keep a
    # higher-quality instruct quant pulled locally.
    chat_model: str = _env_str("JOBAGENT_CHAT_MODEL", "llama3.1:8b")
    rewrite_model: str = _env_str("JOBAGENT_REWRITE_MODEL", "llama3.1:8b")
    embed_model: str = _env_str("JOBAGENT_EMBED_MODEL", "nomic-embed-text")
    request_timeout: float = _env_float("JOBAGENT_LLM_TIMEOUT", 120.0)
    max_retries: int = _env_int("JOBAGENT_LLM_RETRIES", 2)
    temperature: float = _env_float("JOBAGENT_LLM_TEMPERATURE", 0.2)
    # Hard kill-switch. When false, every node uses its deterministic fallback
    # and never touches the network — useful for tests and offline demos.
    enabled: bool = _env_bool("JOBAGENT_LLM_ENABLED", True)

    # --- Performance (local CPU inference) ---
    # Keep the model resident in RAM between calls so the loop's repeated calls
    # don't pay the cold-load cost each time.
    keep_alive: str = _env_str("JOBAGENT_LLM_KEEP_ALIVE", "10m")
    # CPU threads for inference; 0 lets Ollama auto-detect all cores.
    num_thread: int = _env_int("JOBAGENT_LLM_NUM_THREAD", 0)
    # Context window — smaller is faster; large enough for a resume + JD.
    num_ctx: int = _env_int("JOBAGENT_LLM_NUM_CTX", 4096)
    # LLM tone-polish of outreach drafts. Off by default: the deterministic
    # templates are good, and two extra calls are the slowest part of a run.
    outreach_polish: bool = _env_bool("JOBAGENT_OUTREACH_POLISH", False)


@dataclass(frozen=True)
class ScoringConfig:
    """Weights for the composite match score. Normalized at load time."""

    weight_skill: float = _env_float("JOBAGENT_W_SKILL", 0.5)
    weight_semantic: float = _env_float("JOBAGENT_W_SEMANTIC", 0.3)
    weight_ats: float = _env_float("JOBAGENT_W_ATS", 0.2)

    @property
    def normalized(self) -> tuple[float, float, float]:
        total = self.weight_skill + self.weight_semantic + self.weight_ats
        if total <= 0:
            return (0.5, 0.3, 0.2)
        return (
            self.weight_skill / total,
            self.weight_semantic / total,
            self.weight_ats / total,
        )


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds that gate whether a rewrite is triggered."""

    overall_threshold: float = _env_float("JOBAGENT_REWRITE_THRESHOLD", 0.75)
    max_missing_required: int = _env_int("JOBAGENT_MAX_MISSING_REQUIRED", 1)


@dataclass(frozen=True)
class RewriteConfig:
    """Controls the iterative rewrite-and-rescore loop."""

    # How many rewrite attempts to make (each is one LLM call). The loop keeps the
    # best-scoring candidate and stops early once an attempt fails to improve it.
    # Default 2 to stay fast on CPU; raise for more polish.
    max_attempts: int = _env_int("JOBAGENT_REWRITE_MAX_ATTEMPTS", 2)
    # Which score to optimize. Tie broken by overall_match.
    target_metric: str = _env_str("JOBAGENT_REWRITE_TARGET", "ats_match")
    # Minimum improvement over the current best to count as progress.
    min_gain: float = _env_float("JOBAGENT_REWRITE_MIN_GAIN", 0.0)
    # "Good enough" short-circuit: stop retrying once the target metric hits this,
    # even if attempts remain — saves expensive LLM calls when already strong.
    target_threshold: float = _env_float("JOBAGENT_REWRITE_TARGET_THRESHOLD", 0.9)


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    scoring: ScoringConfig
    decision: DecisionConfig
    rewrite: RewriteConfig
    artifacts_dir: Path
    log_level: str


def load_settings() -> Settings:
    artifacts = Path(_env_str("JOBAGENT_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
    return Settings(
        llm=LLMConfig(),
        scoring=ScoringConfig(),
        decision=DecisionConfig(),
        rewrite=RewriteConfig(),
        artifacts_dir=artifacts,
        log_level=_env_str("JOBAGENT_LOG_LEVEL", "INFO"),
    )


# Module-level singleton: import `settings` anywhere in the agent.
settings = load_settings()
