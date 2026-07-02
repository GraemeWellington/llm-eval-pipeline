"""
Central configuration for the LLM evaluation pipeline.

All tunable knobs live here so that test files, the runner, the scorer and the
CI workflow can share a single source of truth. Nothing in this project should
hardcode a threshold or model name -- import it from this module instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------
# CI passes optional overrides like ``${{ vars.AGGREGATE_THRESHOLD }}``. When the
# repository variable is unset, the value arrives as an EMPTY STRING rather than
# being absent -- so os.getenv(name, default) returns "" and float("") explodes.
# These helpers treat empty/whitespace values as "unset" and fall back cleanly.
def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# Resolve everything relative to this file so the pipeline works regardless of
# the current working directory (pytest, CI, and ad-hoc runs all differ).
ROOT_DIR = Path(__file__).resolve().parent
EVALS_DIR = ROOT_DIR / "evals"
REPORTS_DIR = ROOT_DIR / "reports"
TEST_CASES_PATH = EVALS_DIR / "test_cases.json"

# Machine-readable score dump consumed by the reporter and the CI gate.
LATEST_SCORES_PATH = REPORTS_DIR / "latest_scores.json"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# The "target" model is the system under test -- the model whose answers we are
# grading. The "evaluator" model is the judge used by DeepEval's LLM-based
# metrics. Per project constraints, BOTH are served by Groq (no OpenAI).
# Groq decommissioned the original llama3-*-8192 IDs (shutdown 2025-08-30);
# these are the current production Llama replacements.
TARGET_MODEL = _env_str("TARGET_MODEL", "llama-3.1-8b-instant")
# The judge defaults to the 70B model for high-fidelity grading (it scores the
# suite cleanly, ~0.90 aggregate). Its Groq free-tier daily budget is 100k TPD
# (~5 full runs/day); once drained, judge calls 429 until it resets. On a
# constrained free tier, fall back to the higher-throughput but noisier 8B
# judge: EVALUATOR_MODEL=llama-3.1-8b-instant.
EVALUATOR_MODEL = _env_str("EVALUATOR_MODEL", "llama-3.3-70b-versatile")

# Groq credentials. The pipeline reads the key lazily so that --dry-run works
# with no key present. An empty string counts as "not set".
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or None
# Leave unset by default: the Groq SDK already routes to /openai/v1 internally,
# so hardcoding that suffix here would double it (/openai/v1/openai/v1/...).
# Only set GROQ_BASE_URL to point at a proxy/self-hosted gateway.
GROQ_BASE_URL = _env_str("GROQ_BASE_URL", "") or None

# Generation parameters for the target model. Low temperature keeps outputs
# deterministic enough for regression testing.
TARGET_TEMPERATURE = _env_float("TARGET_TEMPERATURE", 0.0)
TARGET_MAX_TOKENS = _env_int("TARGET_MAX_TOKENS", 1024)

# How many times the Groq SDK retries transient HTTP failures (esp. 429 rate
# limits). The SDK backs off exponentially and honours Retry-After headers.
#
# This is deliberately small. There are TWO retry layers: the Groq SDK (here)
# and instructor's own retry loop (GROQ_STRUCTURED_MAX_RETRIES) that wraps the
# structured judge calls. They MULTIPLY -- SDK=6 x instructor=6 meant up to ~36
# HTTP attempts per metric, so a run against an *exhausted* daily quota (every
# call 429s) ground on for ~2 hours before failing instead of failing fast.
# Keep the product small so a drained-quota run errors in minutes, not hours,
# while still absorbing the brief 429s a healthy run hits under Groq's TPM cap.
GROQ_MAX_RETRIES = _env_int("GROQ_MAX_RETRIES", 3)

# Retries for instructor's structured-output loop (schema-validation failures).
# Kept separate and low so it does not multiply the SDK's rate-limit backoff.
GROQ_STRUCTURED_MAX_RETRIES = _env_int("GROQ_STRUCTURED_MAX_RETRIES", 2)

# Per-request wall-clock timeout (seconds) for Groq calls, so a single stalled
# request can't hang the whole suite regardless of the retry counts above.
GROQ_TIMEOUT_SECONDS = _env_float("GROQ_TIMEOUT_SECONDS", 60.0)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
# The aggregate-score threshold the whole suite must clear. CI fails the build
# when the mean score across all test cases drops below this value.
# Configurable via env var; default is 0.75 as required by the blueprint.
AGGREGATE_THRESHOLD = _env_float("AGGREGATE_THRESHOLD", 0.75)


@dataclass(frozen=True)
class MetricWeights:
    """Relative weights used when collapsing per-metric scores into one number.

    Weights are normalised at use time, so they need not sum to 1.0 here.
    """

    answer_relevancy: float = 1.0
    faithfulness: float = 1.0
    hallucination: float = 1.0


METRIC_WEIGHTS = MetricWeights()


@dataclass(frozen=True)
class Settings:
    """Convenience bundle of everything the pipeline needs at runtime."""

    target_model: str = TARGET_MODEL
    evaluator_model: str = EVALUATOR_MODEL
    groq_api_key: str | None = GROQ_API_KEY
    groq_base_url: str | None = GROQ_BASE_URL
    target_temperature: float = TARGET_TEMPERATURE
    target_max_tokens: int = TARGET_MAX_TOKENS
    aggregate_threshold: float = AGGREGATE_THRESHOLD
    test_cases_path: Path = TEST_CASES_PATH
    reports_dir: Path = REPORTS_DIR
    latest_scores_path: Path = LATEST_SCORES_PATH
    weights: MetricWeights = field(default_factory=lambda: METRIC_WEIGHTS)


def get_settings() -> Settings:
    """Return a fresh Settings snapshot (re-reads env-derived module globals)."""
    return Settings()


def require_api_key() -> str:
    """Return the Groq API key or raise a helpful error if it is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it before running live evaluations, "
            "or use the --dry-run flag to preview test cases without API calls."
        )
    return GROQ_API_KEY
