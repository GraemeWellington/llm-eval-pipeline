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
TARGET_MODEL = os.getenv("TARGET_MODEL", "llama3-8b-8192")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "llama3-70b-8192")

# Groq credentials. The pipeline reads the key lazily so that --dry-run works
# with no key present.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Generation parameters for the target model. Low temperature keeps outputs
# deterministic enough for regression testing.
TARGET_TEMPERATURE = float(os.getenv("TARGET_TEMPERATURE", "0.0"))
TARGET_MAX_TOKENS = int(os.getenv("TARGET_MAX_TOKENS", "1024"))


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
# The aggregate-score threshold the whole suite must clear. CI fails the build
# when the mean score across all test cases drops below this value.
# Configurable via env var; default is 0.75 as required by the blueprint.
AGGREGATE_THRESHOLD = float(os.getenv("AGGREGATE_THRESHOLD", "0.75"))


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
    groq_base_url: str = GROQ_BASE_URL
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
