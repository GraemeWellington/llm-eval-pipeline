"""
Scorer: applies DeepEval metrics to runner outputs and produces structured scores.

Metrics available (all judged by the Groq evaluator, never OpenAI):
  * AnswerRelevancyMetric -- is the answer relevant to the question?
  * FaithfulnessMetric    -- is the answer grounded in the provided context?
  * HallucinationMetric   -- does the answer contradict / fabricate vs context?

A case may apply a subset of these via the ``metrics`` field in test_cases.json
(absent => all apply). This lets a case opt out of a metric that is not the
signal it probes -- e.g. an off-topic case is graded on relevancy + hallucination
and skips faithfulness, which is noisy on very short grounded answers.

The scorer writes a machine-readable summary to ``reports/latest_scores.json``
containing per-test scores AND the aggregate score, which the CI gate reads.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config
from .runner import RunResult


@dataclass
class MetricScore:
    """A single metric's verdict for a single test case."""

    name: str
    score: float
    success: bool
    threshold: float
    reason: str = ""
    errored: bool = False  # True when the metric could not be measured at all


@dataclass
class CaseScore:
    """All metric scores for one test case, plus a combined value."""

    id: str
    failure_mode: str
    input: str
    actual_output: str
    minimum_score: float
    metrics: list[MetricScore] = field(default_factory=list)
    combined_score: float = 0.0
    passed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _build_metrics(evaluator: Any) -> dict[str, Any]:
    """Instantiate the DeepEval metrics bound to the Groq evaluator model.

    Imported lazily so importing this module never requires DeepEval to be
    installed (e.g. for --dry-run paths).
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )

    # async_mode=False serializes each metric's internal judge calls. DeepEval
    # otherwise fires them concurrently, which bursts past Groq's per-minute
    # rate limit and produces intermittent 429s mid-evaluation.
    return {
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.5, model=evaluator, include_reason=True, async_mode=False
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=0.5, model=evaluator, include_reason=True, async_mode=False
        ),
        "hallucination": HallucinationMetric(
            threshold=0.5, model=evaluator, async_mode=False
        ),
    }


def _to_llm_test_case(result: RunResult) -> Any:
    """Convert a RunResult into a DeepEval LLMTestCase."""
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=result.case.input,
        actual_output=result.actual_output,
        # FaithfulnessMetric reads retrieval_context; Hallucination reads context.
        retrieval_context=result.case.context or None,
        context=result.case.context or None,
    )


def _weighted_combined(metrics: list[MetricScore]) -> float:
    """Collapse per-metric scores into one number using configured weights.

    HallucinationMetric returns a *hallucination rate* (higher == worse), so we
    invert it before combining with the other (higher == better) metrics.

    Metrics that errored (e.g. could not be measured due to a rate limit) are
    EXCLUDED rather than counted as 0.0 -- counting an errored metric as zero
    would distort the result in either direction (a missing hallucination score
    would otherwise read as "perfect", a missing relevancy score as "failing").
    """
    w = config.METRIC_WEIGHTS
    weight_map = {
        "answer_relevancy": w.answer_relevancy,
        "faithfulness": w.faithfulness,
        "hallucination": w.hallucination,
    }
    total_weight = 0.0
    acc = 0.0
    for m in metrics:
        if m.errored:
            continue
        weight = weight_map.get(m.name, 1.0)
        value = (1.0 - m.score) if m.name == "hallucination" else m.score
        acc += weight * value
        total_weight += weight
    return round(acc / total_weight, 4) if total_weight else 0.0


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "rate_limit" in text or "429" in text


def _retry_after_seconds(exc: Exception, default: float) -> float:
    """Best-effort parse of Groq's 'try again in 5m43.8s' / '12.3s' hint."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if not match:
        return default
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60.0 + seconds


def _measure_with_retry(metric: Any, llm_tc: Any, retries: int = 4) -> None:
    """Run metric.measure, retrying on transient rate-limit (429) errors.

    Honours Groq's suggested wait when present; otherwise backs off. Caps the
    wait so a daily-quota 429 (which can suggest minutes) fails fast instead of
    stalling CI -- those are not recoverable within a run.
    """
    for attempt in range(retries + 1):
        try:
            metric.measure(llm_tc)
            return
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit(exc) or attempt == retries:
                raise
            wait = min(_retry_after_seconds(exc, default=15.0 * (attempt + 1)), 65.0)
            time.sleep(wait)


def score_results(results: list[RunResult], evaluator: Any | None = None) -> list[CaseScore]:
    """Score every runner result with the DeepEval metrics."""
    if evaluator is None:
        from .groq_llm import GroqEvaluatorLLM

        evaluator = GroqEvaluatorLLM()

    metrics = _build_metrics(evaluator)
    case_scores: list[CaseScore] = []

    for result in results:
        cs = CaseScore(
            id=result.case.id,
            failure_mode=result.case.failure_mode,
            input=result.case.input,
            actual_output=result.actual_output,
            minimum_score=result.case.minimum_score,
            error=result.error,
        )

        if result.error or not result.actual_output:
            cs.error = result.error or "empty output from target model"
            cs.combined_score = 0.0
            cs.passed = False
            case_scores.append(cs)
            continue

        # Per-case metric selection: a case may declare a subset of metrics in
        # test_cases.json (``metrics``). Absent/empty => all metrics apply. An
        # unknown metric name is a config error and fails loudly rather than
        # silently mis-scoring the suite.
        selected = result.case.metrics
        if selected:
            unknown = [n for n in selected if n not in metrics]
            if unknown:
                raise ValueError(
                    f"case {result.case.id!r} declares unknown metric(s) {unknown}; "
                    f"valid names are {sorted(metrics)}"
                )

        llm_tc = _to_llm_test_case(result)
        for name, metric in metrics.items():
            if selected and name not in selected:
                continue
            try:
                _measure_with_retry(metric, llm_tc)
                cs.metrics.append(
                    MetricScore(
                        name=name,
                        score=float(metric.score),
                        success=bool(getattr(metric, "success", metric.score >= metric.threshold)),
                        threshold=float(metric.threshold),
                        reason=str(getattr(metric, "reason", "") or ""),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - record metric failures
                cs.metrics.append(
                    MetricScore(
                        name=name,
                        score=0.0,
                        success=False,
                        threshold=0.5,
                        reason=f"metric error: {exc}",
                        errored=True,
                    )
                )

        # If every metric errored, the case has no usable signal -- mark it.
        measured = [m for m in cs.metrics if not m.errored]
        if not measured:
            cs.error = "all metrics errored (see metric reasons)"
            cs.combined_score = 0.0
            cs.passed = False
        else:
            cs.combined_score = _weighted_combined(cs.metrics)
            cs.passed = cs.combined_score >= cs.minimum_score
        case_scores.append(cs)

    return case_scores


def aggregate(case_scores: list[CaseScore]) -> float:
    """Mean combined score across all cases (0.0 when there are none)."""
    if not case_scores:
        return 0.0
    return round(sum(c.combined_score for c in case_scores) / len(case_scores), 4)


def write_scores(case_scores: list[CaseScore], path: Path | None = None) -> dict[str, Any]:
    """Persist per-test and aggregate scores to ``reports/latest_scores.json``."""
    path = path or config.LATEST_SCORES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    agg = aggregate(case_scores)
    payload = {
        "aggregate_score": agg,
        "aggregate_threshold": config.AGGREGATE_THRESHOLD,
        "passed": agg >= config.AGGREGATE_THRESHOLD,
        "target_model": config.TARGET_MODEL,
        "evaluator_model": config.EVALUATOR_MODEL,
        "num_cases": len(case_scores),
        "cases": [c.to_dict() for c in case_scores],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload
