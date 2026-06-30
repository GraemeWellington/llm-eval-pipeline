"""
Scorer: applies DeepEval metrics to runner outputs and produces structured scores.

Metrics used (all judged by the Groq evaluator, never OpenAI):
  * AnswerRelevancyMetric -- is the answer relevant to the question?
  * FaithfulnessMetric    -- is the answer grounded in the provided context?
  * HallucinationMetric   -- does the answer contradict / fabricate vs context?

The scorer writes a machine-readable summary to ``reports/latest_scores.json``
containing per-test scores AND the aggregate score, which the CI gate reads.
"""

from __future__ import annotations

import json
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

    return {
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.5, model=evaluator, include_reason=True
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=0.5, model=evaluator, include_reason=True
        ),
        "hallucination": HallucinationMetric(threshold=0.5, model=evaluator),
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
        weight = weight_map.get(m.name, 1.0)
        value = (1.0 - m.score) if m.name == "hallucination" else m.score
        acc += weight * value
        total_weight += weight
    return round(acc / total_weight, 4) if total_weight else 0.0


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

        llm_tc = _to_llm_test_case(result)
        for name, metric in metrics.items():
            try:
                metric.measure(llm_tc)
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
                    )
                )

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
