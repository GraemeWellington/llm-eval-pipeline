"""
pytest entrypoint for the LLM evaluation suite.

Design notes:
  * Thresholds are NEVER hardcoded here -- they come from ``config.py`` and from
    each test case's ``minimum_score`` in ``test_cases.json``.
  * Each test case becomes its OWN parametrized pytest test, so the report shows
    precisely which case failed and why (not just a pass/fail count).
  * A session-scoped fixture runs the whole pipeline once (run -> score ->
    persist scores -> write Markdown report), then individual tests assert on
    the cached results. This avoids re-calling the API per assertion.
  * If ``GROQ_API_KEY`` is absent, the suite is skipped (so ``--dry-run`` style
    offline use and CI without secrets degrade gracefully rather than erroring).

Run:
    pytest -v
    pytest -v -k tc-02            # a single case
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make config + the pipeline package importable no matter where pytest is
# invoked from. The project root (which holds config.py and pipeline/) is the
# parent of this evals/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from pipeline import reporter, runner, scorer  # noqa: E402


requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live evaluation suite.",
)


@pytest.fixture(scope="session")
def pipeline_results():
    """Run the full pipeline once per test session and cache the scored cases.

    Returns a tuple of (case_scores, aggregate_score, report_path).
    """
    cases = runner.load_test_cases()
    run_results = runner.run_all(cases)
    case_scores = scorer.score_results(run_results)

    # Persist machine-readable scores and a Markdown report for this run.
    scorer.write_scores(case_scores)
    report_path = reporter.write_report(case_scores)

    agg = scorer.aggregate(case_scores)
    print(f"\nAggregate score: {agg:.4f} (threshold {config.AGGREGATE_THRESHOLD})")
    print(f"Report: {report_path}")
    print(reporter.summarize_failures(case_scores))

    return case_scores, agg, report_path


def _case_ids() -> list[str]:
    return [c.id for c in runner.load_test_cases()]


@requires_api_key
@pytest.mark.parametrize("case_id", _case_ids())
def test_case_meets_minimum_score(case_id, pipeline_results):
    """Each test case must meet its own ``minimum_score`` from test_cases.json."""
    case_scores, _agg, _ = pipeline_results
    result = next((c for c in case_scores if c.id == case_id), None)
    assert result is not None, f"No scored result for case {case_id}"

    if result.error:
        pytest.fail(f"[{case_id}] runtime error: {result.error}")

    # Per-metric breakdown in the failure message so it's clear WHY it failed.
    breakdown = "; ".join(
        f"{m.name}={m.score:.3f}(reason: {m.reason[:80]})" for m in result.metrics
    )
    assert result.combined_score >= result.minimum_score, (
        f"[{case_id}] failure_mode={result.failure_mode} "
        f"combined={result.combined_score:.4f} < minimum={result.minimum_score:.2f}\n"
        f"  output: {result.actual_output[:160]!r}\n"
        f"  metrics: {breakdown}"
    )


@requires_api_key
def test_aggregate_meets_threshold(pipeline_results):
    """The mean score across all cases must clear the configured CI threshold.

    This is the gate the GitHub Actions workflow relies on to fail a PR.
    """
    case_scores, agg, _ = pipeline_results
    assert agg >= config.AGGREGATE_THRESHOLD, (
        f"Aggregate score {agg:.4f} is below threshold "
        f"{config.AGGREGATE_THRESHOLD:.2f}.\n"
        f"{reporter.summarize_failures(case_scores)}"
    )
