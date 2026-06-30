"""
Reporter: renders a human-readable Markdown report from scored results.

Writes ``reports/report_{timestamp}.md`` after every run and returns the path.
The report shows the aggregate verdict up top, then a per-test table, then a
detailed per-metric breakdown including the evaluator's reasons -- so a reviewer
can see exactly which tests failed and why.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config
from .scorer import CaseScore, aggregate


def _timestamp() -> str:
    """UTC timestamp safe for filenames, e.g. 20260630T142501Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _verdict_emoji(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def render_markdown(case_scores: list[CaseScore], generated_at: str | None = None) -> str:
    """Build the full Markdown report as a string."""
    agg = aggregate(case_scores)
    threshold = config.AGGREGATE_THRESHOLD
    overall_pass = agg >= threshold
    generated_at = generated_at or _timestamp()

    lines: list[str] = []
    lines.append("# LLM Evaluation Report")
    lines.append("")
    lines.append(f"- **Generated (UTC):** {generated_at}")
    lines.append(f"- **Target model:** `{config.TARGET_MODEL}`")
    lines.append(f"- **Evaluator model:** `{config.EVALUATOR_MODEL}`")
    lines.append(f"- **Aggregate score:** **{agg:.4f}**")
    lines.append(f"- **Threshold:** {threshold:.2f}")
    lines.append(f"- **Result:** {'PASS' if overall_pass else 'FAIL'}")
    lines.append(f"- **Cases:** {len(case_scores)}")
    lines.append("")

    # Summary table -------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append("| Test | Failure mode | Combined | Min | Result |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in case_scores:
        lines.append(
            f"| `{c.id}` | {c.failure_mode} | {c.combined_score:.4f} | "
            f"{c.minimum_score:.2f} | {_verdict_emoji(c.passed)} |"
        )
    lines.append("")

    # Per-case detail -----------------------------------------------------
    lines.append("## Details")
    lines.append("")
    for c in case_scores:
        lines.append(f"### `{c.id}` -- {_verdict_emoji(c.passed)}")
        lines.append("")
        lines.append(f"- **Failure mode:** {c.failure_mode}")
        lines.append(f"- **Input:** {c.input}")
        lines.append(f"- **Combined score:** {c.combined_score:.4f} "
                     f"(minimum {c.minimum_score:.2f})")
        if c.error:
            lines.append(f"- **Error:** {c.error}")
        lines.append("")
        lines.append(f"> **Output:** {c.actual_output or '(empty)'}")
        lines.append("")
        if c.metrics:
            lines.append("| Metric | Score | Threshold | Success | Reason |")
            lines.append("| --- | --- | --- | --- | --- |")
            for m in c.metrics:
                reason = (m.reason or "").replace("\n", " ").replace("|", "\\|")
                if len(reason) > 200:
                    reason = reason[:197] + "..."
                lines.append(
                    f"| {m.name} | {m.score:.4f} | {m.threshold:.2f} | "
                    f"{'yes' if m.success else 'no'} | {reason} |"
                )
            lines.append("")

    # Footer with failed-test callout ------------------------------------
    failed = [c for c in case_scores if not c.passed]
    if failed:
        lines.append("## Failed tests")
        lines.append("")
        for c in failed:
            lines.append(f"- `{c.id}` ({c.failure_mode}): "
                         f"{c.combined_score:.4f} < {c.minimum_score:.2f}")
        lines.append("")

    return "\n".join(lines)


def write_report(
    case_scores: list[CaseScore],
    reports_dir: Path | None = None,
) -> Path:
    """Render and persist a timestamped Markdown report. Returns its path."""
    reports_dir = reports_dir or config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    content = render_markdown(case_scores, generated_at=ts)
    out_path = reports_dir / f"report_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return out_path


def summarize_failures(case_scores: Iterable[CaseScore]) -> str:
    """One-line-per-failure string, handy for console output and CI logs."""
    failed = [c for c in case_scores if not c.passed]
    if not failed:
        return "All test cases passed."
    parts = [
        f"{c.id} ({c.failure_mode}): {c.combined_score:.4f} < {c.minimum_score:.2f}"
        for c in failed
    ]
    return "Failed tests:\n  - " + "\n  - ".join(parts)
