"""
Runner: executes each test case against the target LLM (Groq/Llama3).

Responsibilities:
  * Load test cases from ``evals/test_cases.json``.
  * For each case, build a grounded prompt (context + question) and call Groq.
  * Return the raw outputs alongside the original case metadata.

A ``--dry-run`` flag prints the loaded test cases (and the prompts that would be
sent) WITHOUT making any API calls -- handy for offline inspection and CI smoke
checks where no key is available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config


@dataclass
class TestCase:
    """A single evaluation case loaded from JSON."""

    id: str
    failure_mode: str
    input: str
    context: list[str]
    expected_output_criteria: list[str]
    minimum_score: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TestCase":
        return cls(
            id=raw["id"],
            failure_mode=raw.get("failure_mode", "unspecified"),
            input=raw["input"],
            context=list(raw.get("context", []) or []),
            expected_output_criteria=list(raw.get("expected_output_criteria", [])),
            minimum_score=float(raw.get("minimum_score", config.AGGREGATE_THRESHOLD)),
        )


@dataclass
class RunResult:
    """The outcome of running one test case through the target model."""

    case: TestCase
    actual_output: str
    prompt: str = ""
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def load_test_cases(path: Path | None = None) -> list[TestCase]:
    """Read and parse all test cases from disk."""
    path = path or config.TEST_CASES_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw_cases = json.load(fh)
    return [TestCase.from_dict(rc) for rc in raw_cases]


def build_prompt(case: TestCase) -> str:
    """Compose a grounded prompt from the case context and question.

    When context is present we instruct the model to answer strictly from it --
    this is what makes the faithfulness / hallucination metrics meaningful.
    """
    if case.context:
        joined = "\n".join(f"- {c}" for c in case.context)
        return (
            "You are a helpful assistant. Answer the question using ONLY the "
            "context below. If the context does not contain the answer, say so "
            "explicitly and do not invent details.\n\n"
            f"Context:\n{joined}\n\n"
            f"Question: {case.input}\n\nAnswer:"
        )
    return f"{case.input}"


def run_single(client: Any, case: TestCase) -> RunResult:
    """Run one case against the live Groq target model."""
    prompt = build_prompt(case)
    try:
        response = client.chat.completions.create(
            model=config.TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TARGET_TEMPERATURE,
            max_tokens=config.TARGET_MAX_TOKENS,
        )
        output = response.choices[0].message.content or ""
        return RunResult(case=case, actual_output=output.strip(), prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - surface any API error per-case
        return RunResult(case=case, actual_output="", prompt=prompt, error=str(exc))


def run_all(cases: list[TestCase] | None = None) -> list[RunResult]:
    """Run every test case against the live target model.

    Imports the Groq client lazily so that ``--dry-run`` works without the SDK
    or an API key being configured.
    """
    from groq import Groq

    cases = cases if cases is not None else load_test_cases()
    api_key = config.require_api_key()
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "max_retries": config.GROQ_MAX_RETRIES,
    }
    # Only override base_url for a proxy/gateway; otherwise use the SDK default.
    if config.GROQ_BASE_URL:
        client_kwargs["base_url"] = config.GROQ_BASE_URL
    client = Groq(**client_kwargs)

    return [run_single(client, case) for case in cases]


def _dry_run(cases: list[TestCase]) -> None:
    """Print test cases and the prompts that would be sent -- no API calls."""
    print(f"DRY RUN: {len(cases)} test case(s) loaded from {config.TEST_CASES_PATH}")
    print(f"Target model (not called): {config.TARGET_MODEL}\n")
    for i, case in enumerate(cases, start=1):
        print(f"[{i}] {case.id}  (failure_mode: {case.failure_mode})")
        print(f"    input          : {case.input}")
        print(f"    context        : {case.context if case.context else '(none)'}")
        print(f"    criteria       : {case.expected_output_criteria}")
        print(f"    minimum_score  : {case.minimum_score}")
        print("    prompt preview :")
        for line in build_prompt(case).splitlines():
            print(f"        {line}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM eval test cases.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print test cases and prompts without calling the Groq API.",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=config.TEST_CASES_PATH,
        help="Path to the test_cases.json file.",
    )
    args = parser.parse_args(argv)

    cases = load_test_cases(args.test_cases)

    if args.dry_run:
        _dry_run(cases)
        return 0

    results = run_all(cases)
    for res in results:
        status = "ERROR" if res.error else "OK"
        print(f"[{status}] {res.case.id}: {res.actual_output[:120]!r}")
        if res.error:
            print(f"        -> {res.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
