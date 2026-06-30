# LLM Evaluation Pipeline with Automated Regression Testing

A production-grade pipeline that grades a target LLM's answers against a suite of
test cases, scores them with [DeepEval](https://github.com/confident-ai/deepeval)
metrics, writes machine- and human-readable reports, and **fails CI when quality
regresses** below a configurable threshold.

Everything runs on **Groq (Llama 3)** — both the model under test *and* the
evaluator/judge. No OpenAI required.

---

## What it does

1. **Defines test cases** (`evals/test_cases.json`) with inputs, optional RAG
   context, expected-behavior criteria, and a per-case `minimum_score`.
2. **Runs** each case through the target model (`pipeline/runner.py`).
3. **Scores** each output with three DeepEval metrics — Answer Relevancy,
   Faithfulness, Hallucination (`pipeline/scorer.py`).
4. **Reports** results to `reports/latest_scores.json` (machine-readable) and a
   timestamped `reports/report_{timestamp}.md` (human-readable).
5. **Gates CI** — the GitHub Actions workflow fails the PR if the aggregate
   score drops below the threshold in `config.py` (default **0.75**).

---

## Project layout

```
llm_eval_pipeline/
├── evals/
│   ├── test_cases.json        # 10 cases across 5 failure modes
│   └── test_llm_quality.py    # pytest suite (one test per case + aggregate gate)
├── pipeline/
│   ├── runner.py              # runs the target LLM; has --dry-run
│   ├── scorer.py              # applies DeepEval metrics -> structured scores
│   ├── reporter.py            # renders the Markdown report
│   └── groq_llm.py            # Groq-backed DeepEval evaluator (no OpenAI)
├── config.py                  # thresholds, model + env configuration
├── .github/workflows/eval_ci.yml
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd llm_eval_pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."                        # Windows: $env:GROQ_API_KEY="gsk_..."
```

Get a free key at <https://console.groq.com>.

---

## Usage

### Preview test cases without calling the API

```bash
python -m pipeline.runner --dry-run
```

Prints every test case and the exact prompt that would be sent — no key needed.

### Run the full evaluation suite

```bash
pytest -v
```

`pytest -v` shows **each** test case (named by its `id`) and, on failure, the
combined score, the per-metric breakdown, and the judge's reasons — not just a
pass/fail count. Run a single case with `pytest -v -k tc-02`.

After a run you'll find:

- `reports/latest_scores.json` — per-test and aggregate scores
- `reports/report_<timestamp>.md` — a readable report

---

## Configuration

All knobs live in `config.py` and can be overridden via environment variables.
**Thresholds are never hardcoded in the tests.**

| Setting | Env var | Default |
| --- | --- | --- |
| Aggregate gate threshold | `AGGREGATE_THRESHOLD` | `0.75` |
| Target model | `TARGET_MODEL` | `llama3-8b-8192` |
| Evaluator (judge) model | `EVALUATOR_MODEL` | `llama3-70b-8192` |
| Groq API key | `GROQ_API_KEY` | _(required for live runs)_ |
| Target temperature | `TARGET_TEMPERATURE` | `0.0` |

Per-case minimums live in `evals/test_cases.json` under `minimum_score`.

---

## Test cases & failure modes

The 10 bundled cases cover: **hallucination**, **off-topic response**,
**missing key information**, **incorrect format**, and **overly verbose output**.
Each entry has: `input`, `context` (for RAG/faithfulness), an
`expected_output_criteria` list, and a `minimum_score`.

---

## CI / CD

`.github/workflows/eval_ci.yml` runs on every pull request:

1. Installs dependencies.
2. Runs a `--dry-run` sanity check.
3. Runs `pytest -v -ra` (shows which tests failed and why).
4. Runs an independent gate that reads `reports/latest_scores.json` and fails
   the build if the aggregate score is below the threshold.
5. Uploads the JSON + Markdown reports as build artifacts.

Add your Groq key as a repository secret named **`GROQ_API_KEY`**. Optionally
override the threshold with a repository variable **`AGGREGATE_THRESHOLD`**.

---

## Where to take it next

- Add domain-specific cases (customer support, code-gen, summarization).
- Build a Streamlit dashboard that charts score trends across `reports/`.
- Post a Slack notification when CI fails.
- Compare evaluator models for judgment consistency.
