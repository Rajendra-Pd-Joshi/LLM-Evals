<p align="center">
  <img src="assets/LLM-Evals_%20RAG%20Evaluation%20Like%20a%20Pro.png" alt="LLM-Evals - RAG Evaluation Like a Pro" width="100%">
</p>

<h1 align="center">LLM-Evals</h1>

<p align="center">
  <strong>End-to-end evaluation and regression testing for RAG and LLM applications.</strong><br>
  Measure retrieval, generation, application quality, safety, latency, cost, reliability, and regressions from one reproducible suite.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/DeepEval-4.x-6C5CE7?style=for-the-badge" alt="DeepEval">
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge" alt="LangChain">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=for-the-badge" alt="uv">
</p>

---

## Overview

**LLM-Evals** is a research and engineering project for evaluating a real Retrieval-Augmented Generation (RAG) application.

The core question is:

> **When I change my RAG/LLM application, did it improve, or did I introduce a regression somewhere else?**

A single LLM score cannot answer that. A change can improve correctness while increasing latency, improve recall while increasing cost, or improve answer quality while weakening safety. This project therefore evaluates the system across multiple dimensions and compares a candidate run against a known baseline.

### What is evaluated?

- Retriever quality
- Generator quality
- End-to-end RAG quality
- Application correctness, completeness, and style
- Safety and security behavior
- Latency and streaming performance
- Token usage and estimated cost
- Reliability and retry behavior
- Online evaluation of LangSmith traces
- Baseline vs candidate regressions

---

## Architecture

```text
                    VTT TRANSCRIPTS
                          │
                          ▼
                 Clean + Chunk Text
                    1000 / 150
                          │
                          ▼
             OpenAI text-embedding-3-small
                          │
                          ▼
                      ChromaDB
                   Vector Store
                          │
                  Similarity Search
                          │
                          ▼
                ┌─────────────────┐
                │ Reranking       │
                │ OpenRouter      │
                │ rerank-2.5-lite │
                └────────┬────────┘
                         │
                       top_k
                         │
                         ▼
                ┌─────────────────┐
                │ Generator       │
                │ GPT-4o-mini     │
                │ temperature=0   │
                └────────┬────────┘
                         │
                         ▼
                    RAG ANSWER
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                  ▼
   QUALITY             SAFETY              OPS
   Retrieval           Scope               Latency
   Generation          Leakage             TTFT
   RAG/App             PII                 Cost
                       Toxicity            Reliability
       │                 │                  │
       └─────────────────┼──────────────────┘
                         ▼
                  EVALUATION SNAPSHOT
                         │
                  ┌──────┴──────┐
                  ▼             ▼
               BASELINE      CANDIDATE
                  │             │
                  └──────┬──────┘
                         ▼
                 REGRESSION ENGINE
                         │
                  PASS / REVIEW / FAIL
```

A key design decision is in `evals/run_suite.py`: the pipeline is built once and its retriever is injected into the component evaluations. This keeps one evaluation snapshot tied to one pipeline configuration.

---

# RAG application

The runtime implementation lives in `src/llm_evals/`.

| File | Responsibility |
|---|---|
| `retriever.py` | Load VTT transcripts, remove timestamps, chunk, embed, persist ChromaDB, retrieve documents |
| `reranker.py` | Over-fetch candidates and rerank them through OpenRouter |
| `generator.py` | Grounded answer generation and streaming generation |
| `rag_pipeline.py` | Combine retrieval, reranking, and generation |

### Retrieval configuration

The current implementation uses:

- `text-embedding-3-small`
- chunk size: `1000`
- chunk overlap: `150`
- ChromaDB persistence
- base retrieval `k=5`
- reranker fetch size: `10`
- reranker output: `5`
- `voyageai/rerank-2.5-lite` through OpenRouter

If reranking fails, the current implementation falls back to the original similarity-search order.

### Generator

The generator uses `gpt-4o-mini` with temperature `0`.

Its prompt follows a **faithfulness-first** strategy:

- answer from course context;
- do not invent outside information;
- abstain when the context is insufficient;
- address all parts of a question;
- explain concepts naturally;
- resist instructions embedded in retrieved content;
- do not expose hidden prompts or private configuration;
- avoid unnecessary disclosure of sensitive information;
- maintain a respectful teaching style.

The same chain supports both normal generation and streaming, allowing TTFT measurement without creating a separate generation configuration.

---

# Evaluation layers

## 1. Retriever evaluation

`evals/eval_retriever.py` evaluates retrieval independently from generation using `goldens/retriever_goldens.json`.

It measures:

- **Contextual Recall** — whether required evidence was retrieved.
- **Contextual Precision** — whether retrieved evidence was useful and appropriately ranked.

This answers:

> **Did we retrieve the right evidence?**

---

## 2. Generator evaluation

`evals/eval_generator.py` isolates generation by testing the generator against controlled/golden context.

This separates two failure modes:

```text
Poor retrieval → missing evidence → bad answer

Correct retrieval → evidence available → bad generation
```

That distinction is important when debugging a RAG system.

---

## 3. RAG pipeline / triad evaluation

`evals/eval_rag_pipeline.py` evaluates the complete pipeline.

The pipeline returns:

```python
{
    "query": query,
    "context": context,
    "answer": answer,
}
```

This exposes the relationship between:

```text
Question → Retrieved Context → Generated Answer
```

and allows the complete RAG behavior to be evaluated rather than evaluating retrieval and generation only in isolation.

---

## 4. Application evaluation

`evals/eval_application.py` runs the actual RAG pipeline against `goldens/correctness_goldens.json`.

It intentionally separates three concepts:

| Metric | Question |
|---|---|
| Correctness | Is the answer factually correct? |
| Completeness | Did it cover the important expected points? |
| Style | Was the explanation clear and appropriate for teaching? |

The correctness rubric does not punish a concise but correct answer merely because it is shorter. Completeness is evaluated separately.

---

# Safety evaluation

Safety is a separate evaluation dimension rather than a side effect of answer-quality scoring.

Relevant modules include:

```text
evals/eval_saftey.py
evals/eval_scope_adherence.py
evals/eval_leakage.py
evals/eval_toxicity.py
```

> `eval_saftey.py` is the existing repository filename; the spelling is preserved here to match the codebase.

The safety suite covers:

| Area | Purpose |
|---|---|
| Scope adherence | Keep the assistant within the intended course/domain scope |
| Protected-information leakage | Detect attempts to extract protected material |
| PII leakage | Detect inappropriate disclosure of personal information |
| Toxicity | Detect abusive, hateful, humiliating, or toxic responses |
| Prompt injection | Test whether retrieved content can override intended behavior |

The generator prompt treats retrieved context and student input as untrusted content. Instructions contained inside those blocks must not override the assistant's rules.

For high-stakes use, aggregate safety scores should be combined with explicit per-case safety gates so one severe failure cannot disappear inside an average.

---

# Operational evaluation

`evals/eval_ops.py` evaluates production-oriented behavior.

### Latency

The suite tracks measurements such as:

- p50
- p95
- p99
- end-to-end latency
- time-to-first-token (TTFT)

The metric registry currently uses end-to-end p95 latency as the main latency guardrail while retaining other measurements for analysis.

### Cost

Token usage is converted into estimated model cost. This allows quality improvements to be considered alongside their financial impact.

### Reliability

The project measures request success/error behavior and retry characteristics.

This makes comparisons such as the following possible:

```text
                 Quality   Latency   Cost   Reliability
Model A            0.88      900ms    low      high
Model B            0.91     1600ms    high     high
```

The best choice depends on the application's requirements rather than one quality number.

---

# Online evaluation

`evals/eval_online.py` provides an online evaluation path using LangSmith traces.

```text
Real Application Traffic
          │
          ▼
      LangSmith
          │
          ▼
     Trace Sampling
          │
          ▼
    Online Evaluator
       ┌────┴────┐
       ▼         ▼
 Faithfulness  Relevancy
       │         │
       └────┬────┘
            ▼
      Trace Feedback
```

Offline evaluation uses known questions and expectations. Online evaluation uses real application traces. Together they provide broader coverage.

---

# Regression testing

The central regression workflow is implemented through:

```text
evals/run_suite.py
evals/metric_registry.py
evals/compare.py
```

### `run_suite.py`

The orchestrator:

1. builds one `RagPipeline`;
2. obtains the pipeline retriever;
3. runs retriever evaluation;
4. runs generator evaluation;
5. runs RAG pipeline evaluation;
6. runs application evaluation;
7. runs safety evaluation;
8. runs operational evaluation;
9. flattens the metrics into common dotted IDs;
10. records provenance metadata;
11. writes a JSON snapshot.

The snapshot is the atomic unit used for regression comparison.

### Example metric IDs

```text
retriever.contextual_recall.avg_score
retriever.contextual_precision.avg_score
pipeline.faithfulness.avg_score
application.correctness.avg_score
application.completeness.avg_score
safety.scope.avg_score
safety.leakage.pii_avg_score
ops.latency.e2e_p95_ms
ops.cost.cost_per_query_usd
ops.reliability.success_rate
```

---

# Metric registry

`evals/metric_registry.py` is the policy layer.

Every metric is interpreted through:

### Direction

```text
Higher is better:
  correctness
  faithfulness
  recall
  success rate

Lower is better:
  latency
  cost
  toxicity
```

### Kind

| Kind | Meaning |
|---|---|
| `gate` | Regression can block promotion |
| `guardrail` | Regression triggers review |
| `info` | Tracked but does not affect verdict |

### Tolerance

Because LLM judges and external APIs have run-to-run noise, a small change is not automatically treated as a regression. The registry uses absolute and relative tolerance rules.

The current policy uses tighter safety gates, a wider quality guardrail, and percentage-based latency/cost guardrails.

---

# PASS / REVIEW / FAIL

`evals/compare.py` compares baseline and candidate snapshots.

```text
                 BASELINE vs CANDIDATE
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        GATES          GUARDRAILS        INFO
          │               │               │
          ▼               ▼               ▼
       BLOCKED         REGRESSED        TRACKED
          │               │
          ▼               ▼
         FAIL           REVIEW

   No blocking regression → PASS
```

Exit codes:

```text
PASS   = 0
FAIL   = 1
REVIEW = 2
```

This makes the comparison engine suitable for future CI/release automation.

---

# Golden datasets

The repository currently contains several evaluation datasets:

```text
goldens/
├── correctness_goldens.json
├── faithfulness_dataset.json
├── leakage_goldens.json
├── retriever_goldens.json
├── scope_goldens.json
└── toxicity_goldens.json
```

A strong evaluation set should contain:

- factual questions;
- multi-part questions;
- retrieval-sensitive questions;
- incomplete-context cases;
- out-of-scope questions;
- prompt-injection attempts;
- leakage attempts;
- toxic/abusive inputs;
- questions with multiple valid answer phrasings.

The quality of the evaluation framework is limited by the quality of the evaluation data.

---

# Course data

The project contains eight VTT transcript sessions under `data/`:

```text
data/
├── YT Sandbox   LLM Evals Session 1.vtt
├── YT Sandbox   LLM Evals Session 2.vtt
├── YT Sandbox   LLM Evals Session 3.vtt
├── YT Sandbox   LLM Evals Session 4.vtt
├── YT Sandbox   LLM Evals Session 5.vtt
├── YT Sandbox   LLM Evals Session 6.vtt
├── YT Sandbox   LLM Evals Session 7.vtt
└── YT Sandbox   LLM Evals Session 8.vtt
```

The retriever removes VTT timestamps, joins the transcript text, extracts the session number into document metadata, and then chunks the material for embedding.

---

# Handwritten notes and research material

The repository also contains:

**[Application Eval.pdf](Application%20Eval.pdf)**

This file is kept alongside the implementation as supporting research material and handwritten notes.

The project therefore has two connected layers:

```text
HANDWRITTEN / RESEARCH NOTES
             │
             ▼
      Understand concepts
             │
             ▼
       Implement evals
             │
             ▼
       Run experiments
             │
             ▼
      Capture measurements
             │
             ▼
     Compare with baseline
```

This makes the repository useful not only as code, but also as a record of the learning and evaluation process.

---

# Repository structure

```text
LLM-Evals/
│
├── assets/
│   └── LLM-Evals_ RAG Evaluation Like a Pro.png
│
├── data/
│   ├── YT Sandbox   LLM Evals Session 1.vtt
│   ├── YT Sandbox   LLM Evals Session 2.vtt
│   ├── ...
│   └── YT Sandbox   LLM Evals Session 8.vtt
│
├── goldens/
│   ├── correctness_goldens.json
│   ├── faithfulness_dataset.json
│   ├── leakage_goldens.json
│   ├── retriever_goldens.json
│   ├── scope_goldens.json
│   └── toxicity_goldens.json
│
├── evals/
│   ├── compare.py
│   ├── eval_application.py
│   ├── eval_cost.py
│   ├── eval_generator.py
│   ├── eval_latency.py
│   ├── eval_leakage.py
│   ├── eval_online.py
│   ├── eval_ops.py
│   ├── eval_rag_pipeline.py
│   ├── eval_reliability.py
│   ├── eval_retriever.py
│   ├── eval_saftey.py
│   ├── eval_scope_adherence.py
│   ├── eval_toxicity.py
│   ├── harness.py
│   ├── metric_registry.py
│   └── run_suite.py
│
├── src/
│   └── llm_evals/
│       ├── __init__.py
│       ├── generator.py
│       ├── rag_pipeline.py
│       ├── reranker.py
│       └── retriever.py
│
├── Application Eval.pdf
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

---

# Technology stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core implementation |
| uv | Environment/dependency management |
| LangChain | RAG components |
| ChromaDB | Persistent vector store |
| OpenAI embeddings | `text-embedding-3-small` |
| GPT-4o-mini | Generation and evaluation judging |
| OpenRouter | Reranking API |
| DeepEval | LLM evaluation framework |
| LangSmith | Tracing and online evaluation |
| python-dotenv | Environment configuration |
| pytest | Testing |
| Google GenAI | Additional model integration dependency |

The repository requires Python `>=3.11` and pins `.python-version` to `3.11`.

---

# Installation

## 1. Clone

```bash
git clone https://github.com/Rajendra-Pd-Joshi/LLM-Evals.git
cd LLM-Evals
```

## 2. Install uv

```bash
pip install uv
```

## 3. Install dependencies

```bash
uv sync
```

The repository includes `uv.lock` for reproducible dependency resolution.

---

# Environment variables

Create `.env` in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=llm-evals
```

Only configure the credentials required by the evaluations you run.

**Never commit `.env`, API keys, access tokens, or provider secrets.**

---

# Running the project

### Retriever smoke test

```bash
uv run python src/llm_evals/retriever.py
```

### Generator smoke test

```bash
uv run python src/llm_evals/generator.py
```

### RAG pipeline smoke test

```bash
uv run python src/llm_evals/rag_pipeline.py
```

---

# Running the full evaluation suite

### Candidate evaluation

```bash
uv run python -m evals.run_suite
```

Default output:

```text
baselines/candidate.json
```

### Create a baseline

```bash
uv run python -m evals.run_suite --baseline
```

### Label an experiment

```bash
uv run python -m evals.run_suite --label "new reranker experiment"
```

### Custom output

```bash
uv run python -m evals.run_suite --out results/my-run.json
```

### Quiet mode

```bash
uv run python -m evals.run_suite --quiet
```

### Save informational metrics too

```bash
uv run python -m evals.run_suite --full
```

---

# Compare baseline and candidate

After generating a baseline and candidate:

```bash
uv run python -m evals.compare
```

Show informational metrics as well:

```bash
uv run python -m evals.compare --all
```

Recommended workflow:

```text
1. Establish known-good baseline
2. Change one meaningful variable
3. Run candidate evaluation
4. Compare snapshots
5. Investigate regressions
6. Keep or revert the change
```

Example variables to experiment with:

- `top_k`
- `fetch_k`
- chunk size
- chunk overlap
- embedding model
- reranker
- generation model
- generation prompt

---

# Example experiment matrix

| Experiment | Expected upside | Watch |
|---|---|---|
| Increase `top_k` | More evidence | Recall, correctness, latency, cost |
| Decrease `top_k` | Less context/cost | Recall, faithfulness, correctness |
| Increase `fetch_k` | More reranking candidates | Precision, latency |
| Change reranker | Better ranking | Precision, correctness, latency |
| Change embeddings | Different semantic retrieval | Recall, precision, application quality |
| Change generator | Potential answer-quality gain | Correctness, completeness, cost, latency |
| Change prompt | Different behavior | Faithfulness, correctness, completeness, safety |
| Change chunk size | Different evidence granularity | Retrieval and answer quality |

The objective is not to maximize one score. It is to find a configuration that satisfies **quality + safety + performance + cost** requirements.

---

# Reproducibility

LLM evaluations are noisy. Identical runs can differ because of model behavior, external APIs, retrieval changes, and system load.

The suite records provenance such as:

- evaluation timestamp;
- Git SHA;
- generator prompt hash;
- experiment label;
- suite duration;
- number of metrics.

A future centralized configuration should additionally record the exact effective model, embedding model, reranker, retrieval parameters, and dataset versions.

---

# Known limitations

This is an active learning/research implementation. Important areas for future hardening include:

### Vector-store freshness

The current retriever opens an existing Chroma directory when present. It does not yet maintain a source/configuration fingerprint proving that the stored index matches the current transcripts and embedding settings.

### Reranker fallback observability

A reranker failure falls back to similarity-search order. This is useful for availability but should eventually produce explicit fallback metrics so degraded runs are visible.

### Required metric validation

The comparison engine reports new/dropped metrics, but required release metrics should eventually be validated explicitly and missing gate metrics should fail closed.

### Central configuration

Model, retrieval, chunking, and dataset configuration should eventually be centralized and stamped into every evaluation snapshot.

### Judge calibration

LLM-as-a-judge metrics should be calibrated against human-reviewed examples before thresholds are treated as absolute truth.

---

# Roadmap

## Foundation

- [x] RAG application
- [x] Retriever evaluation
- [x] Generator evaluation
- [x] RAG/application evaluation
- [x] Safety evaluation
- [x] Operational evaluation
- [x] Baseline/candidate snapshots
- [x] Metric registry
- [x] PASS/REVIEW/FAIL comparison

## Reliability

- [ ] Unit-test regression engine
- [ ] Validate snapshot schema
- [ ] Fail closed on missing gate metrics
- [ ] Record complete runtime configuration
- [ ] Add dataset/configuration fingerprints
- [ ] Make reranker fallback observable

## Production evaluation

- [ ] CI integration
- [ ] Stable online sampling
- [ ] Incremental online trace processing
- [ ] Judge-cost monitoring
- [ ] Concurrency/load testing
- [ ] Per-case safety reporting

## Evaluation science

- [ ] Calibrate judges against human labels
- [ ] Estimate metric noise from repeated runs
- [ ] Add experiment comparison reports
- [ ] Compare providers/models systematically
- [ ] Add statistical confidence analysis

---

# Evaluation philosophy

### Measure before optimizing

Do not change a model, prompt, or retriever because it feels better. Establish a baseline and measure the change.

### Evaluate at multiple levels

Retriever, generator, RAG pipeline, application, safety, and operations answer different questions.

### Keep safety independent from quality

A system can be helpful and unsafe, or safe and unhelpful. These dimensions need separate measurements.

### Treat evaluation data as a product

Weak golden data produces a weak evaluation framework. Test cases should represent real failure modes.

### Preserve provenance

A score is difficult to trust without knowing which code, prompt, model, dataset, and configuration produced it.

### Optimize trade-offs

The best production configuration is usually not the one with the highest quality score. It is the one that meets quality and safety requirements at an acceptable cost and latency.

---

# Security and privacy

Never commit:

```text
.env
API keys
access tokens
provider credentials
private user data
private production traces
sensitive evaluation data
```

If a credential is accidentally committed, revoke/rotate it immediately. Removing it from the latest file is not enough if the secret remains in Git history.

---

# Quick command reference

```bash
# Install
uv sync

# Smoke tests
uv run python src/llm_evals/retriever.py
uv run python src/llm_evals/generator.py
uv run python src/llm_evals/rag_pipeline.py

# Full evaluation
uv run python -m evals.run_suite

# Baseline
uv run python -m evals.run_suite --baseline

# Experiment
uv run python -m evals.run_suite --label "my experiment"

# Full snapshot
uv run python -m evals.run_suite --full

# Compare
uv run python -m evals.compare
uv run python -m evals.compare --all
```

---

# Project status

**Active learning / research implementation**

This repository is intentionally more than a collection of DeepEval examples. It is being developed toward a repeatable evaluation workflow in which changes to an LLM application can be measured, compared, and eventually used as a CI/release gate.

The handwritten research material in `Application Eval.pdf`, golden datasets, RAG implementation, individual evaluators, operational measurements, and regression engine together form the project's learning and experimentation record.

---

# Author

**Rajendra Prasad Joshi**

Computer Science & Engineering | LLM / RAG Evaluation

GitHub: [Rajendra-Pd-Joshi](https://github.com/Rajendra-Pd-Joshi)

---

<p align="center">
  <strong>LLM-Evals — Measure first. Change deliberately. Ship with evidence.</strong>
</p>
