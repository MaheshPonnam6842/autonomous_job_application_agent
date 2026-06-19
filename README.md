# Autonomous Job Application Agent

[![CI](https://github.com/MaheshPonnam6842/autonomous_job_application_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/MaheshPonnam6842/autonomous_job_application_agent/actions/workflows/ci.yml)

A **LangGraph** agent that reads a resume and a job description, scores fit along
three explainable dimensions, and — only when warranted — produces a *scoped,
factual* resume rewrite plus recruiter outreach drafts. It runs fully locally on
**Ollama**; no data leaves the machine.

The repository is intentionally organized to show how such a system is built and
hardened incrementally:

| Path | What it is |
|------|------------|
| [`v1_baseline/`](v1_baseline) | The first cut — deterministic flow, conservative GenAI, proves the data contract. |
| [`v_final/`](v_final) | The production-grade version — typed state, a resilient LLM layer, composite scoring, scoped rewrites, run artifacts. |
| [`utils/`](utils), [`src/`](src) | Shared text parsing, PDF link extraction, logging. |
| [`web/`](web) | Flask UI with async two-phase execution (fast analysis → background rewrite). |
| [`tests/`](tests) | Offline test suite (no model server required). |

---

## Pipeline

```
START
  └─ resume_ingest      normalize, section-split, extract bullets & skills
     └─ resume_structure parse typed experience / project / education entries
       └─ jd_ingest     LLM → validated structured JD (skills/responsibilities/seniority)
            └─ skill_extraction   canonical skill profiles + overlap/gaps
                 └─ matching      skill + semantic + ATS  →  weighted overall score
                      └─ decision rewrite? which scoped strategy?  (explainable)
                           ├─ rewrite_required → resume_rewrite (guardrailed GenAI)
                           └─ skip ───────────┐
                                              └─ outreach   DM + email drafts
                                                   └─ tracking  persist run artifact
                                                        └─ END
```

See [`docs/architecture.md`](docs/architecture.md) for node-by-node read/write contracts.

## Design principles

- **Graceful degradation.** Nodes never crash because the model server is down —
  every LLM call falls back to deterministic behavior and records a warning. The
  whole graph runs offline (that's how the tests work).
- **Validated LLM output.** JD extraction is parsed into a Pydantic schema, so a
  malformed model response degrades to a well-formed object instead of breaking a node.
- **Factual, scoped rewriting.** The rewrite node is gated by the decision node and
  constrained to re-emphasize content that *already exists* — no invented skills,
  metrics, employers, or dates.
- **Auditability.** Decisions record a human-readable reason and strategy; every run
  writes a JSON artifact under `artifacts/`.
- **Config over code.** Models, scoring weights, and thresholds live in
  [`v_final/config.py`](v_final/config.py) and are overridable via env (`.env.example`).

## Quickstart

```bash
python -m venv venv && source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt

# Pull local models (one-time)
ollama pull llama3.1:8b
ollama pull nomic-embed-text     # enables semantic similarity (optional)

# Run the agent on a bundled sample
python -m v_final.pipeline

# Run fully offline (deterministic path, no Ollama needed)
JOBAGENT_LLM_ENABLED=0 python -m v_final.pipeline

# Tests
pytest -q

# Web UI
python web/app.py    # http://localhost:5000
```

> Run commands from the repo root so the `v_final`, `utils`, and `src` packages resolve.

### Run with Docker

Brings up the web app and a local Ollama server together:

```bash
docker compose up --build                                   # app on http://localhost:5000
docker compose exec ollama ollama pull llama3.1:8b          # one-time
docker compose exec ollama ollama pull nomic-embed-text     # optional: semantic score
```

The app degrades gracefully if models aren't pulled yet (analysis still runs).

## Continuous integration

Every push runs [CI](.github/workflows/ci.yml) on Python 3.11 and 3.12: `ruff`
lint + the full pytest suite on the offline deterministic path (no model server
needed in CI).

## Scoring

`overall = w_skill·skill_match + w_semantic·semantic_similarity + w_ats·ats_match`
(weights default 0.5 / 0.3 / 0.2, auto-normalized, configurable).

- **skill_match** — coverage of JD skills by the resume's canonical skill profile.
- **semantic_similarity** — embedding cosine of resume vs JD (lexical-cosine fallback
  when embeddings are unavailable).
- **ats_match** — coverage of required/keyword ATS terms.
