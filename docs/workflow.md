# Workflow

## End-to-end run

1. **Ingest** — the resume is normalized, split into sections, and mined for
   bullets and a structured skills map. The JD is sent to the LLM and parsed into
   a validated `JDExtraction` (skills required/preferred, tools, responsibilities,
   seniority, keywords).
2. **Understand** — `skill_extraction` canonicalizes both sides into
   `SkillProfile`s (hard skills / tools / cloud / ml concepts / soft / keywords)
   and computes overlap and gaps.
3. **Score** — `matching` blends three sub-scores (skill / semantic / ATS) into a
   single weighted `overall_match_score`, and records `matched_skills` and
   `missing_required_skills`.
4. **Decide** — `decision` compares the score and gaps to thresholds, then either
   skips the rewrite or selects the *narrowest effective* rewrite strategy, with a
   human-readable reason.
5. **Act** — if a rewrite is warranted, `resume_rewrite` produces a scoped,
   guardrailed rewrite; `outreach` drafts a DM and email; `tracking` writes a JSON
   run artifact.

## Two execution modes

- **Single graph** ([`job_application_graph.py`](../v_final/graph/job_application_graph.py))
  — full pipeline in one `invoke`; used by the CLI / `pipeline.py`.
- **Split graphs** ([`analysis_graph.py`](../v_final/graph/analysis_graph.py) +
  [`rewrite_graph.py`](../v_final/graph/rewrite_graph.py)) — the web app runs
  analysis synchronously to show scores instantly, then runs the slow GenAI
  rewrite in a background thread and polls for completion.

## Reproducing a run

Every run writes `artifacts/run_<timestamp>.json` containing the scores, the
decision and its reason, the skill gaps, and previews of the generated outputs —
the agent's audit trail.
