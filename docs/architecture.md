# Architecture

The agent is a LangGraph `StateGraph` over a single typed state object
([`JobApplicationState`](../v_final/state/job_application_state.py)). Each node is a
pure `state -> state` function; the graph wires them together.

## Layers

```
config.py        env-driven settings: models, scoring weights, thresholds, paths
llm/             OllamaClient: retries, JSON-mode structured output, embeddings,
                 availability probing, graceful degradation  (nodes never import ollama)
state/           the typed contract every node reads and writes
nodes/           one responsibility each (below)
graph/           full graph + analysis/rewrite sub-graphs (for the async web flow)
pipeline.py      programmatic entrypoint + human-readable report
```

## Node contracts

| Node | Reads | Writes | LLM? |
|------|-------|--------|------|
| `resume_ingest` | `resume_raw_text`, `resume_pdf_path?` | `resume_clean_text`, `resume_sections`, `experience/project_bullets[_by_group]`, `resume_skills_structured`, `resume_links` | no |
| `jd_ingest` | `job_description_text` | `jd_clean_text`, `jd_title`, `jd_domain`, `jd_seniority_level`, `jd_skills_required/preferred`, `jd_tools_process`, `jd_responsibilities`, `jd_keywords` | **yes** (validated) |
| `skill_extraction` | resume + jd text/skills | `resume_skill_profile`, `jd_skill_profile`, `skill_overlap`, `skill_gap_hard`, `skill_gap_soft` | no |
| `matching` | skill profiles, clean texts | `skill_match_score`, `semantic_similarity_score`, `ats_match_score`, `overall_match_score`, `score_breakdown`, `matched_skills`, `missing_required_skills` | embeddings (optional) |
| `decision` | `overall_match_score`, `ats_match_score`, `missing_required_skills` | `rewrite_required`, `rewrite_strategy`, `rewrite_reason` | no |
| `resume_rewrite` | resume/jd text, `rewrite_strategy`, `missing_required_skills` | `optimized_resume_text`, `resume_version`, `rewrite_error?` | **yes** |
| `outreach` | `matched_skills` | `outreach_dm_text`, `outreach_email_text` | yes (polish, optional) |
| `tracking` | scoring/decision/output fields | `run_artifact_path`, `agent_notes` | no |

## Rewrite strategies (chosen by `decision`)

| Strategy | When | Scope given to the model |
|----------|------|--------------------------|
| `none` | strong fit | (rewrite skipped) |
| `summary_only` | low ATS, skills otherwise OK | summary only |
| `summary_and_skills` | several required skills under-represented | summary + skills reorg |
| `experience_bullets` | moderate gap | summary + sharpen experience bullets |
| `full_rewrite` | overall < 0.40 | all sections |

## Failure behavior

The model server being unavailable is a *normal*, handled condition:

- `jd_ingest` → empty structured fields + a `warnings` entry; downstream skill
  extraction still works by scanning the raw JD text.
- `matching` → semantic score uses the lexical-cosine fallback.
- `resume_rewrite` → returns the original resume, `resume_version =
  "rewrite_failed_fallback"`.
- `outreach` → returns the un-polished deterministic draft.

This is why the full pipeline (and the test suite) runs with
`JOBAGENT_LLM_ENABLED=0`.
