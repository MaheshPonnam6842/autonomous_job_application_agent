"""Rewrite Loop Controller — scores each rewrite candidate and keeps the best.

Forms a cycle with ``resume_rewrite``: rewrite -> this -> (retry | done). It
scores the candidate the rewrite node just produced against the JD using the same
text-based scorer as the baseline (so deltas are trustworthy), tracks the best
candidate, derives feedback for the next attempt, and exposes a before/after
comparison. Offline / failed rewrites short-circuit immediately (no looping).

Reads:  optimized_resume_text (current candidate), resume_*_text, jd_*, resume_skill_profile
Writes: rewrite_attempts, rewrite_improved, baseline_scores, best_scores,
        best_resume_text, rewrite_candidates, rewrite_feedback_terms,
        optimized_* scores, optimized_resume_text (best), rewrite_score_comparison
"""

from __future__ import annotations

from v_final.config import settings
from v_final.nodes.matching import ats_pass_label, score_resume_against_jd
from v_final.nodes.skill_extraction import _build_profile, _canon, _flatten, _scan_vocab
from v_final.state.job_application_state import JobApplicationState

_METRICS = ("skill_match", "semantic_similarity", "ats_match", "overall", "ats_pass")


def _is_better(cand: dict, best: dict | None, target: str, min_gain: float) -> bool:
    if best is None:
        return True
    if cand[target] > best[target] + min_gain:
        return True
    # tie on target -> prefer higher overall
    return abs(cand[target] - best[target]) <= 1e-9 and cand["overall"] > best["overall"]


def _feedback_terms(state: JobApplicationState, candidate_text: str) -> list[str]:
    """JD skills the resume HAS but this candidate didn't surface — safe to push."""
    original_set = _flatten(state.get("resume_skill_profile") or {})
    cand_set = _flatten(_build_profile(_scan_vocab(candidate_text)))
    jd_target: set[str] = set()
    for key in ("jd_skills_required", "jd_skills_preferred", "jd_keywords"):
        for t in state.get(key, []) or []:
            c = _canon(t)
            if c:
                jd_target.add(c)
    return sorted((jd_target & original_set) - cand_set)


def rewrite_loop_node(state: JobApplicationState) -> JobApplicationState:
    cfg = settings.rewrite
    target = cfg.target_metric if cfg.target_metric in _METRICS else "ats_match"

    # Score the original once, with the same text-based method used for candidates.
    if "baseline_scores" not in state:
        original_text = state.get("resume_clean_text") or state.get("resume_raw_text", "")
        state["baseline_scores"] = score_resume_against_jd(original_text, state)

    candidate_text = state.get("optimized_resume_text", "") or ""
    real_rewrite = state.get("resume_version") == "rewritten" and bool(candidate_text.strip())
    if not real_rewrite:
        # Offline or failed rewrite: nothing to score; let the router end the loop.
        state["rewrite_attempts"] = int(state.get("rewrite_attempts", 0))
        state["rewrite_improved"] = False
        return state

    cand = score_resume_against_jd(candidate_text, state)
    candidates = list(state.get("rewrite_candidates", []) or [])
    candidates.append({"attempt": len(candidates) + 1, **{m: cand[m] for m in _METRICS}})
    state["rewrite_candidates"] = candidates

    improved = _is_better(cand, state.get("best_scores"), target, cfg.min_gain)
    if improved:
        state["best_scores"] = {m: cand[m] for m in _METRICS}
        state["best_resume_text"] = candidate_text

    state["rewrite_attempts"] = int(state.get("rewrite_attempts", 0)) + 1
    state["rewrite_improved"] = improved
    state["rewrite_feedback_terms"] = _feedback_terms(state, candidate_text)

    # Finalize-so-far: outputs always reflect the best candidate + baseline delta.
    best = state["best_scores"]
    baseline = state["baseline_scores"]
    state["optimized_resume_text"] = state["best_resume_text"]
    state["optimized_skill_match_score"] = best["skill_match"]
    state["optimized_semantic_similarity_score"] = best["semantic_similarity"]
    state["optimized_ats_match_score"] = best["ats_match"]
    state["optimized_overall_match_score"] = best["overall"]
    state["optimized_ats_pass_score"] = best["ats_pass"]
    state["optimized_ats_pass_label"] = ats_pass_label(best["ats_pass"])
    state["rewrite_score_comparison"] = {
        m: {
            "before": baseline[m],
            "after": best[m],
            "delta": round(best[m] - baseline[m], 4),
        }
        for m in _METRICS
    }
    return state


def improve_router(state: JobApplicationState) -> str:
    """retry while improving and under the attempt cap; otherwise done."""
    if state.get("resume_version") != "rewritten":
        return "done"
    if int(state.get("rewrite_attempts", 0)) >= settings.rewrite.max_attempts:
        return "done"
    if not state.get("rewrite_improved", False):
        return "done"
    return "retry"
