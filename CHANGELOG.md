# Changelog

## 0.2.0 — 2026-07-12

### Added
- `SlateBank.recall(..., with_signals=True)` returns a fifth element
  `{"familiarity", "margin"}`, both computed before settling. **`margin`
  (top1−top2 pre-settle overlap gap) is the calibrated retrieval-quality
  signal** — correct-vs-wrong AUC 0.88 in the slate-bench limits battery —
  gate trust in the winner on it. `familiarity` (max pre-settle overlap)
  detects out-of-domain queries (AUC 1.00) but does not reliably detect a
  plausible-but-absent fact (AUC ~0.62).

### Guidance
- The existing `confidence` return value measures settle depth, not
  correctness (correct-vs-wrong AUC 0.10 — inverted: settling reaches the
  bottom of *some* attractor at ~1.0 whether or not it is the right one).
  It is unchanged for compatibility; prefer `margin` for decisions.
- When committing a correction to an already-stored fact, write it as a
  clean restatement — no "UPDATE:"/"Correction:" prefixes. Explicit flags
  push the embedding away from the queries that will later look for it,
  and the stale fact wins recall (fresh-win 39% → 20% in the supersession
  benchmark). Better: `remove()` the stale pattern, then `commit()` the
  correction.

## 0.1.1
- `remove()` for re-ingestion flows, `with_scores` ranked output.

## 0.1.0
- Initial release: one-shot commit, error-correcting settle recall,
  dedup guard, persistence, familiarity check.
