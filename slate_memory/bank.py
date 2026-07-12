"""SlateBank — one-shot attractor memory with error-correcting recall.

Text or image embeddings are sign-projected onto N bipolar cells (SimHash),
converting cosine similarity into Hamming similarity, and committed one-shot.
Recall settles the probe into the nearest stored attractor via a softmax-
weighted (modern Hopfield) feedback update.

Benchmarked: accuracy parity with exact vector search; 98% prompt-token
savings over context stuffing; haiku+slate ties opus+full-context at 1/640th
the cost.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np


class SlateBank:
    """One-shot attractor memory with error-correcting recall.

    Parameters
    ----------
    n_cells : int
        Number of bipolar cells in the attractor state. More cells = more
        capacity, but slower recall. 10,000 is the verified sweet spot.
    dim : int
        Dimensionality of input embeddings. Must match your embedder
        (384 for MiniLM, 768 for many others, 1536 for OpenAI).
    beta : float
        Winner-take-all sharpness. Higher = harder selection among stored
        patterns. 60.0 is verified; don't change without benchmarking.
    distinctiveness : bool
        Down-weight cells where all stored patterns agree (shared template
        scaffolding). Essential for visual patterns; neutral for text.
    dedup_threshold : float
        Overlap above which a new commit is refused as a duplicate.
        0.95 is conservative; lower to allow finer distinctions.
    seed : int
        RNG seed for the projection matrix. Two SlateBank instances with
        the same seed + dim + n_cells produce identical projections, so
        patterns are portable between them.
    """

    def __init__(
        self,
        n_cells: int = 10_000,
        dim: int = 384,
        beta: float = 60.0,
        distinctiveness: bool = True,
        dedup_threshold: float = 0.95,
        seed: int = 7,
    ):
        rng = np.random.default_rng(seed)
        self._R = rng.standard_normal((n_cells, dim)).astype(np.float32)
        self._n = n_cells
        self._beta = beta
        self._distinctiveness = distinctiveness
        self._dedup_threshold = dedup_threshold
        self._lock = threading.Lock()
        self._patterns: list[np.ndarray] = []
        self._meta: list[dict] = []
        self._P: np.ndarray | None = None
        self._w: np.ndarray | None = None

    @property
    def count(self) -> int:
        """Number of stored patterns."""
        return len(self._patterns)

    def _project(self, emb: np.ndarray) -> np.ndarray:
        e = np.asarray(emb, dtype=np.float32).ravel()
        return np.where(self._R @ e >= 0, 1.0, -1.0).astype(np.float32)

    def _finalize(self) -> None:
        if not self._patterns:
            return
        self._P = np.stack(self._patterns)
        if self._distinctiveness and len(self._patterns) >= 2:
            agree = np.abs(self._P.mean(axis=0))
            w = 1.0 - agree
            s = float(w.sum())
            if s > self._n * 0.05:
                self._w = (w * (self._n / s)).astype(np.float32)
            else:
                self._w = None
        else:
            self._w = None

    def _overlaps(self, s: np.ndarray) -> np.ndarray:
        if self._w is not None:
            return (self._P @ (self._w * s)) / self._n
        return (self._P @ s) / self._n

    # ── write ────────────────────────────────────────────────────────────

    def commit(self, embedding: np.ndarray, meta: dict) -> tuple[bool, str]:
        """Store a pattern one-shot. Returns (committed, reason).

        Parameters
        ----------
        embedding : array-like
            The embedding vector (must match `dim`).
        meta : dict
            Arbitrary metadata returned on recall (text, id, tags, etc.).
        """
        pat = self._project(embedding)
        with self._lock:
            if self._patterns and self._dedup_threshold < 1.0:
                if self._P is None:
                    self._finalize()
                overlaps = (self._P @ pat) / self._n
                best = float(np.max(overlaps))
                if best >= self._dedup_threshold:
                    idx = int(np.argmax(overlaps))
                    return False, f"duplicate (overlap {best:.3f} with {self._meta[idx].get('id', idx)})"
            self._patterns.append(pat)
            self._meta.append(meta)
            self._P = None
            return True, "committed"

    def commit_batch(
        self, embeddings: list[np.ndarray], metas: list[dict]
    ) -> list[tuple[bool, str]]:
        """Commit multiple patterns. Dedup checks run against the growing store."""
        results = []
        for emb, meta in zip(embeddings, metas):
            results.append(self.commit(emb, meta))
        return results

    def remove(self, predicate) -> int:
        """Remove every pattern whose meta matches `predicate(meta) -> bool`.

        Returns the number of patterns removed. Use for re-ingestion flows
        where a source document changed and its old chunks must go:

            bank.remove(lambda m: m.get("chapter_id") == chapter_id)
        """
        with self._lock:
            keep = [
                (p, m)
                for p, m in zip(self._patterns, self._meta)
                if not predicate(m)
            ]
            removed = len(self._patterns) - len(keep)
            if removed:
                self._patterns = [p for p, _ in keep]
                self._meta = [m for _, m in keep]
                self._P = None
            return removed

    # ── read ─────────────────────────────────────────────────────────────

    def recall(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
        max_cycles: int = 5,
        with_scores: bool = False,
        with_signals: bool = False,
    ) -> tuple[dict | None, list, float, int] | tuple[dict | None, list, float, int, dict]:
        """Settle into the nearest attractor. Returns (winner, ranked, confidence, cycles).

        Parameters
        ----------
        embedding : array-like
            Query embedding.
        top_k : int
            Number of ranked results to return.
        max_cycles : int
            Maximum settle iterations. Usually converges in 2.
        with_scores : bool
            When True, `ranked` is a list of (meta, score) pairs instead of
            bare metas, where score is the pattern's pre-settle overlap with
            the query. Cosine-like: ~1 for a match, ~0 for unrelated; may
            slightly exceed ±1 under distinctiveness weighting.
        with_signals : bool
            When True, returns a fifth element: {"familiarity", "margin"},
            both computed BEFORE settling. `margin` (top1−top2 overlap gap)
            is the calibrated retrieval-quality signal — correct-vs-wrong
            AUC 0.88 in the slate-bench limits battery; gate decisions on
            it. `familiarity` (max overlap) flags out-of-domain queries
            (AUC 1.00) but does NOT reliably detect a plausible-but-absent
            fact (AUC ~0.62). The `confidence` value is post-settle depth,
            NOT correctness (AUC 0.10, inverted — settling reaches the
            bottom of SOME attractor at ~1.0 whether or not it is the right
            one); prefer `margin` for any trust decision.
        """
        with self._lock:
            if not self._patterns:
                if with_signals:
                    return None, [], 0.0, 0, {"familiarity": 0.0, "margin": 0.0}
                return None, [], 0.0, 0
            if self._P is None:
                self._finalize()
            s = self._project(embedding)
            initial = self._overlaps(s)
            cycles = 0
            for _ in range(max_cycles):
                o = self._overlaps(s)
                a = np.exp(self._beta * (o - o.max()))
                a /= a.sum()
                s_new = np.where(self._P.T @ a >= 0, 1.0, -1.0).astype(np.float32)
                cycles += 1
                if np.array_equal(s_new, s):
                    break
                s = s_new
            final = self._overlaps(s)
            winner = int(np.argmax(final))
            order = [int(i) for i in np.argsort(-initial)[:top_k]]
            if winner in order:
                order.remove(winner)
            order = ([winner] + order)[:top_k]
            if with_scores:
                ranked = [(self._meta[i], float(initial[i])) for i in order]
            else:
                ranked = [self._meta[i] for i in order]
            if with_signals:
                if initial.size >= 2:
                    two = np.partition(initial, -2)[-2:]
                    signals = {"familiarity": float(two[-1]),
                               "margin": float(two[-1] - two[-2])}
                else:
                    signals = {"familiarity": float(initial[0]),
                               "margin": float(initial[0])}
                return self._meta[winner], ranked, float(final[winner]), cycles, signals
            return self._meta[winner], ranked, float(final[winner]), cycles

    def familiar(self, embedding: np.ndarray) -> tuple[dict | None, float]:
        """Quick similarity check without full settle. Returns (best_meta, score)."""
        with self._lock:
            if not self._patterns:
                return None, 0.0
            if self._P is None:
                self._finalize()
            s = self._project(embedding)
            o = self._overlaps(s)
            best = int(np.argmax(o))
            return self._meta[best], float(o[best])

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save patterns and metadata to a directory."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._patterns:
                P = np.stack(self._patterns)
                np.save(str(d / "patterns.npy"), P)
            with open(d / "meta.json", "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=1)
            with open(d / "config.json", "w") as f:
                json.dump({
                    "n_cells": self._n,
                    "dim": self._R.shape[1],
                    "beta": self._beta,
                    "seed": 7,
                    "distinctiveness": self._distinctiveness,
                    "dedup_threshold": self._dedup_threshold,
                    "count": len(self._patterns),
                }, f, indent=2)

    def load(self, path: str | Path) -> int:
        """Load patterns and metadata. Returns number of patterns loaded."""
        d = Path(path)
        if not (d / "patterns.npy").exists():
            return 0
        with self._lock:
            P = np.load(str(d / "patterns.npy"))
            with open(d / "meta.json", encoding="utf-8") as f:
                self._meta = json.load(f)
            n = min(len(P), len(self._meta))
            self._patterns = [P[i] for i in range(n)]
            self._meta = self._meta[:n]
            self._P = None
            return n

    # ── utility ──────────────────────────────────────────────────────────

    def list(self, limit: int = 50) -> list[dict]:
        """Return metadata for stored patterns."""
        with self._lock:
            return list(self._meta[:limit])

    def stats(self) -> dict:
        """Return summary statistics."""
        with self._lock:
            return {
                "count": len(self._patterns),
                "n_cells": self._n,
                "dim": self._R.shape[1],
                "beta": self._beta,
                "distinctiveness": self._distinctiveness,
                "dedup_threshold": self._dedup_threshold,
            }
