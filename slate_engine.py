"""SlateBank — the CMYK attractor processor as a recall engine (v2).

Text enters as a MiniLM embedding, is sign-projected onto N bipolar cells,
and is committed one-shot. Recall settles the probe into the nearest stored
attractor via softmax-attention. v2 internals, same external API:

  - PACKED storage (default): patterns live as bit-packed uint8 rows
    (n_cells/8 bytes each — 32x less RAM than float32; ~38 MB at 30k
    patterns instead of ~2.4 GB), overlaps computed exactly via
    XOR + popcount. Requires numpy >= 2.0. Packed mode disables
    distinctiveness weighting (per-cell float weights don't fit the
    popcount path; benchmarked null on text — slate-bench finding 5).
  - INCREMENTAL writes: preallocated growing buffer, no full re-stack per
    commit. The v1 engine re-finalized the whole bank on every guarded
    commit (O(K) — 362 ms at 30k, quadratic to sync); v2 commits are
    dominated by the dedup scan alone.
  - SUPERSEDES: commit(emb, meta, supersedes=<id>) replaces the stale
    attractor in place. Without it, corrections lose recall to the fact
    they correct 60-80% of the time (slate-bench supersession benchmark).
  - SIGNALS: recall returns pre-settle familiarity + margin; the margin is
    the calibrated retrieval-quality signal (AUC 0.88 vs 0.10 for the
    legacy settled confidence). See recall().
  - MIGRATION: load() reads v1 float snapshots (patterns.npy) and packs
    them transparently; save() writes v2 (patterns_packed.npy). The old
    patterns.npy is left untouched and can be archived once verified.
"""
import json
import threading
from pathlib import Path

import numpy as np

if not hasattr(np, "bitwise_count"):
    raise ImportError("slate_engine v2 requires numpy >= 2.0 (np.bitwise_count)")


class SlateBank:
    def __init__(self, n_cells=10_000, dim=384, beta=60.0, seed=7,
                 distinctiveness=True, dedup_threshold=0.95, packed=True,
                 settle_floor=0.22):
        if n_cells % 8:
            raise ValueError("n_cells must be divisible by 8 for packed storage")
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((n_cells, dim)).astype(np.float32)
        self.n = n_cells
        self.beta = beta
        self.packed = packed
        # packed mode cannot apply per-cell float weights on the popcount
        # path; the weighting is benchmarked null on text (finding 5)
        self.distinctiveness = distinctiveness and not packed
        self.dedup_threshold = dedup_threshold
        # below this pre-settle familiarity no basin has captured the probe:
        # settling would be a random walk, not error correction. The junk
        # band is set by random EMBEDDING cosines (~±0.05 at 384-d), whose
        # max over 30k stored patterns reaches ~0.15 (~0.19 at 3M); genuinely
        # related text pulls 0.3+ even under heavy corruption. recall()
        # skips the settle below the floor and reports the pre-settle state
        # with cycles=0 — so junk probes get honestly LOW confidence.
        self.settle_floor = settle_floor
        self._lock = threading.Lock()
        self._nbytes = n_cells // 8                 # stored row width (I/O)
        self._wbytes = ((self._nbytes + 7) // 8) * 8  # padded to u64 words
        self._words = self._wbytes // 8
        self._cap = 1024
        self._count = 0
        self._Pb = self._Pf = self._Pb64 = self._xbuf = self._cbuf = None
        if packed:
            self._alloc_packed(self._cap)
        else:
            self._Pf = np.zeros((self._cap, n_cells), np.float32)
        self._sum = np.zeros(n_cells, np.float64)      # float mode: for weights
        self._w = None
        self._meta = []
        self._id_index = {}
        self._total_commits = 0
        self._total_deduped = 0
        self.P = None      # legacy attribute (v1 external pokes); unused in v2

    # ── encoding ─────────────────────────────────────────────────────────
    def pattern(self, emb):
        """Sign projection: embedding -> bipolar (+1/-1) float32 pattern."""
        e = np.asarray(emb, dtype=np.float32)
        return np.where(self.R @ e >= 0, 1.0, -1.0).astype(np.float32)

    def _alloc_packed(self, cap):
        """(Re)allocate the packed store + XOR/popcount scratch at capacity.

        Rows are padded to a u64 boundary (pad bytes are zero on both the
        store and the query, so they XOR to zero and never affect counts);
        popcount runs on u64 words — 8x fewer elements than per-byte.
        """
        old, old_count = self._Pb, self._count
        self._Pb = np.zeros((cap, self._wbytes), np.uint8)
        if old is not None and old_count:
            self._Pb[:old_count] = old[:old_count]
        self._Pb64 = self._Pb.view(np.uint64)
        self._xbuf = np.empty((cap, self._words), np.uint64)
        self._cbuf = np.empty((cap, self._words), np.uint8)

    def _pack(self, s_float):
        row = np.zeros(self._wbytes, np.uint8)
        row[:self._nbytes] = np.packbits(s_float > 0)
        return row

    def _unpack_rows(self, idx):
        bits = np.unpackbits(self._Pb[idx][:, :self._nbytes], axis=1, count=self.n)
        return bits.astype(np.float32) * 2.0 - 1.0

    # ── overlaps (exact in both modes) ───────────────────────────────────
    def _overlaps_packed(self, sq):
        k = self._count
        x = np.bitwise_xor(self._Pb64[:k], sq.view(np.uint64), out=self._xbuf[:k])
        np.bitwise_count(x, out=self._cbuf[:k])
        ham = self._cbuf[:k].sum(axis=1, dtype=np.int64)
        return 1.0 - (2.0 / self.n) * ham

    def _overlaps(self, s_float, weighted=True):
        if self.packed:
            return self._overlaps_packed(self._pack(s_float))
        Pv = self._Pf[:self._count]
        if weighted and self._w is not None:
            return (Pv @ (self._w * s_float)) / self.n
        return (Pv @ s_float) / self.n

    # ── internals ────────────────────────────────────────────────────────
    def _ensure_capacity(self):
        if self._count < self._cap:
            return
        self._cap *= 2
        if self.packed:
            self._alloc_packed(self._cap)
        else:
            nf = np.zeros((self._cap, self.n), np.float32)
            nf[:self._count] = self._Pf[:self._count]
            self._Pf = nf

    def _refresh_w(self):
        if not self.distinctiveness or self._count < 2:
            self._w = None
            return
        agree = np.abs(self._sum / self._count).astype(np.float32)
        w = 1.0 - agree
        s = float(w.sum())
        self._w = (w * (self.n / s)).astype(np.float32) if s > 0 else None

    def _finalize_unlocked(self):
        """v1 compat: storage is always finalized incrementally in v2.
        Refreshes distinctiveness weights in float mode; otherwise no-op."""
        self._refresh_w()

    def _meta_id(self, meta):
        return meta.get("id") if isinstance(meta, dict) else None

    def _mix(self, a):
        """Softmax-weighted pattern mix -> new bipolar state (one settle step).

        Packed mode unpacks only rows carrying non-negligible softmax mass
        (beta=60 concentrates it brutally); bounded, documented approximation
        — excluded rows have weight < max(a) * 1e-8.
        """
        if not self.packed:
            mixed = a @ self._Pf[:self._count]
        else:
            k = min(self._count, 1024)
            idx = np.argpartition(a, -k)[-k:] if self._count > k else np.arange(self._count)
            idx = idx[a[idx] > a.max() * 1e-8]
            idx = np.sort(idx)
            mixed = a[idx] @ self._unpack_rows(idx)
        return np.where(mixed >= 0, 1.0, -1.0).astype(np.float32)

    # ── write ────────────────────────────────────────────────────────────
    def commit(self, emb, meta, supersedes=None):
        """One-shot write with duplicate guard. Returns (committed, reason).

        supersedes: id of an already-stored memory this commit REPLACES in
        place. The stale attractor is overwritten, so it can never out-pull
        the correction (without this, corrections lose recall 60-80% of the
        time — slate-bench supersession benchmark). A superseding commit
        skips the dedup guard: the caller has declared intent. If the id is
        not found, the commit is stored as new and the reason says so.
        """
        s_float = self.pattern(emb)
        row_packed = self._pack(s_float) if self.packed else None
        with self._lock:
            mid = self._meta_id(meta)
            if supersedes is not None:
                idx = self._id_index.get(supersedes)
                if idx is not None:
                    if self.packed:
                        self._Pb[idx] = row_packed
                    else:
                        old_row = self._Pf[idx].copy()
                        self._Pf[idx] = s_float
                        self._sum += (s_float - old_row).astype(np.float64)
                        self._refresh_w()
                    old_id = self._meta_id(self._meta[idx])
                    if old_id is not None and old_id in self._id_index:
                        del self._id_index[old_id]
                    self._meta[idx] = meta
                    if mid is not None:
                        self._id_index[mid] = idx
                    self._total_commits += 1
                    return True, f"superseded '{supersedes}'"
            if self._count and self.dedup_threshold < 1.0:
                o = (self._overlaps_packed(row_packed) if self.packed
                     else (self._Pf[:self._count] @ s_float) / self.n)
                best_i = int(np.argmax(o))
                best = float(o[best_i])
                if best >= self.dedup_threshold:
                    self._total_deduped += 1
                    return False, (f"duplicate (overlap {best:.3f} with "
                                   f"'{self._meta_id(self._meta[best_i]) or best_i}')")
            self._ensure_capacity()
            if self.packed:
                self._Pb[self._count] = row_packed
            else:
                self._Pf[self._count] = s_float
                self._sum += s_float.astype(np.float64)
            self._meta.append(meta)
            if mid is not None:
                self._id_index[mid] = self._count
            self._count += 1
            if not self.packed:
                self._refresh_w()
            self._total_commits += 1
            if supersedes is not None:
                return True, (f"committed (supersedes id '{supersedes}' "
                              f"not found; stored as new)")
            return True, "committed"

    # ── read ─────────────────────────────────────────────────────────────
    def recall(self, emb, topk=5, max_cycles=5):
        """Returns (winner_meta, ranked_topk, confidence, cycles, signals).

        signals = {"familiarity": pre-settle max overlap,
                   "margin":      pre-settle top1 - top2 overlap gap}

        Trust the MARGIN as the retrieval-quality signal (correct-vs-wrong
        AUC 0.88, slate-bench limits battery 2026-07-12). `confidence` is
        the post-settle winner overlap — it measures settle DEPTH, not
        correctness (AUC 0.10, inverted: the state lands at the bottom of
        SOME attractor at ~1.0 whether or not it is the right one). It is
        kept for backward compatibility and convergence diagnostics only.

        When familiarity < settle_floor, no basin has captured the probe:
        the settle is skipped (cycles=0) and confidence equals the
        pre-settle winner overlap — so out-of-domain probes now report
        honestly LOW confidence instead of settling into a random basin.
        """
        with self._lock:
            if self._count == 0:
                return None, [], 0.0, 0, {"familiarity": 0.0, "margin": 0.0}
            s = self.pattern(emb)
            o0 = self._overlaps(s)
            if o0.size >= 2:
                two = np.partition(o0, -2)[-2:]
                signals = {"familiarity": float(two[-1]),
                           "margin": float(two[-1] - two[-2])}
            else:
                signals = {"familiarity": float(o0[0]), "margin": float(o0[0])}
            if signals["familiarity"] < self.settle_floor:
                winner = int(np.argmax(o0))
                order = [int(i) for i in np.argsort(-o0)[:topk]]
                if winner in order:
                    order.remove(winner)
                order = ([winner] + order)[:topk]
                ranked = [self._meta[i] for i in order]
                return self._meta[winner], ranked, float(o0[winner]), 0, signals
            o = o0
            cycles = 0
            converged = False
            for _ in range(max_cycles):
                if cycles > 0:
                    o = self._overlaps(s)
                a = np.exp(self.beta * (o - o.max()))
                a /= a.sum()
                s_new = self._mix(a)
                cycles += 1
                if np.array_equal(s_new, s):
                    converged = True    # o was computed for this exact state
                    break
                s = s_new
            o_final = o if converged else self._overlaps(s)
            winner = int(np.argmax(o_final))
            order = [int(i) for i in np.argsort(-o0)[:topk]]
            if winner in order:
                order.remove(winner)
            order = ([winner] + order)[:topk]
            ranked = [self._meta[i] for i in order]
            return self._meta[winner], ranked, float(o_final[winner]), cycles, signals

    def familiar(self, emb):
        """Quick familiarity check without full settle. Returns (best_meta, score)."""
        with self._lock:
            if self._count == 0:
                return None, 0.0
            o = self._overlaps(self.pattern(emb))
            best = int(np.argmax(o))
            return self._meta[best], float(o[best])

    def count(self):
        with self._lock:
            return self._count

    def list_entries(self, limit=50):
        with self._lock:
            return [m for m in self._meta[:limit]]

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, data_dir):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._count:
                if self.packed:
                    np.save(str(data_dir / "patterns_packed.npy"),
                            self._Pb[:self._count, :self._nbytes])
                else:
                    np.save(str(data_dir / "patterns.npy"),
                            self._Pf[:self._count])
            with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=1)
            state = {
                "format": 2,
                "packed": self.packed,
                "count": self._count,
                "total_commits": self._total_commits,
                "total_deduped": self._total_deduped,
                "n_cells": self.n,
                "beta": self.beta,
                "seed": 7,
                "distinctiveness": self.distinctiveness,
                "dedup_threshold": self.dedup_threshold,
            }
            with open(data_dir / "state.json", "w") as f:
                json.dump(state, f, indent=2)

    def load(self, data_dir):
        """Load a v2 (patterns_packed.npy) or v1 (patterns.npy float32)
        snapshot; v1 rows are packed transparently on the way in. Returns
        the number of patterns loaded."""
        data_dir = Path(data_dir)
        packed_path = data_dir / "patterns_packed.npy"
        float_path = data_dir / "patterns.npy"
        meta_path = data_dir / "meta.json"
        if not meta_path.exists() or not (packed_path.exists() or float_path.exists()):
            return 0
        use_packed = packed_path.exists()
        if use_packed and float_path.exists():
            # both formats present (post-migration dir): trust the newer
            # write — protects against a rolled-back v1 process having
            # saved patterns.npy after v2 wrote the packed snapshot
            use_packed = packed_path.stat().st_mtime >= float_path.stat().st_mtime
        with self._lock:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if use_packed:
                Pb = np.load(str(packed_path))
                n_rows = min(len(Pb), len(meta))
                src_packed, src = True, Pb[:n_rows]
            else:
                Pf = np.load(str(float_path))
                n_rows = min(len(Pf), len(meta))
                src_packed, src = False, Pf[:n_rows]
            self._cap = max(1024, 1 << int(np.ceil(np.log2(max(n_rows, 2)))))
            self._sum = np.zeros(self.n, np.float64)
            if self.packed:
                self._count = 0                  # nothing to carry over
                self._alloc_packed(self._cap)    # rebuilds u64 view + scratch
                if src_packed:
                    self._Pb[:n_rows, :self._nbytes] = src
                else:
                    for i0 in range(0, n_rows, 4096):       # v1 -> v2 migration
                        j = min(i0 + 4096, n_rows)
                        self._Pb[i0:j, :self._nbytes] = np.packbits(src[i0:j] > 0, axis=1)
                self._Pf = None
            else:
                self._Pf = np.zeros((self._cap, self.n), np.float32)
                if src_packed:
                    for i0 in range(0, n_rows, 4096):
                        j = min(i0 + 4096, n_rows)
                        bits = np.unpackbits(src[i0:j], axis=1, count=self.n)
                        self._Pf[i0:j] = bits.astype(np.float32) * 2.0 - 1.0
                else:
                    self._Pf[:n_rows] = src
                self._sum = self._Pf[:n_rows].sum(axis=0, dtype=np.float64)
                self._Pb = None
            self._meta = meta[:n_rows]
            self._id_index = {}
            for i, m in enumerate(self._meta):
                mid = self._meta_id(m)
                if mid is not None:
                    self._id_index[mid] = i
            self._count = n_rows
            state_path = data_dir / "state.json"
            if state_path.exists():
                with open(state_path) as f:
                    st = json.load(f)
                self._total_commits = st.get("total_commits", n_rows)
                self._total_deduped = st.get("total_deduped", 0)
            self._refresh_w()
            return n_rows
