"""Tests for SlateBank — the core attractor memory."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from slate_memory import SlateBank


@pytest.fixture
def bank():
    return SlateBank(n_cells=1000, dim=64, seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(99)


def _random_emb(rng, dim=64):
    e = rng.standard_normal(dim).astype(np.float32)
    return e / np.linalg.norm(e)


def test_commit_and_recall(bank, rng):
    emb = _random_emb(rng)
    ok, reason = bank.commit(emb, {"id": "a"})
    assert ok
    assert reason == "committed"
    assert bank.count == 1

    winner, ranked, conf, cycles = bank.recall(emb)
    assert winner["id"] == "a"
    assert conf > 0.9


def test_recall_empty(bank, rng):
    winner, ranked, conf, cycles = bank.recall(_random_emb(rng))
    assert winner is None
    assert ranked == []


def test_noisy_recall(bank, rng):
    emb = _random_emb(rng)
    bank.commit(emb, {"id": "original"})
    noisy = emb + rng.standard_normal(64).astype(np.float32) * 0.3
    winner, _, conf, _ = bank.recall(noisy)
    assert winner["id"] == "original"


def test_multiple_patterns(bank, rng):
    embs = [_random_emb(rng) for _ in range(20)]
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"fact-{i}"})
    assert bank.count == 20
    for i, e in enumerate(embs):
        winner, _, _, _ = bank.recall(e)
        assert winner["id"] == f"fact-{i}"


def test_dedup_guard(bank, rng):
    emb = _random_emb(rng)
    ok1, _ = bank.commit(emb, {"id": "first"})
    ok2, reason = bank.commit(emb, {"id": "second"})
    assert ok1
    assert not ok2
    assert "duplicate" in reason
    assert bank.count == 1


def test_familiar(bank, rng):
    emb = _random_emb(rng)
    meta, score = bank.familiar(emb)
    assert meta is None
    assert score == 0.0

    bank.commit(emb, {"id": "x"})
    meta, score = bank.familiar(emb)
    assert meta["id"] == "x"
    assert score > 0.9


def test_save_load(bank, rng):
    for i in range(5):
        bank.commit(_random_emb(rng), {"id": f"p-{i}"})

    with tempfile.TemporaryDirectory() as d:
        bank.save(d)
        assert (Path(d) / "patterns.npy").exists()
        assert (Path(d) / "meta.json").exists()

        bank2 = SlateBank(n_cells=1000, dim=64, seed=42)
        n = bank2.load(d)
        assert n == 5
        assert bank2.count == 5


def test_persistence_recall_matches(bank, rng):
    embs = [_random_emb(rng) for _ in range(10)]
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"r-{i}"})

    with tempfile.TemporaryDirectory() as d:
        bank.save(d)
        bank2 = SlateBank(n_cells=1000, dim=64, seed=42)
        bank2.load(d)
        for i, e in enumerate(embs):
            w1, _, c1, _ = bank.recall(e)
            w2, _, c2, _ = bank2.recall(e)
            assert w1["id"] == w2["id"]


def test_commit_batch(bank, rng):
    embs = [_random_emb(rng) for _ in range(5)]
    metas = [{"id": f"b-{i}"} for i in range(5)]
    results = bank.commit_batch(embs, metas)
    assert all(ok for ok, _ in results)
    assert bank.count == 5


def test_stats(bank, rng):
    bank.commit(_random_emb(rng), {"id": "s"})
    s = bank.stats()
    assert s["count"] == 1
    assert s["n_cells"] == 1000
    assert s["dim"] == 64


def test_different_dims():
    for dim in [384, 768, 1536]:
        b = SlateBank(dim=dim, n_cells=500)
        rng = np.random.default_rng(0)
        e = rng.standard_normal(dim).astype(np.float32)
        ok, _ = b.commit(e, {"dim": dim})
        assert ok
        w, _, c, _ = b.recall(e)
        assert w["dim"] == dim
        assert c > 0.8


def test_recall_with_scores(bank, rng):
    embs = [_random_emb(rng) for _ in range(5)]
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"fact-{i}"})

    winner, ranked, conf, _ = bank.recall(embs[2], top_k=3, with_scores=True)
    assert winner["id"] == "fact-2"
    assert len(ranked) == 3
    meta, score = ranked[0]
    assert meta["id"] == "fact-2"          # winner first
    assert score > 0.9                     # exact match ≈ 1.0
    assert all(-1.5 <= s <= 1.5 for _, s in ranked)  # weighted overlap can slightly exceed ±1
    assert ranked[0][1] >= ranked[1][1]    # non-winner tail sorted by overlap

    # default shape unchanged: bare metas
    _, ranked_plain, _, _ = bank.recall(embs[2], top_k=3)
    assert isinstance(ranked_plain[0], dict)


def test_recall_with_signals(bank, rng):
    embs = [_random_emb(rng) for _ in range(10)]
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"fact-{i}"})

    # exact probe: high familiarity, healthy margin
    winner, ranked, conf, cycles, sig = bank.recall(embs[3], with_signals=True)
    assert winner["id"] == "fact-3"
    assert sig["familiarity"] > 0.9
    assert sig["margin"] > 0.3

    # junk probe: margin collapses even when settled confidence stays high
    junk = _random_emb(rng)
    _, _, _, _, sig_junk = bank.recall(junk, with_signals=True)
    assert sig_junk["margin"] < sig["margin"]
    assert sig_junk["familiarity"] < 0.5

    # default shape unchanged: 4-tuple, and composes with with_scores
    assert len(bank.recall(embs[3])) == 4
    out = bank.recall(embs[3], with_scores=True, with_signals=True)
    assert len(out) == 5 and isinstance(out[1][0], tuple)


def test_recall_with_signals_empty_and_single(bank, rng):
    out = bank.recall(_random_emb(rng), with_signals=True)
    assert len(out) == 5
    assert out[4] == {"familiarity": 0.0, "margin": 0.0}

    e = _random_emb(rng)
    bank.commit(e, {"id": "solo"})
    w, _, _, _, sig = bank.recall(e, with_signals=True)
    assert w["id"] == "solo"
    assert np.isfinite(sig["margin"]) and np.isfinite(sig["familiarity"])


def test_remove(bank, rng):
    embs = [_random_emb(rng) for _ in range(6)]
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"p-{i}", "chapter": "one" if i < 4 else "two"})

    removed = bank.remove(lambda m: m["chapter"] == "one")
    assert removed == 4
    assert bank.count == 2

    # removed patterns no longer win recall
    winner, _, _, _ = bank.recall(embs[0])
    assert winner["chapter"] == "two"

    # survivors still recall exactly
    winner, _, conf, _ = bank.recall(embs[5])
    assert winner["id"] == "p-5" and conf > 0.9

    # no-match predicate is a no-op
    assert bank.remove(lambda m: m.get("chapter") == "nine") == 0
    assert bank.count == 2


def test_remove_then_save_load(bank, rng):
    embs = [_random_emb(rng) for _ in range(4)]
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"p-{i}"})
    bank.remove(lambda m: m["id"] == "p-1")

    with tempfile.TemporaryDirectory() as d:
        bank.save(d)
        fresh = SlateBank(n_cells=1000, dim=64, seed=42)
        assert fresh.load(d) == 3
        winner, _, _, _ = fresh.recall(embs[2])
        assert winner["id"] == "p-2"
