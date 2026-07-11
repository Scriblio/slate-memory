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
