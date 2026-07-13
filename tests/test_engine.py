"""Engine tests: signals, settle floor, supersedes, dedup, persistence,
v1-float migration. Run: python -m pytest tests/ -q"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slate_engine import SlateBank  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture(params=[True, False], ids=["packed", "float"])
def bank(request):
    return SlateBank(packed=request.param,
                     distinctiveness=not request.param)


def _fill(bank, rng, k=50):
    embs = rng.standard_normal((k, 384)).astype(np.float32)
    for i, e in enumerate(embs):
        bank.commit(e, {"id": f"m{i}", "content": f"fact {i}"})
    return embs


def test_recall_signals_and_shapes(bank, rng):
    out = bank.recall(rng.standard_normal(384))
    assert out[0] is None and out[4] == {"familiarity": 0.0, "margin": 0.0}

    embs = _fill(bank, rng)
    w, ranked, conf, cycles, sig = bank.recall(embs[7])
    assert w["id"] == "m7"
    assert sig["familiarity"] > 0.9 and sig["margin"] > 0.3
    assert conf > 0.9


def test_settle_floor_skips_junk(bank, rng):
    embs = _fill(bank, rng)
    junk = rng.standard_normal(384).astype(np.float32)
    w, _, conf, cycles, sig = bank.recall(junk)
    assert sig["familiarity"] < bank.settle_floor
    assert cycles == 0                      # no basin captured -> no settle
    assert conf < 0.5                       # junk reports honestly low conf
    # real probes still settle
    _, _, _, cycles, _ = bank.recall(embs[3])
    assert cycles >= 1


def test_supersedes(bank, rng):
    embs = _fill(bank, rng)
    n0 = bank.count()
    correction = embs[7] + 0.15 * rng.standard_normal(384).astype(np.float32)
    ok, reason = bank.commit(correction, {"id": "m7v2"}, supersedes="m7")
    assert ok and "superseded" in reason and bank.count() == n0
    w, _, _, _, _ = bank.recall(embs[7])
    assert w["id"] == "m7v2"                # stale attractor is gone
    # near-identical superseding recommit skips the dedup guard
    ok, reason = bank.commit(correction, {"id": "m7v3"}, supersedes="m7v2")
    assert ok and "superseded" in reason
    # unknown id -> stored as new, honest reason
    ok, reason = bank.commit(embs[1] * 0.9 + embs[2] * 0.1, {"id": "x"},
                             supersedes="never-existed")
    assert ok and "not found" in reason and bank.count() == n0 + 1


def test_dedup_guard(bank, rng):
    embs = _fill(bank, rng)
    ok, reason = bank.commit(embs[3], {"id": "dup"})
    assert not ok and "duplicate" in reason


def test_save_load_roundtrip(bank, rng):
    embs = _fill(bank, rng)
    with tempfile.TemporaryDirectory() as d:
        bank.save(d)
        fresh = SlateBank(packed=bank.packed,
                          distinctiveness=bank.distinctiveness)
        assert fresh.load(d) == bank.count()
        w1 = bank.recall(embs[5])[0]["id"]
        w2 = fresh.recall(embs[5])[0]["id"]
        assert w1 == w2 == "m5"


def test_v1_float_snapshot_migration(rng):
    """A v1-era snapshot (patterns.npy float32 ±1) loads into packed mode."""
    src = SlateBank(packed=False, distinctiveness=False)
    embs = _fill(src, rng, k=30)
    with tempfile.TemporaryDirectory() as d:
        src.save(d)                          # float mode writes patterns.npy
        assert (Path(d) / "patterns.npy").exists()
        mig = SlateBank(packed=True)
        assert mig.load(d) == 30
        for i in (0, 7, 29):
            assert mig.recall(embs[i])[0]["id"] == f"m{i}"
        # growth after migration works (capacity/scratch rebuilt correctly)
        ok, _ = mig.commit(rng.standard_normal(384).astype(np.float32),
                           {"id": "post-migration"})
        assert ok and mig.count() == 31
