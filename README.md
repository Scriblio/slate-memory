# slate-memory

One-shot attractor memory for LLMs. Vector-search accuracy at 98% fewer tokens.

## The number

| Setup | Accuracy | Cost per 1k queries | Latency |
|---|---|---|---|
| **haiku + slate** | **100%** | **$0.10** | **0.6s** |
| opus + full context | 100% | $64.04 | 1.7s |
| opus bare | 0% | $0.19 | 1.2s |
| haiku bare | 0% | $0.04 | 0.7s |

A cheap model with slate-memory **ties the most expensive model with the entire corpus in context** — at 1/640th the cost. Both models score 0% without memory (facts are synthetic, unknowable parametrically). Benchmark: N=25 questions, K=600 stored facts, hermetic SDK calls. Full methodology and reproduction scripts in [slate-bench](https://github.com/Scriblio/slate-bench).

## How it works

Slate-memory is a [modern Hopfield network](https://arxiv.org/abs/2008.02217) (Ramsauer et al. 2020) — the same math as transformer attention, but used as a memory instead of a layer. Embeddings are sign-projected onto 10,000 bipolar cells (SimHash) and stored one-shot. On recall, the query settles into the nearest stored attractor via softmax-weighted feedback. Two cycles. No training. No index. No database.

The attractor dynamics error-correct: noisy, partial, or corrupted queries converge to the exact stored pattern. Verified at 20,000 stored patterns with 25% corruption.

## Install

```bash
pip install slate-memory
```

For the built-in embedder convenience wrapper:

```bash
pip install slate-memory[embed]
```

## Quick start

```python
from slate_memory import SlateBank
import numpy as np

# Your embeddings (384-dim for MiniLM, 1536 for OpenAI, etc.)
bank = SlateBank(dim=384)

# Commit facts one-shot
bank.commit(embed("The capital of France is Paris"), {"text": "Paris is the capital of France", "id": "fact-1"})
bank.commit(embed("Python was created by Guido van Rossum"), {"text": "Guido created Python", "id": "fact-2"})

# Recall — even from a noisy or partial query
winner, ranked, confidence, cycles = bank.recall(embed("What's the capital of France?"))
print(winner)  # {"text": "Paris is the capital of France", "id": "fact-1"}
print(f"Confidence: {confidence:.3f}, settled in {cycles} cycles")
```

### With sentence-transformers

```python
from slate_memory import SlateBank
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
bank = SlateBank(dim=384)

# Commit
texts = ["The Earth orbits the Sun", "Water boils at 100°C", "Light travels at 299,792 km/s"]
for text in texts:
    bank.commit(model.encode(text), {"text": text})

# Recall
query = model.encode("what temperature does water boil")
winner, ranked, confidence, cycles = bank.recall(query)
print(winner["text"])  # "Water boils at 100°C"
```

### With OpenAI embeddings

```python
from slate_memory import SlateBank
from openai import OpenAI

client = OpenAI()
bank = SlateBank(dim=1536)  # text-embedding-3-small

def embed(text):
    return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

bank.commit(embed("Revenue was $4.2M in Q3"), {"text": "Q3 revenue: $4.2M"})
winner, _, conf, _ = bank.recall(embed("how much revenue in Q3"))
```

## Persistence

```python
# Save
bank.save("./my_memory")

# Load
bank2 = SlateBank(dim=384)
n = bank2.load("./my_memory")
print(f"Loaded {n} patterns")
```

The entire memory is two files: `patterns.npy` (the attractor states) and `meta.json` (your metadata). No server. No database. Copy them anywhere.

## Trust signals

`confidence` measures how deeply recall settled — **not whether the winner is
right**. Settling reaches the bottom of *some* attractor at ~1.0 whether or
not it's the right one (correct-vs-wrong AUC 0.10 in the limits battery). For
trust decisions, ask for signals:

```python
winner, ranked, conf, cycles, sig = bank.recall(query, with_signals=True)
if sig["margin"] < 0.02:
    # thin margin: the winner is a guess, not a memory
    answer_i_dont_know()
```

- **margin** — gap between the best and second-best overlap, computed before
  settling. The calibrated retrieval-quality signal (AUC 0.88): wide margin
  means a confident memory, thin margin means a guess.
- **familiarity** — the best overlap itself. Low values flag out-of-domain
  queries (AUC 1.00); it does *not* reliably detect a plausible-but-absent
  fact (AUC ~0.62) — that's what margin is for.

Both come from work already done during recall — no extra cost.

## API

### `SlateBank(dim, n_cells=10000, beta=60.0, distinctiveness=True, dedup_threshold=0.95, seed=7)`

- **dim**: Embedding dimensionality (must match your embedder)
- **n_cells**: Bipolar cells in the attractor state. 10,000 is verified for up to 20k patterns
- **beta**: Winner-take-all sharpness. 60.0 is benchmarked; don't change without reason
- **distinctiveness**: Down-weight cells where all patterns agree. Essential for images; neutral for text
- **dedup_threshold**: Refuse commits with overlap above this. 0.95 prevents near-duplicates
- **seed**: RNG seed for the projection matrix. Same seed = same projections = portable patterns

### `bank.commit(embedding, meta) → (bool, str)`

One-shot storage. Returns `(True, "committed")` or `(False, "duplicate (...)")`.

### `bank.recall(embedding, top_k=3, max_cycles=5, with_scores=False, with_signals=False) → (winner, ranked, confidence, cycles[, signals])`

Full attractor settle. Returns the winner's metadata, top-k ranked results, confidence score, and settle cycles. `with_scores=True` makes `ranked` a list of `(meta, score)` pairs (pre-settle overlaps). `with_signals=True` appends a fifth element `{"familiarity", "margin"}` — see [Trust signals](#trust-signals).

### `bank.remove(predicate) → int`

Remove every pattern whose meta matches `predicate(meta)`. For re-ingestion flows where a source document changed:

```python
bank.remove(lambda m: m.get("chapter_id") == chapter_id)
```

### `bank.familiar(embedding) → (meta, score)`

Quick overlap check without settling. Use for "have I seen this before?" checks.

### `bank.save(path)` / `bank.load(path)`

Persist to / restore from a directory.

## Benchmarks

Full retrieval benchmarks across 5 corruption conditions and 4 capacity levels show accuracy parity with exact cosine search (±1 point). The attractor substrate recalls 20,000 random patterns perfectly at 25% corruption. Capacity is limited by embedding discriminability (the embedder), not the attractor.

Honest limitations in software: per-query latency is ~35x slower than exact vector search (19ms vs 0.5ms at K=2000) because the projected patterns are 10,000-d vs 384-d. The speed case is the [photonic implementation](https://en.wikipedia.org/wiki/Optical_computing) where recall is one pass of light — that's the roadmap, not the demo.

See [slate-bench](https://github.com/Scriblio/slate-bench) for full reproduction scripts and results.

## License

Apache 2.0. Patent pending.
