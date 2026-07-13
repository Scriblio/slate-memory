# Slate Memory

**Claude forgets everything between sessions. Slate fixes that.**

An MCP memory server and Python library with error-correcting attractor dynamics. Ask a sloppy question with typos and missing words — Slate still finds the right answer.

## The numbers

| Feature | Basic vector memory | Slate |
|---|---|---|
| Typo in your query | Accuracy drops 17% | Drops 2% |
| Half the words missing | Loses 42% of memories | Loses 28% |
| Speed (30K memories) | ~160ms | ~39ms |
| Storage per memory | 1.5KB (float32) | 48 bytes (packed binary) |

Numbers from head-to-head benchmarks on the same 30K-memory dataset. Patent pending (#64/109,622).

| Setup | Accuracy | Cost per 1k queries | Latency |
|---|---|---|---|
| **haiku + slate** | **100%** | **$0.10** | **0.6s** |
| opus + full context | 100% | $64.04 | 1.7s |
| opus bare | 0% | $0.19 | 1.2s |
| haiku bare | 0% | $0.04 | 0.7s |

A cheap model with Slate **ties the most expensive model with the entire corpus in context** — at 1/640th the cost. Full methodology in [slate-bench](https://github.com/Scriblio/slate-bench).

## Use as MCP server (Claude Desktop / Claude Code)

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "slate-memory": {
      "command": "python",
      "args": ["/path/to/slate-memory/run_mcp.py"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add slate-memory python /path/to/slate-memory/run_mcp.py
```

### MCP Tools

- **`slate_recall`** — Ask a question, get the matching memory. Works even with typos, partial queries, or vague phrasing.
- **`slate_commit`** — Store a new memory. Automatic deduplication.
- **`slate_status`** — Check how many memories are stored and system health.

### MCP Requirements

```bash
pip install slate-memory[mcp]
```

## Use as Python library

```bash
pip install slate-memory
```

```python
from slate_memory import SlateBank

bank = SlateBank(dim=384)

# Commit facts one-shot
bank.commit(embed("The capital of France is Paris"), {"text": "Paris is the capital"})

# Recall — even from a noisy or partial query
winner, ranked, confidence, cycles = bank.recall(embed("whats the captial of farnce"))
print(winner["text"])  # "Paris is the capital"
```

### With sentence-transformers

```python
from slate_memory import SlateBank
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
bank = SlateBank(dim=384)

texts = ["The Earth orbits the Sun", "Water boils at 100C", "Light travels at 299,792 km/s"]
for text in texts:
    bank.commit(model.encode(text), {"text": text})

winner, ranked, confidence, cycles = bank.recall(model.encode("what temperature does water boil"))
print(winner["text"])  # "Water boils at 100C"
```

### Persistence

```python
bank.save("./my_memory")

bank2 = SlateBank(dim=384)
bank2.load("./my_memory")
```

Two files: `patterns.npy` + `meta.json`. No server. No database. Copy them anywhere.

## How it works

Slate is a [modern Hopfield network](https://arxiv.org/abs/2008.02217) (Ramsauer et al. 2020) — the same math as transformer attention, but used as a memory instead of a layer. Embeddings are sign-projected onto 10,000 bipolar cells and stored one-shot. On recall, the query settles into the nearest stored attractor via softmax-weighted feedback. Two cycles. No training. No index. No database.

The attractor dynamics error-correct: noisy, partial, or corrupted queries converge to the exact stored pattern. Verified at 20,000 stored patterns with 25% corruption.

## Trust signals

Each recall returns quality signals so you know how confident the answer is:

```python
winner, ranked, conf, cycles, sig = bank.recall(query, with_signals=True)
if sig["margin"] < 0.02:
    # thin margin — treat as "don't know"
    pass
```

- **margin** — gap between the best and second-best match. The calibrated retrieval-quality signal (AUC 0.88). Wide margin = confident memory. Thin margin = guess.
- **familiarity** — how well the query matches anything stored. Low = topic was never stored (AUC 1.00 for out-of-domain).

## API

### `SlateBank(dim, n_cells=10000, beta=60.0, distinctiveness=True, dedup_threshold=0.95, seed=7)`

- **dim**: Embedding dimensionality (must match your embedder)
- **n_cells**: Bipolar cells. 10,000 verified for up to 20K patterns
- **beta**: Winner-take-all sharpness. 60.0 is benchmarked
- **dedup_threshold**: Refuse commits with overlap above this. 0.95 prevents near-duplicates

### `bank.commit(embedding, meta) -> (bool, str)`

One-shot storage. Returns `(True, "committed")` or `(False, "duplicate (...)")`.

### `bank.recall(embedding, top_k=3, max_cycles=5, with_scores=False, with_signals=False)`

Full attractor settle. Returns `(winner, ranked, confidence, cycles[, signals])`.

### `bank.remove(predicate) -> int`

Remove patterns matching `predicate(meta)`. For re-ingestion when source data changed.

### `bank.save(path)` / `bank.load(path)`

Persist to / restore from a directory.

## Benchmarks

Full retrieval benchmarks across 5 corruption conditions and 4 capacity levels in [slate-bench](https://github.com/Scriblio/slate-bench).

## License

Apache 2.0. Patent pending.

<!-- mcp-name: io.github.scriblio/slate-memory -->
