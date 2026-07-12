# How I Made the Cheapest Model Match the Best — at 1/640th the Cost

I built a memory system that lets Claude Haiku (the $1/M-token model) answer questions with **100% accuracy** — tying Claude Opus (the $5/M-token model) running with the entire knowledge base in context. Haiku + my memory costs **$0.10 per thousand questions**. Opus + full context costs **$64.04**. Same accuracy. 640x cheaper.

Both models score **0%** without memory. The facts are synthetic — there's nothing in their training data to fall back on. The memory is the entire difference.

I'm open-sourcing the system today. `pip install slate-memory`.

## What it is

Slate-memory is a one-shot attractor memory. You commit facts by embedding them once. When you query, the system settles into the nearest stored pattern via softmax-weighted feedback — the same math as transformer attention, but used as a lookup table instead of a layer.

It's a [modern Hopfield network](https://arxiv.org/abs/2008.02217) (Ramsauer et al. 2020), but engineered for production: persistence, dedup, thread safety, and support for any embedding model.

```python
from slate_memory import SlateBank
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
bank = SlateBank(dim=384)

# Commit facts one-shot
bank.commit(model.encode("Revenue was $4.2M in Q3"), {"text": "Q3 revenue: $4.2M"})

# Recall — even from a noisy query
winner, _, confidence, _ = bank.recall(model.encode("how much revenue last quarter"))
print(winner["text"])  # "Q3 revenue: $4.2M"
```

## The benchmark

I tested against the standard approach: exact cosine vector search (the ceiling for what Pinecone/Weaviate/pgvector return). Five corruption conditions, four capacity levels, 300 queries per condition.

| Condition | Slate | Vector | Difference |
|---|---|---|---|
| Clean query | 88.7% | 90.0% | -1.3 |
| 30% word dropout | 76.3% | 77.0% | -0.7 |
| 50% word dropout | 47.7% | 46.7% | +1.0 |
| 15% character typos | 55.7% | 57.7% | -2.0 |
| Keywords only | 88.3% | 90.3% | -2.0 |

**Accuracy parity.** The attractor loses at most 2 points anywhere, and at top-3 retrieval they're essentially identical.

## The economics

The real pitch isn't accuracy — it's what happens to your token bill.

When you use RAG (or slate), you send the model ~75 tokens per question (the retrieved fact + the question). When you stuff context, you send thousands. At frontier pricing:

- **75 tokens/question (retrieval)**: $0.23 per thousand questions
- **4,023 tokens/question (4k context)**: $12.07 per thousand questions
- **32k context**: ~$96 per thousand questions

That's a 98.1% reduction in prompt tokens at equal accuracy. The saving gets more dramatic as the knowledge base grows — context stuffing scales linearly with corpus size; retrieval doesn't.

## Where it's honest

**Latency.** In pure software, exact vector search is 35x faster per query (0.5ms vs 19ms). Slate patterns are 10,000-dimensional; embeddings are 384-d. Numpy pays for the expansion.

**Why it doesn't matter much.** 19ms is still fast enough for any LLM pipeline (the LLM call itself takes 500-2000ms). And the 35x gap is a software artifact — in the photonic implementation (on the roadmap), the compare-against-everything step is one pass of light.

**Distinctiveness weighting.** Tested; null effect on text (±0.6 points). It matters for visual patterns where stored items share literal pixel regions, but text embeddings are already decorrelated by SimHash.

## Why not just use a vector database?

You can. Slate matches vector search accuracy. The differences are:

1. **Zero infrastructure.** Your memory is two files: `patterns.npy` and `meta.json`. No server, no index, no HNSW tuning, no connection string.
2. **Error correction.** The attractor dynamics converge noisy queries to clean stored patterns. Vector search returns whatever's closest; slate *settles* into the right answer.
3. **One-shot writes.** Commit a fact and it's immediately recallable. No re-indexing, no batch upserts.
4. **Portable.** Copy two files. That's your entire memory. Load them on any machine with numpy.

## Where it came from

I built this as a memory organ for an AI agent — literally a felt-recall system where visual moments and conversations are committed one-shot and recognized later. The same math runs motor control for a 3D character (attractor chains of body poses — show it a motion once and it converges from any starting position). Four live deployments, all on my laptop.

The LLM memory layer is the productized version of the same core.

## Try it

```bash
pip install slate-memory
```

The entire codebase is ~200 lines of numpy. No frameworks, no dependencies beyond numpy. Add `slate-memory[embed]` if you want built-in sentence-transformers support.

Full benchmark reproduction: [slate-bench](https://github.com/Scriblio/slate-bench)

Patent pending. Apache 2.0 license.
