"""slate-memory — one-shot attractor memory for LLMs and agents.

    from slate_memory import SlateBank

    bank = SlateBank()
    bank.commit(embedding, {"text": "Paris is the capital of France"})
    winner, top_k, confidence, cycles = bank.recall(query_embedding)

    # trustworthy retrieval-quality signal (see recall docstring):
    winner, top_k, conf, cycles, sig = bank.recall(q, with_signals=True)
    if sig["margin"] < 0.02:
        ...treat as "don't know" rather than trusting the winner...
"""

from slate_memory.bank import SlateBank

__all__ = ["SlateBank"]
__version__ = "0.2.0"
