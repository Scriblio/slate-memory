"""slate-memory — one-shot attractor memory for LLMs and agents.

    from slate_memory import SlateBank

    bank = SlateBank()
    bank.commit(embedding, {"text": "Paris is the capital of France"})
    winner, top_k, confidence, cycles = bank.recall(query_embedding)
"""

from slate_memory.bank import SlateBank

__all__ = ["SlateBank"]
__version__ = "0.1.1"
