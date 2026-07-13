"""Slate MCP Server — attractor memory as a recall front-end for FMS.

Tools:
  - slate_recall: Settle a query into the nearest stored attractor
  - slate_commit: One-shot commit text (with duplicate guard)
  - slate_status: Stats and health
  - slate_sync:   Bulk-sync from FMS database

FMS remains the store-of-record. Slate is the fast recall front-end:
attractor dynamics error-correct noisy queries and converge in 2 cycles.
"""
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent))

from slate_engine import SlateBank
from embedder import embed, embed_batch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

DATA_DIR = os.environ.get("SLATE_DATA_DIR", str(Path(__file__).parent / "data"))
FMS_DB = os.environ.get(
    "SLATE_FMS_DB",
    r"C:\Users\matth\fractal-memory-data\memory.db"
)

mcp = FastMCP("slate-memory")
_bank: SlateBank | None = None


def _get_bank() -> SlateBank:
    global _bank
    if _bank is None:
        # packed: 32x less RAM + faster popcount overlaps. Packed mode drops
        # distinctiveness weighting — benchmarked null on text (finding 5).
        _bank = SlateBank(
            n_cells=10_000, dim=384, beta=60.0, seed=7,
            distinctiveness=False, dedup_threshold=0.95, packed=True
        )
        loaded = _bank.load(DATA_DIR)
        if loaded:
            logger.info("Loaded %d patterns from disk", loaded)
        else:
            logger.info("No saved patterns — starting empty")
    return _bank


def _sync_from_fms(bank: SlateBank, limit: int = 0, owner: str | None = None):
    """Bulk-commit FMS memory nodes into the Slate."""
    if not os.path.exists(FMS_DB):
        return 0, f"FMS DB not found at {FMS_DB}"

    conn = sqlite3.connect(FMS_DB)
    conn.row_factory = sqlite3.Row
    try:
        where = "WHERE is_archived = 0"
        params = []
        if owner:
            where += " AND owner = ?"
            params.append(owner)
        order = "ORDER BY created_at DESC"
        lim = f"LIMIT {limit}" if limit > 0 else ""
        rows = conn.execute(
            f"SELECT id, content, tags, source, importance, agent, owner, created_at "
            f"FROM memory_nodes {where} {order} {lim}",
            params
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0, "no matching rows"

    texts = [r["content"] for r in rows]
    t0 = time.time()
    embeddings = embed_batch(texts)
    embed_sec = time.time() - t0

    committed = 0
    for row, emb in zip(rows, embeddings):
        tags = row["tags"]
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = [tags]
        meta = {
            "id": row["id"],
            "content": row["content"][:500],
            "tags": tags or [],
            "source": row["source"],
            "importance": row["importance"],
            "agent": row["agent"],
            "owner": row["owner"],
            "created_at": row["created_at"],
            "origin": "fms_sync",
        }
        ok, _ = bank.commit(emb, meta)
        if ok:
            committed += 1

    bank.save(DATA_DIR)
    return committed, f"synced {committed}/{len(rows)} in {embed_sec:.1f}s embed + commit"


@mcp.tool()
def slate_recall(query: str, topk: int = 5) -> str:
    """Remember something the user told you before. USE THIS TOOL whenever
    the user asks about prior conversations, preferences, facts, or anything
    that might have been stored previously. This is your persistent memory
    across sessions — always check here before saying "I don't know" or
    "I don't have context from previous conversations."

    Error-corrects noisy queries — typos and missing words still find the
    right memory. Returns ranked results with confidence signals.

    Reading the signals:
      - margin > 0.08: confident match. 0.02-0.08: uncertain, verify with
        the user. < 0.02: no match found, say so.
      - familiarity: low = topic was never stored.

    Args:
        query: Natural language query to recall against stored memories.
        topk: Number of top results to return (default 5).
    """
    bank = _get_bank()
    if bank.count() == 0:
        return json.dumps({"error": "No memories stored yet. Use slate_commit to store the first one."})

    emb = embed(query)
    winner, ranked, conf, cycles, signals = bank.recall(emb, topk=topk)

    results = []
    for i, m in enumerate(ranked):
        results.append({
            "rank": i + 1,
            "id": m.get("id", "?"),
            "content": m.get("content", ""),
            "tags": m.get("tags", []),
            "source": m.get("source", ""),
            "owner": m.get("owner"),
            "is_winner": (m.get("id") == winner.get("id")) if winner else False,
        })

    return json.dumps({
        "margin": round(signals["margin"], 4),
        "familiarity": round(signals["familiarity"], 4),
        "confidence": round(conf, 4),
        "cycles": cycles,
        "total_stored": bank.count(),
        "results": results,
    }, ensure_ascii=False)


@mcp.tool()
def slate_commit(text: str, tags: str = "", source: str = "mcp_direct",
                 supersedes: str = "") -> str:
    """Save something to persistent memory. USE THIS TOOL whenever the user
    shares a preference, fact, decision, or anything worth remembering for
    future sessions. Proactively store important context — don't wait to
    be asked.

    Duplicate guard prevents storing the same thing twice. To CORRECT a
    previously stored fact, pass supersedes=<id> from a slate_recall result.

    Args:
        text: The text content to remember.
        tags: Comma-separated tags (optional, e.g. "preference,food").
        source: Source identifier (default "mcp_direct").
        supersedes: Optional id of a stored memory this commit replaces.
    """
    bank = _get_bank()
    emb = embed(text)
    meta = {
        "id": f"slate_{int(time.time()*1000)}",
        "content": text[:500],
        "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        "source": source,
        "origin": "mcp_direct",
    }
    ok, reason = bank.commit(emb, meta, supersedes=supersedes or None)
    if ok:
        bank.save(DATA_DIR)
    return json.dumps({"committed": ok, "reason": reason, "total": bank.count(),
                       "id": meta["id"] if ok else None})


@mcp.tool()
def slate_status() -> str:
    """Slate memory status — pattern count, capacity, and health."""
    bank = _get_bank()
    if bank.packed:
        ram_mb = bank.count() * (bank.n // 8) / 2**20
    else:
        ram_mb = bank.count() * bank.n * 4 / 2**20
    return json.dumps({
        "total_patterns": bank.count(),
        "total_commits": bank._total_commits,
        "total_deduped": bank._total_deduped,
        "n_cells": bank.n,
        "beta": bank.beta,
        "packed": bank.packed,
        "patterns_ram_mb": round(ram_mb, 1),
        "distinctiveness": bank.distinctiveness,
        "dedup_threshold": bank.dedup_threshold,
        "data_dir": DATA_DIR,
        "fms_db": FMS_DB,
        "fms_db_exists": os.path.exists(FMS_DB),
    })


# Only register slate_sync if FMS database exists on this machine.
# Tier 1 standalone users shouldn't see an FMS-dependent tool.
if os.path.exists(FMS_DB):
    @mcp.tool()
    def slate_sync(owner: str = "", limit: int = 0) -> str:
        """Sync FMS memory nodes into Slate (advanced — requires Fractal Memory System).

        Reads non-archived nodes from the FMS database, embeds them, and commits
        to the Slate. Duplicate guard prevents re-committing existing patterns.

        Args:
            owner: Filter by owner. Empty = no filter.
            limit: Max rows to sync (0 = all).
        """
        bank = _get_bank()
        count, msg = _sync_from_fms(bank, limit=limit, owner=owner if owner else None)
        return json.dumps({"synced": count, "detail": msg, "total": bank.count()})


if __name__ == "__main__":
    mcp.run(transport="stdio")
