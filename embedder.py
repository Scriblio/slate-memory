"""Embedder for the Slate MCP server — uses FMS's ONNX MiniLM (384-dim).

Falls back to sentence-transformers if FMS repo isn't available.
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)

_embedder = None


def _init():
    global _embedder

    fms_repo = os.environ.get(
        "FMS_REPO",
        r"C:\Users\matth\OneDrive - Always Black Friday (3)\fractal-memory"
    )
    if os.path.isdir(fms_repo):
        try:
            # Strip Aurelia's env vars that make FMSConfig's pydantic choke
            clean_keys = [k for k in os.environ if k.startswith("AURELIA_")]
            stash = {k: os.environ.pop(k) for k in clean_keys}
            os.environ.pop("ANTHROPIC_API_KEY", None)
            if fms_repo not in sys.path:
                sys.path.insert(0, fms_repo)
            from src.config import FMSConfig
            from src.engine.embedder import Embedder
            _embedder = Embedder(FMSConfig())
            # Restore stashed vars
            os.environ.update(stash)
            logger.info("Using FMS ONNX embedder (384-dim)")
            return
        except Exception as e:
            logger.warning("FMS embedder failed: %s — falling back", e)

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(
            "Downloading embedding model (~90MB, first time only)... "
            "This may take a few minutes."
        )
        print(
            "[Slate] Downloading embedding model (~90MB, first time only)... "
            "this may take a few minutes.",
            file=sys.stderr, flush=True,
        )
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _embedder = model
        logger.info("Using sentence-transformers MiniLM (384-dim)")
    except ImportError:
        raise RuntimeError(
            "No embedder available. Install sentence-transformers: "
            "pip install sentence-transformers"
        )


def embed(text):
    if _embedder is None:
        _init()
    if hasattr(_embedder, "embed"):
        return _embedder.embed(text)
    return _embedder.encode(text, convert_to_numpy=True)


def embed_batch(texts, batch=512):
    if _embedder is None:
        _init()
    if hasattr(_embedder, "embed_batch"):
        out = []
        for i in range(0, len(texts), batch):
            out.extend(_embedder.embed_batch(texts[i:i + batch]))
        return out
    if hasattr(_embedder, "encode"):
        return list(_embedder.encode(texts, batch_size=batch, convert_to_numpy=True))
    return [embed(t) for t in texts]
