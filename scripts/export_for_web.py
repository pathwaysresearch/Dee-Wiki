"""
Export wiki pages + RAG chunks → webapp/data/ for Vercel deployment.

Reads wiki/*.md and data/chunks.json, generates embeddings for wiki pages,
writes:
  - webapp/data/chunks.json      (RAG text only, NO embeddings, ~19MB)
  - webapp/data/chunks.faiss     (FAISS inner-product index of chunk embeddings)

The split keeps the deployment under Vercel's 250MB function size limit.

APPEND BEHAVIOUR: If the output files already exist, only NEW records are
added. Existing records are identified by:
  - chunks.json      → SHA-256 hash of "content" field

If output files do not exist they are created from scratch.

IMPORTANT: This script only reads from local wiki/*.md files on disk.
If there are query-synthesized pages in Redis that haven't been pulled yet,
they will be MISSING from the export. Always run:

    python scripts/sync_wiki.py --pull

BEFORE running this script to ensure all pages are included.

Usage:
    python scripts/export_for_web.py
"""

import hashlib
import os
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chunker import get_embeddings_batch

PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
_vault_name = os.environ.get("WIKI_VAULT_NAME", "webapp/Vault")
VAULT = PROJECT_ROOT / _vault_name
WIKI_DIR = VAULT / "wiki"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.json"
WEBAPP_DATA = PROJECT_ROOT / "webapp" / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """Stable SHA-256 fingerprint for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Loaders (source data)
# ---------------------------------------------------------------------------


def load_chunks():
    """Load RAG chunks from data/chunks.json."""
    if not CHUNKS_FILE.exists():
        return []
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    # Ensure type field
    for c in chunks:
        c.setdefault("type", "rag")
    return chunks


# ---------------------------------------------------------------------------
# Existing-output loaders (what's already been exported)
# ---------------------------------------------------------------------------

def load_existing_chunks(
    chunks_path: Path,
) -> tuple[list[dict], set[str]]:
    """
    Returns (existing_text_chunks, existing_hashes).

    existing_text_chunks — list of chunk dicts WITHOUT embeddings (as stored on disk)
    existing_hashes      — set of content hashes already stored
    """
    if not chunks_path.exists():
        return [], set()

    text_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    for c in text_chunks:
        c.setdefault("type", "rag")

    existing_hashes = {content_hash(c["content"]) for c in text_chunks}
    return text_chunks, existing_hashes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    chunks_out = WEBAPP_DATA / "chunks.json"
    faiss_out = WEBAPP_DATA / "chunks.faiss"

    print("NOTE: This exports local wiki/*.md files only.")

    # -----------------------------------------------------------------------
    # RAG CHUNKS
    # -----------------------------------------------------------------------
    print("Loading RAG chunks…")
    source_chunks = load_chunks()
    print(f"  {len(source_chunks)} chunk(s) found in source")

    existing_text_chunks, existing_hashes = load_existing_chunks(chunks_out)
    if existing_text_chunks:
        print(
            f"  {len(existing_text_chunks)} chunk(s) already in output — will skip duplicates"
        )

    # Filter to genuinely new chunks only (by content hash)
    new_chunks = [
        c for c in source_chunks if content_hash(c["content"]) not in existing_hashes
    ]
    print(f"  {len(new_chunks)} new chunk(s) to add")

    # Generate embeddings for new chunks that are missing them
    missing_emb = [c for c in new_chunks if "embedding" not in c]
    if missing_emb and gemini_key:
        print(
            f"  Generating embeddings for {len(missing_emb)} chunk(s) missing embeddings…"
        )
        texts = [c["content"] for c in missing_emb]
        embs = get_embeddings_batch(texts, gemini_key, batch_pause=0.05)
        for chunk, emb in zip(missing_emb, embs):
            chunk["embedding"] = emb
        print("  Chunk embeddings done.")

    # -----------------------------------------------------------------------
    # Build FAISS index from source_chunks (offline data/chunks.json).
    # source_chunks has embeddings for ALL chunks — old ones stored from ingest,
    # new ones just generated above.
    # -----------------------------------------------------------------------
    all_embs = [c.get("embedding") for c in source_chunks]
    has_all_embs = bool(all_embs) and all(e is not None for e in all_embs)
    missing_count = sum(1 for e in all_embs if e is None)

    if has_all_embs:
        try:
            import faiss  # noqa: PLC0415
            merged_emb_array = np.array(all_embs, dtype=np.float32)
            norms = np.linalg.norm(merged_emb_array, axis=1, keepdims=True)
            normed = (merged_emb_array / (norms + 1e-8)).astype(np.float32)
            index = faiss.IndexFlatIP(normed.shape[1])
            index.add(normed)
            faiss.write_index(index, str(faiss_out))
            print(
                f"  FAISS index → {faiss_out} "
                f"({index.ntotal} vectors, "
                f"{faiss_out.stat().st_size / 1024 / 1024:.1f} MB)"
            )
        except ImportError:
            print(" faiss not installed — run: pip install faiss-cpu")
    elif not source_chunks:
        print("  No chunks — FAISS index unchanged.")
    else:
        print(f"  WARN: {missing_count}/{len(source_chunks)} chunks missing embeddings — skipping FAISS export")

    # -----------------------------------------------------------------------
    # Save text-only chunks JSON (no embeddings) — order matches source_chunks
    # so chunk[i] always aligns with FAISS vector[i].
    # -----------------------------------------------------------------------
    chunks_text = [
        {k: v for k, v in c.items() if k != "embedding"} for c in source_chunks
    ]
    chunks_out.write_text(json.dumps(chunks_text, indent=2), encoding="utf-8")
    print(
        f"  Chunks (text) → {chunks_out} "
        f"({chunks_out.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_docs =len(source_chunks)
    has_embeddings = sum(1 for d in source_chunks if "embedding" in d)
    print(f"\n{'='*50}")
    print(f"  Export complete: {total_docs} total documents ({has_embeddings} with embeddings)")
    print(f"  RAG chunks : {len(source_chunks)} total ({len(new_chunks)} new)")
    print(f"  Output     : {WEBAPP_DATA}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()