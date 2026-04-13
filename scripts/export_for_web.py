"""
Export wiki pages + RAG chunks → webapp/data/ for Vercel deployment.

Reads wiki/*.md and data/chunks.json, generates embeddings for wiki pages,
writes:
  - webapp/data/wiki_pages.json  (wiki text + embeddings, ~700KB)
  - webapp/data/chunks.json      (RAG text only, NO embeddings, ~19MB)
  - webapp/data/chunks_embeddings.npy  (embeddings as numpy binary, ~70MB)

The split keeps the deployment under Vercel's 250MB function size limit.

IMPORTANT: This script only reads from local wiki/*.md files on disk.
If there are query-synthesized pages in Redis that haven't been pulled yet,
they will be MISSING from the export. Always run:

    python scripts/sync_wiki.py --pull

BEFORE running this script to ensure all pages are included.

Usage:
    python scripts/export_for_web.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chunker import get_embeddings_batch

PROJECT_ROOT = Path(__file__).parent.parent
VAULT = PROJECT_ROOT / "Vault"
WIKI_DIR = VAULT / "wiki"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.json"
WEBAPP_DATA = PROJECT_ROOT / "webapp" / "data"


def load_wiki_pages():
    """Read all .md files from wiki/ directory."""
    pages = []
    if not WIKI_DIR.exists():
        return pages

    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        rel_path = md_file.relative_to(VAULT)
        content = md_file.read_text(encoding="utf-8").strip()

        # Skip near-empty scaffold files
        if len(content) < 50:
            continue

        # Extract title from first heading or filename
        title = md_file.stem.replace("-", " ").replace("_", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        pages.append(
            {
                "title": title,
                "path": str(rel_path),
                "content": content,
                "type": "wiki",
            }
        )

    return pages


def load_chunks():
    """Load RAG chunks from data/chunks.json."""
    if not CHUNKS_FILE.exists():
        return []
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    # Ensure type field
    for c in chunks:
        c.setdefault("type", "rag")
    return chunks


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    print("NOTE: This exports local wiki/*.md files only.")
    print("      Run 'python scripts/sync_wiki.py --pull' first to include Redis-only pages.\n")

    # --- Wiki pages ---
    print("Loading wiki pages…")
    wiki_pages = load_wiki_pages()
    print(f"  {len(wiki_pages)} wiki page(s)")

    if wiki_pages and gemini_key:
        # Check which pages already have embeddings
        needs_embedding = [p for p in wiki_pages if "embedding" not in p]
        if needs_embedding:
            print(f"  Generating embeddings for {len(needs_embedding)} wiki page(s)…")
            texts = [p["content"] for p in needs_embedding]
            embs = get_embeddings_batch(texts, gemini_key, batch_pause=0.05)
            for page, emb in zip(needs_embedding, embs):
                page["embedding"] = emb
            print("  Wiki embeddings done.")
    elif not gemini_key:
        print("  WARN: No GEMINI_API_KEY — wiki pages exported without embeddings")

    wiki_out = WEBAPP_DATA / "wiki_pages.json"
    wiki_out.write_text(json.dumps(wiki_pages, indent=2), encoding="utf-8")
    print(f"  Saved → {wiki_out}")

    # --- RAG chunks ---
    print("Loading RAG chunks…")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunk(s)")

    # Chunks should already have embeddings from ingest.
    # Generate for any that are missing.
    missing_emb = [c for c in chunks if "embedding" not in c]
    if missing_emb and gemini_key:
        print(f"  Generating embeddings for {len(missing_emb)} chunk(s) missing embeddings…")
        texts = [c["content"] for c in missing_emb]
        embs = get_embeddings_batch(texts, gemini_key, batch_pause=0.05)
        for chunk, emb in zip(missing_emb, embs):
            chunk["embedding"] = emb
        print("  Chunk embeddings done.")

    # --- Split chunks: text JSON + embeddings numpy binary ---
    # Extract embeddings into a numpy array (preserving chunk order)
    emb_dim = None
    all_embeddings = []
    for c in chunks:
        emb = c.get("embedding")
        if emb is not None:
            all_embeddings.append(emb)
            if emb_dim is None:
                emb_dim = len(emb)

    has_all_chunk_embeddings = len(all_embeddings) == len(chunks)

    if has_all_chunk_embeddings and emb_dim:
        emb_array = np.array(all_embeddings, dtype=np.float32)
        emb_out = WEBAPP_DATA / "chunks_embeddings.npy"
        np.save(emb_out, emb_array)
        print(f"  Embeddings → {emb_out} ({emb_array.shape}, {emb_out.stat().st_size / 1024 / 1024:.1f}MB)")
    else:
        print("  WARN: Not all chunks have embeddings — skipping numpy export")

    # Save text-only chunks JSON (no embeddings — saves ~330MB)
    chunks_text = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
    chunks_out = WEBAPP_DATA / "chunks.json"
    chunks_out.write_text(json.dumps(chunks_text, indent=2), encoding="utf-8")
    print(f"  Chunks (text) → {chunks_out} ({chunks_out.stat().st_size / 1024 / 1024:.1f}MB)")

    # --- Summary ---
    total_docs = len(wiki_pages) + len(chunks)
    has_embeddings = sum(1 for d in wiki_pages + chunks if "embedding" in d)
    print(f"\n{'='*50}")
    print(f"  Export complete: {total_docs} documents ({has_embeddings} with embeddings)")
    print(f"  Wiki pages: {len(wiki_pages)}")
    print(f"  RAG chunks: {len(chunks)}")
    print(f"  Output: {WEBAPP_DATA}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
