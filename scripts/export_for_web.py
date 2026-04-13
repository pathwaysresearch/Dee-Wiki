"""
Export wiki pages + RAG chunks → webapp/data/ for Vercel deployment.

Reads wiki/*.md and data/chunks.json, generates embeddings for wiki pages,
writes:
  - webapp/data/wiki_pages.json  (wiki text + embeddings, ~700KB)
  - webapp/data/chunks.json      (RAG text only, NO embeddings, ~19MB)
  - webapp/data/chunks_embeddings.npy  (embeddings as numpy binary, ~70MB)

The split keeps the deployment under Vercel's 250MB function size limit.

APPEND BEHAVIOUR: If the output files already exist, only NEW records are
added. Existing records are identified by:
  - wiki_pages.json        → "path" field
  - chunks.json / .npy     → SHA-256 hash of "content" field

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
VAULT = PROJECT_ROOT / "prof-bhagwan-hybrid-demo"
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


# ---------------------------------------------------------------------------
# Existing-output loaders (what's already been exported)
# ---------------------------------------------------------------------------

def load_existing_wiki_pages(path: Path) -> tuple[list[dict], set[str]]:
    """
    Returns (existing_pages, existing_paths).
    existing_pages include their stored embeddings so we don't regenerate them.
    """
    if not path.exists():
        return [], set()
    pages = json.loads(path.read_text(encoding="utf-8"))
    existing_paths = {p["path"] for p in pages}
    return pages, existing_paths


def load_existing_chunks(
    chunks_path: Path, emb_path: Path
) -> tuple[list[dict], np.ndarray | None, set[str]]:
    """
    Returns (existing_text_chunks, existing_embeddings_array, existing_hashes).

    existing_text_chunks  — list of chunk dicts WITHOUT embeddings (as stored on disk)
    existing_embeddings   — float32 numpy array shape (N, D), or None if missing/empty
    existing_hashes       — set of content hashes already stored
    """
    if not chunks_path.exists():
        return [], None, set()

    text_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    for c in text_chunks:
        c.setdefault("type", "rag")

    existing_hashes = {content_hash(c["content"]) for c in text_chunks}

    embeddings = None
    if emb_path.exists() and len(text_chunks) > 0:
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != len(text_chunks):
            print(
                f"  WARN: chunks.json has {len(text_chunks)} records but "
                f"chunks_embeddings.npy has {embeddings.shape[0]} rows — "
                "treating embeddings as missing to avoid misalignment."
            )
            embeddings = None

    return text_chunks, embeddings, existing_hashes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    wiki_out = WEBAPP_DATA / "wiki_pages.json"
    chunks_out = WEBAPP_DATA / "chunks.json"
    emb_out = WEBAPP_DATA / "chunks_embeddings.npy"

    print("NOTE: This exports local wiki/*.md files only.")
    print("      Run 'python scripts/sync_wiki.py --pull' first to include Redis-only pages.\n")

    # -----------------------------------------------------------------------
    # WIKI PAGES
    # -----------------------------------------------------------------------
    print("Loading wiki pages…")
    source_wiki = load_wiki_pages()
    print(f"  {len(source_wiki)} wiki page(s) found in source")

    existing_wiki, existing_wiki_paths = load_existing_wiki_pages(wiki_out)
    if existing_wiki:
        print(f"  {len(existing_wiki)} wiki page(s) already in output — will skip duplicates")

    # Filter to genuinely new pages only
    new_wiki = [p for p in source_wiki if p["path"] not in existing_wiki_paths]
    print(f"  {len(new_wiki)} new wiki page(s) to add")

    if new_wiki and gemini_key:
        needs_embedding = [p for p in new_wiki if "embedding" not in p]
        if needs_embedding:
            print(f"  Generating embeddings for {len(needs_embedding)} new wiki page(s)…")
            texts = [p["content"] for p in needs_embedding]
            embs = get_embeddings_batch(texts, gemini_key, batch_pause=0.05)
            for page, emb in zip(needs_embedding, embs):
                page["embedding"] = emb
            print("  Wiki embeddings done.")
    elif new_wiki and not gemini_key:
        print("  WARN: No GEMINI_API_KEY — new wiki pages exported without embeddings")

    # Merge and write
    merged_wiki = existing_wiki + new_wiki
    wiki_out.write_text(json.dumps(merged_wiki, indent=2), encoding="utf-8")
    print(f"  Saved {len(merged_wiki)} total wiki page(s) → {wiki_out}\n")

    # -----------------------------------------------------------------------
    # RAG CHUNKS
    # -----------------------------------------------------------------------
    print("Loading RAG chunks…")
    source_chunks = load_chunks()
    print(f"  {len(source_chunks)} chunk(s) found in source")

    existing_text_chunks, existing_embeddings, existing_hashes = load_existing_chunks(
        chunks_out, emb_out
    )
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
    # Build merged embeddings numpy array
    # -----------------------------------------------------------------------
    # Collect embeddings for new chunks
    new_embeddings_list = [c.get("embedding") for c in new_chunks]
    has_all_new_embs = all(e is not None for e in new_embeddings_list)

    if has_all_new_embs and new_chunks:
        new_emb_array = np.array(new_embeddings_list, dtype=np.float32)

        if existing_embeddings is not None:
            merged_emb_array = np.concatenate([existing_embeddings, new_emb_array], axis=0)
        else:
            # Existing file missing/misaligned — rebuild from scratch using
            # whatever embeddings we have for old chunks too.
            old_embs = [c.get("embedding") for c in existing_text_chunks]
            if all(e is not None for e in old_embs) and old_embs:
                old_emb_array = np.array(old_embs, dtype=np.float32)
                merged_emb_array = np.concatenate([old_emb_array, new_emb_array], axis=0)
            else:
                # Old chunks have no in-memory embeddings — only save new ones
                # and warn that the file will be misaligned until a full rebuild.
                print(
                    "  WARN: Existing chunks have no embeddings in memory. "
                    "The .npy file will only contain embeddings for new chunks. "
                    "Run a full rebuild to fix alignment."
                )
                merged_emb_array = new_emb_array

        np.save(emb_out, merged_emb_array)
        print(
            f"  Embeddings → {emb_out} "
            f"({merged_emb_array.shape}, "
            f"{emb_out.stat().st_size / 1024 / 1024:.1f} MB)"
        )
    elif not new_chunks:
        print("  No new chunks — embeddings file unchanged.")
    else:
        print("  WARN: Not all new chunks have embeddings — skipping numpy export")

    # -----------------------------------------------------------------------
    # Save merged text-only chunks JSON (no embeddings)
    # -----------------------------------------------------------------------
    merged_chunks = existing_text_chunks + new_chunks
    chunks_text = [
        {k: v for k, v in c.items() if k != "embedding"} for c in merged_chunks
    ]
    chunks_out.write_text(json.dumps(chunks_text, indent=2), encoding="utf-8")
    print(
        f"  Chunks (text) → {chunks_out} "
        f"({chunks_out.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_docs = len(merged_wiki) + len(merged_chunks)
    has_embeddings = sum(1 for d in merged_wiki + merged_chunks if "embedding" in d)
    print(f"\n{'='*50}")
    print(f"  Export complete: {total_docs} total documents ({has_embeddings} with embeddings)")
    print(f"  Wiki pages : {len(merged_wiki)} total ({len(new_wiki)} new)")
    print(f"  RAG chunks : {len(merged_chunks)} total ({len(new_chunks)} new)")
    print(f"  Output     : {WEBAPP_DATA}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()