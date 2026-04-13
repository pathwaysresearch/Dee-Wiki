"""
Ingest orchestrator for the Two-Tier LLM Wiki.

Usage (from project root):
    python scripts/ingest.py --scan              # list new files + routing plan
    python scripts/ingest.py --process <path>    # chunk + embed a single RAG file
    python scripts/ingest.py --process-all       # chunk + embed ALL RAG-routed files
    python scripts/ingest.py --search "query"    # BM25 search over RAG chunks (quick lookup)

After running --process or --process-all, use Claude Code to create wiki pages / stubs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import date, datetime

# Add scripts dir to path so chunker is importable
sys.path.insert(0, str(Path(__file__).parent))
from chunker import (
    chunk_pdf,
    chunk_markdown,
    count_words,
    get_page_count,
    get_embeddings_batch,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
VAULT = PROJECT_ROOT / "Vault"
RAW_DIR = VAULT / "raw"
WIKI_DIR = VAULT / "wiki"
LOG_FILE = WIKI_DIR / "log.md"
INDEX_FILE = WIKI_DIR / "index.md"
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
INGESTED_FILE = DATA_DIR / "ingested.json"

WORD_THRESHOLD = 5000
PAGE_THRESHOLD = 20

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_already_ingested():
    """Load the structured ingested.json sidecar tracking file.
    Returns a dict of relative_path → {chunks, words, timestamp}."""
    if INGESTED_FILE.exists():
        try:
            return json.loads(INGESTED_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return {}
    return {}


def save_ingested(ingested):
    """Write the ingested tracking file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INGESTED_FILE.write_text(
        json.dumps(ingested, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def scan_raw_files():
    """Return list of all supported files under raw/."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(RAW_DIR.rglob(f"*{ext}"))
    # Sort by path for deterministic ordering
    return sorted(files)


def route_file(file_path):
    """Determine routing: 'wiki' or 'rag'."""
    words = count_words(str(file_path))
    pages = get_page_count(str(file_path))

    if pages > PAGE_THRESHOLD or words > WORD_THRESHOLD:
        return "rag", words, pages
    return "wiki", words, pages


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan():
    """List all files and their routing."""
    files = scan_raw_files()
    if not files:
        print("No files found in raw/. Upload source documents first.")
        return

    ingested = get_already_ingested()

    wiki_files = []
    rag_files = []

    print(f"\n{'='*70}")
    print(f"  INGEST SCAN — {len(files)} file(s) found in raw/")
    print(f"{'='*70}\n")

    for f in files:
        rel = str(f.relative_to(PROJECT_ROOT))
        route, words, pages = route_file(f)

        # Check if already ingested via structured tracking
        already = rel in ingested
        status = " [DONE]" if already else " [NEW]"

        page_info = f", {pages} pages" if pages else ""
        print(f"  {'WIKI' if route == 'wiki' else 'RAG ':>4}  {rel}  ({words} words{page_info}){status}")

        if not already:
            if route == "wiki":
                wiki_files.append((f, words, pages))
            else:
                rag_files.append((f, words, pages))

    print(f"\n{'─'*70}")
    print(f"  New files: {len(wiki_files)} wiki + {len(rag_files)} RAG")

    if rag_files:
        print(f"\n  Run this to chunk & embed RAG files:")
        print(f"    python scripts/ingest.py --process-all")

    if wiki_files:
        print(f"\n  After processing, tell Claude Code:")
        print(f"    'Ingest these wiki files: {', '.join(f.name for f, _, _ in wiki_files)}'")

    print()


def cmd_process(file_path, gemini_key):
    """Chunk and embed a single file."""
    fp = Path(file_path).resolve()
    if not fp.exists():
        print(f"File not found: {fp}")
        sys.exit(1)

    ext = fp.suffix.lower()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {fp.name}")

    if ext == ".pdf":
        # Save converted markdown to research_papers_md/ folder
        md_output = VAULT / "raw" / "research_papers_md"
        chunks, text, page_count = chunk_pdf(str(fp), gemini_key, md_output_dir=str(md_output))
        print(f"  Extracted {page_count} pages → {len(chunks)} chunks")
    elif ext in (".md", ".txt"):
        chunks, text = chunk_markdown(str(fp))
        print(f"  {len(chunks)} chunks")
    else:
        print(f"  Unsupported: {ext}")
        return

    # Generate embeddings
    if gemini_key:
        print(f"  Generating embeddings for {len(chunks)} chunks…")
        texts = [c["content"] for c in chunks]
        embs = get_embeddings_batch(texts, gemini_key, batch_pause=0.05)
        for chunk, emb in zip(chunks, embs):
            chunk["embedding"] = emb
        print(f"  Embeddings done.")
    else:
        print("  WARN: No GEMINI_API_KEY — skipping embeddings")

    # Merge into chunks file
    existing = []
    if CHUNKS_FILE.exists():
        existing = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    existing = [c for c in existing if c.get("source") != str(fp)]
    existing.extend(chunks)

    CHUNKS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"  Saved → {CHUNKS_FILE} ({len(existing)} total chunks)")

    # Print summary for Claude Code to create stub
    try:
        rel_source = fp.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        rel_source = fp.name
    print(f"\n  ┌─ STUB INFO (for Claude Code) ────────────────")
    print(f"  │ Title: {fp.stem.replace('-', ' ').replace('_', ' ').title()}")
    print(f"  │ Source: {rel_source}")
    print(f"  │ Words: ~{len(text.split())}")
    if ext == ".pdf":
        print(f"  │ Pages: {page_count}")
    print(f"  │ Chunks: {len(chunks)}")
    abstract_hint = text[:500].replace("\n", " ").strip()
    print(f"  │ Abstract hint: {abstract_hint}…")
    print(f"  └────────────────────────────────────────────────")

    # BUG 7 fix: Track in ingested.json
    ingested = get_already_ingested()
    rel_key = str(rel_source)
    ingested[rel_key] = {
        "chunks": len(chunks),
        "words": len(text.split()),
        "timestamp": datetime.now().isoformat(),
    }
    save_ingested(ingested)
    print(f"  Tracked in ingested.json: {rel_key}")

    # BUG 5 fix: Log to wiki log
    from wiki_logger import log_to_wiki_log
    log_to_wiki_log(
        "ingest",
        fp.stem.replace("-", " ").replace("_", " ").title(),
        {
            "source": str(rel_source),
            "chunks": len(chunks),
            "words": len(text.split()),
        }
    )


def cmd_process_all(gemini_key):
    """Process all RAG-routed files."""
    files = scan_raw_files()
    ingested = get_already_ingested()

    rag_files = []
    for f in files:
        rel = str(f.relative_to(PROJECT_ROOT))
        route, _, _ = route_file(f)
        if route == "rag" and rel not in ingested:
            rag_files.append(f)

    if not rag_files:
        print("No new RAG files to process.")
        return

    print(f"\nProcessing {len(rag_files)} RAG file(s)…\n")
    for f in rag_files:
        cmd_process(str(f), gemini_key)
        print()


def cmd_search(query):
    """Quick BM25 search over chunks."""
    if not CHUNKS_FILE.exists():
        print("No chunks yet. Run --process-all first.")
        return

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    if not chunks:
        print("Chunk index is empty.")
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Install rank-bm25: pip install rank-bm25")
        sys.exit(1)

    tokenized = [c["content"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    # Top 5
    import numpy as np
    top_idx = np.argsort(scores)[::-1][:5]

    print(f"\nTop 5 results for: '{query}'\n")
    for rank, idx in enumerate(top_idx, 1):
        c = chunks[idx]
        snippet = c["content"][:200].replace("\n", " ")
        print(f"  {rank}. [{scores[idx]:.2f}] {c['source']}")
        print(f"     {snippet}…\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    parser = argparse.ArgumentParser(description="Ingest orchestrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="List files + routing plan")
    group.add_argument("--process", type=str, metavar="PATH", help="Chunk + embed one file")
    group.add_argument("--process-all", action="store_true", help="Chunk + embed all RAG files")
    group.add_argument("--search", type=str, metavar="QUERY", help="BM25 search over chunks")
    args = parser.parse_args()

    if args.scan:
        cmd_scan()
    elif args.process:
        cmd_process(args.process, gemini_key)
    elif args.process_all:
        cmd_process_all(gemini_key)
    elif args.search:
        cmd_search(args.search)
