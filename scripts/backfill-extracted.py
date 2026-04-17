"""
Backfill data/extracted.json from existing wiki pages.

For every wiki page that already exists, reads its `sourced_from` frontmatter
field, finds matching chunks in chunks.json, and marks their hashes as
extracted — exactly as if extract_entities.py had processed them.

Run this ONCE after adding chunk tracking to catch up with pre-tracking work.
Safe to re-run: it only adds hashes, never removes them.

Usage:
    python scripts/backfill-extracted.py [--dry-run]
"""

import os
import json
import re
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
VAULT        = PROJECT_ROOT / os.environ.get("WIKI_VAULT_NAME", "webapp/Vault")
WIKI_DIR     = VAULT / "wiki"
DATA_DIR     = PROJECT_ROOT / "data"
CHUNKS_FILE  = DATA_DIR / "chunks.json"
EXTRACTED_FILE = DATA_DIR / "extracted.json"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers (duplicated from extract_entities.py to keep this script standalone)
# ---------------------------------------------------------------------------

def chunk_hash(chunk: dict) -> str:
    import hashlib
    return hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()


def load_extracted_hashes() -> set:
    if not EXTRACTED_FILE.exists():
        return set()
    try:
        return set(json.loads(EXTRACTED_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def save_extracted_hashes(hashes: set):
    EXTRACTED_FILE.write_text(
        json.dumps(sorted(hashes), indent=2), encoding="utf-8"
    )


def parse_sourced_from(md_content: str) -> str | None:
    """Extract the sourced_from value from YAML frontmatter."""
    m = FRONTMATTER_RE.match(md_content)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if line.startswith("sourced_from:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            return value if value else None
    return None


def source_matches(chunk_source: str, sourced_from: str) -> bool:
    """
    Check whether a chunk's source path corresponds to a wiki page's sourced_from.

    sourced_from is typically a human-readable title like:
        "Driving Digital Strategy   A Guide To Reimagining Your (1)"
    chunk source is a file path like:
        "raw/books/driving_digital_strategy.md"

    We normalise both to lowercase words and check for significant overlap.
    """
    def tokenise(s: str) -> set:
        # lowercase, split on non-alpha, drop short tokens and stopwords
        stopwords = {"a", "an", "the", "to", "of", "and", "in", "for",
                     "is", "its", "by", "on", "at", "or", "be", "1"}
        return {
            t for t in re.split(r"[^a-z0-9]+", s.lower())
            if len(t) > 2 and t not in stopwords
        }

    chunk_tokens    = tokenise(Path(chunk_source).stem)
    sourced_tokens  = tokenise(sourced_from)

    if not chunk_tokens or not sourced_tokens:
        return False

    overlap = chunk_tokens & sourced_tokens
    # Require at least half the shorter set to overlap
    threshold = min(len(chunk_tokens), len(sourced_tokens)) / 2
    return len(overlap) >= max(1, threshold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backfill(dry_run: bool = False):
    # 1. Load chunks
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found.")
        sys.exit(1)
    all_chunks = [
        c for c in json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        if isinstance(c, dict) and "content" in c and "source" in c
    ]
    print(f"Loaded {len(all_chunks)} chunks from chunks.json")

    # 2. Collect sourced_from values from all existing wiki pages
    sourced_from_values: set = set()
    page_count = 0
    skip_stems = {"index", "log"}

    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        if md_file.stem in skip_stems or md_file.stem.startswith("_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError:
            continue
        sf = parse_sourced_from(content)
        if sf:
            sourced_from_values.add(sf)
            page_count += 1

    print(f"Found {page_count} wiki pages with sourced_from")
    print(f"Unique sources referenced: {len(sourced_from_values)}")

    # 3. Match chunks to sourced_from values
    already_extracted = load_extracted_hashes()
    new_hashes: set = set()
    unmatched_sources: set = set()

    # Group chunks by source for reporting
    matched_by_source: dict = {}

    for sf in sorted(sourced_from_values):
        matching_chunks = [
            c for c in all_chunks
            if source_matches(c["source"], sf)
        ]
        if not matching_chunks:
            unmatched_sources.add(sf)
            continue

        hashes = {chunk_hash(c) for c in matching_chunks}
        actually_new = hashes - already_extracted
        new_hashes |= actually_new

        # Use stem of first match as display name
        display = Path(matching_chunks[0]["source"]).stem
        matched_by_source[sf] = {
            "display": display,
            "total_chunks": len(matching_chunks),
            "new_hashes": len(actually_new),
        }

    # 4. Report
    print(f"\n{'Source (sourced_from)':<55} {'Chunks':>7} {'New hashes':>11}")
    print("-" * 76)
    for sf, info in sorted(matched_by_source.items()):
        print(f"{sf[:55]:<55} {info['total_chunks']:>7} {info['new_hashes']:>11}")

    if unmatched_sources:
        print(f"\nCould not match {len(unmatched_sources)} sourced_from value(s) to any chunk:")
        for sf in sorted(unmatched_sources):
            print(f"  ✗ {sf}")
        print("  → These pages may be synthesized, persona, or stub pages with no raw chunks.")

    print(f"\nAlready in extracted.json : {len(already_extracted)}")
    print(f"New hashes to add         : {len(new_hashes)}")
    print(f"Total after backfill      : {len(already_extracted | new_hashes)}")

    # 5. Write
    if dry_run:
        print("\n[dry-run] No changes written.")
        return

    if not new_hashes:
        print("\nNothing new to add — extracted.json is already up to date.")
        return

    save_extracted_hashes(already_extracted | new_hashes)
    print(f"\n✓ Saved {len(already_extracted | new_hashes)} hashes → {EXTRACTED_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill extracted.json from existing wiki pages."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be added without writing anything."
    )
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)