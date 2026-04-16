"""
Remove RAG chunks by source path prefix from chunks.json (and chunks.faiss if aligned).

TWO MODES depending on which file you target:

  --offline (default)
      Cleans data/chunks.json — the offline pipeline file that stores embeddings.
      After running, execute:  python scripts/export_for_web.py
      That rebuilds webapp/data/chunks.json + chunks.faiss from the stored embeddings.
      No re-embedding. No Gemini API calls.

  --deployed
      Cleans webapp/data/chunks.json + webapp/data/chunks.faiss directly.
      Only works when the two files are in sync (same entry count).
      Use this when you just need to patch a deployed instance without touching
      the offline pipeline files.

Usage:
    python scripts/remove_chunks.py <source-path-prefix>              # offline mode
    python scripts/remove_chunks.py <source-path-prefix> --deployed   # deployed mode

Examples:
    python scripts/remove_chunks.py "Vault/raw/books/some-book.md"
    python scripts/remove_chunks.py "E:/OtherProject/raw/" --deployed

Path matching is case-insensitive and normalised to forward slashes.
"""

import sys
import json
import shutil
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    print("Error: faiss not installed.  Run: pip install faiss-cpu")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DATA  = PROJECT_ROOT / "webapp" / "data"
OFFLINE_DATA = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def _classify(chunks, prefix):
    keep_idx, remove_idx = [], []
    for i, chunk in enumerate(chunks):
        src = _norm(chunk.get("source", ""))
        if src.startswith(prefix):
            remove_idx.append(i)
        else:
            keep_idx.append(i)
    return keep_idx, remove_idx


def _print_summary(chunks, remove_idx, prefix):
    keep_idx = [i for i in range(len(chunks)) if i not in set(remove_idx)]
    affected = sorted({_norm(chunks[i].get("source", "")) for i in remove_idx})
    print(f"\nChunks to remove : {len(remove_idx)}")
    print(f"Chunks to keep   : {len(keep_idx)}")
    print(f"Affected sources ({len(affected)}):")
    for s in affected:
        count = sum(1 for i in remove_idx if _norm(chunks[i].get("source", "")) == s)
        print(f"  [{count:>4} chunks]  {s}")


# ---------------------------------------------------------------------------
# Mode: offline  (data/chunks.json — has embeddings stored)
# ---------------------------------------------------------------------------

def run_offline(prefix):
    chunks_path = OFFLINE_DATA / "chunks.json"

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found.")
        sys.exit(1)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from data/chunks.json")

    keep_idx, remove_idx = _classify(chunks, prefix)

    if not remove_idx:
        print(f"No chunks match prefix {prefix!r} — nothing to do.")
        sys.exit(0)

    _print_summary(chunks, remove_idx, prefix)

    print()
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted — no files changed.")
        sys.exit(0)

    # Backup
    bak = chunks_path.with_suffix(".json.bak")
    shutil.copy2(chunks_path, bak)
    print(f"\nBacked up → {bak.name}")

    # Write cleaned file (embeddings preserved for kept chunks)
    kept = [chunks[i] for i in keep_idx]
    chunks_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    size_mb = chunks_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(kept)} chunks → data/chunks.json  ({size_mb:.1f} MB)")

    print(f"\nDone. Removed {len(remove_idx)} chunk(s).")
    print("\nNext: rebuild webapp/data/ from the cleaned offline file (no re-embedding):")
    print("    python scripts/export_for_web.py")
    print("This reuses the stored embeddings — Gemini API is NOT called for existing chunks.")
    print(f"\nTo restore backup:  copy data/{bak.name} data/chunks.json")


# ---------------------------------------------------------------------------
# Mode: deployed  (webapp/data/chunks.json + chunks.faiss — no embeddings in JSON)
# ---------------------------------------------------------------------------

def run_deployed(prefix):
    chunks_path = WEBAPP_DATA / "chunks.json"
    faiss_path  = WEBAPP_DATA / "chunks.faiss"

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found.")
        sys.exit(1)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from webapp/data/chunks.json")

    keep_idx, remove_idx = _classify(chunks, prefix)

    if not remove_idx:
        print(f"No chunks match prefix {prefix!r} — nothing to do.")
        sys.exit(0)

    _print_summary(chunks, remove_idx, prefix)

    # Check FAISS alignment
    faiss_index = None
    all_vecs    = None

    if not faiss_path.exists():
        print(f"\nWarning: {faiss_path} not found — only chunks.json will be updated.")
    else:
        faiss_index = faiss.read_index(str(faiss_path))
        n_vecs   = faiss_index.ntotal
        n_chunks = len(chunks)

        if n_vecs != n_chunks:
            print(
                f"\nERROR: FAISS has {n_vecs} vectors but chunks.json has {n_chunks} entries — out of sync."
                f"\nThe webapp/data/ files are already corrupted and can't be fixed by position."
                f"\n\nFix via the offline file instead:"
                f"\n  1. python scripts/remove_chunks.py \"{sys.argv[1]}\"   (no --deployed flag)"
                f"\n  2. python scripts/export_for_web.py"
                f"\nThis does NOT re-embed — it reuses embeddings stored in data/chunks.json."
            )
            sys.exit(1)

        print(f"\nFAISS index      : {n_vecs} vectors  dim={faiss_index.d}  (aligned ✓)")
        all_vecs = faiss_index.reconstruct_n(0, n_vecs)

    print()
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted — no files changed.")
        sys.exit(0)

    # Backup
    bak_json = chunks_path.with_suffix(".json.bak")
    shutil.copy2(chunks_path, bak_json)
    print(f"\nBacked up → {bak_json.name}")
    if faiss_index is not None:
        bak_faiss = faiss_path.with_suffix(".faiss.bak")
        shutil.copy2(faiss_path, bak_faiss)
        print(f"Backed up → {bak_faiss.name}")

    # Write cleaned chunks.json
    kept = [chunks[i] for i in keep_idx]
    chunks_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    size_kb = chunks_path.stat().st_size / 1024
    print(f"\nSaved {len(kept)} chunks → chunks.json  ({size_kb:.0f} KB)")

    # Rebuild FAISS from kept vectors
    if faiss_index is not None:
        kept_vecs = all_vecs[np.array(keep_idx)].astype(np.float32)
        new_index = faiss.IndexFlatIP(kept_vecs.shape[1])
        new_index.add(kept_vecs)
        faiss.write_index(new_index, str(faiss_path))
        size_mb = faiss_path.stat().st_size / 1024 / 1024
        print(f"Rebuilt FAISS    → chunks.faiss  ({new_index.ntotal} vectors, {size_mb:.1f} MB)")

    print(f"\nDone. Removed {len(remove_idx)} chunk(s).")
    print("Restart the server (or re-deploy) to pick up the changes.")
    print(f"\nTo restore backups:")
    print(f"  copy webapp/data/{bak_json.name} webapp/data/chunks.json")
    if faiss_index is not None:
        print(f"  copy webapp/data/{bak_faiss.name} webapp/data/chunks.faiss")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if not args:
        print(__doc__)
        sys.exit(1)

    raw_prefix = args[0]
    prefix = _norm(raw_prefix)
    # Auto-append slash for directory-style prefixes (no extension in final segment)
    if not prefix.endswith("/") and "." not in Path(prefix).name:
        prefix += "/"

    deployed = "--deployed" in flags

    print(f"Mode           : {'--deployed (webapp/data/)' if deployed else '--offline (data/)'}")
    print(f"Source prefix  : {prefix!r}")

    if deployed:
        run_deployed(prefix)
    else:
        run_offline(prefix)


if __name__ == "__main__":
    main()
