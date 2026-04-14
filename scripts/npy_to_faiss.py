"""
Convert existing chunks_embeddings.npy → chunks.faiss

Run once after upgrading to FAISS-based RAG search:

    python scripts/npy_to_faiss.py

Reads:   webapp/data/chunks_embeddings.npy
Writes:  webapp/data/chunks.faiss

The .npy file is NOT deleted — remove it manually once you've confirmed
the server loads the FAISS index correctly.
"""

from pathlib import Path
import sys

import numpy as np

try:
    import faiss
except ImportError:
    print("Error: faiss not installed.")
    print("Install with:  pip install faiss-cpu")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DATA  = PROJECT_ROOT / "webapp" / "data"
NPY_PATH     = WEBAPP_DATA / "chunks_embeddings.npy"
FAISS_PATH   = WEBAPP_DATA / "chunks.faiss"


def convert():
    if not NPY_PATH.exists():
        print(f"Not found: {NPY_PATH}")
        print("Run export_for_web.py first to generate the embedding file.")
        sys.exit(1)

    print(f"Loading {NPY_PATH} ...")
    emb = np.load(str(NPY_PATH)).astype(np.float32)
    print(f"  Shape: {emb.shape}  ({emb.nbytes / 1024 / 1024:.1f} MB)")

    # L2-normalise so inner-product == cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / (norms + 1e-8)

    dim = normed.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normed)
    print(f"  Built IndexFlatIP: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(FAISS_PATH))
    size_mb = FAISS_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved → {FAISS_PATH}  ({size_mb:.1f} MB)")
    print("\nDone. You can now delete chunks_embeddings.npy if no longer needed.")


if __name__ == "__main__":
    convert()
