"""
Document chunker for the Two-Tier LLM Wiki RAG pipeline.

Handles .md, .txt, and .pdf files.
PDF extraction: PyMuPDF → Gemini 2.5 Flash page-by-page → Gemini direct PDF upload.
"""

import os
import json
import sys
import re
import base64
import time
import requests
from pathlib import Path

# Try PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# PDF text extraction — three-tier fallback
# ---------------------------------------------------------------------------

def extract_text_pymupdf(pdf_path):
    """Tier 1: Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def extract_text_gemini_pages(pdf_path, gemini_api_key):
    """Tier 2: Send each page as an image to Gemini 2.5 Flash."""
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError(
            "PyMuPDF required for page-image extraction. "
            "Install with: pip install PyMuPDF"
        )

    doc = fitz.open(pdf_path)
    pages = []
    total = len(doc)

    for page_num in range(total):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Extract ALL text from this page. "
                                "Convert to clean markdown. "
                                "Preserve headings, lists, tables, formatting. "
                                "Output ONLY the extracted text."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 2:
                    print(f"    WARN: page {page_num+1} failed after 3 attempts: {exc}")
                    result = {}
                time.sleep(2 ** attempt)

        text = ""
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        pages.append({"page": page_num + 1, "text": text.strip()})
        print(f"    Gemini page-image: {page_num+1}/{total}")

    doc.close()
    return pages


def extract_text_gemini_direct(pdf_path, gemini_api_key):
    """Tier 3: Send entire PDF to Gemini (works without PyMuPDF, <20 MB)."""
    pdf_bytes = Path(pdf_path).read_bytes()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > 20:
        raise ValueError(
            f"PDF is {size_mb:.1f} MB (>20 MB limit for direct upload). "
            "Install PyMuPDF for page-by-page extraction: pip install PyMuPDF"
        )

    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Extract ALL text from this PDF document. "
                            "Convert to clean markdown. Preserve all headings, "
                            "paragraphs, lists, tables, and formatting. "
                            "Output ONLY the extracted text."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 65536},
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    text = ""
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)

    # Split into pseudo-pages (~3000 chars each) for consistency
    chars_per_page = 3000
    pages = []
    for i in range(0, len(text), chars_per_page):
        pages.append({"page": i // chars_per_page + 1, "text": text[i : i + chars_per_page].strip()})

    return pages


def extract_text_from_pdf(pdf_path, gemini_api_key):
    """
    Extract text from PDF with three-tier fallback:
    1. PyMuPDF (fast, free)
    2. Gemini page-by-page via images (needs PyMuPDF for rendering)
    3. Gemini direct PDF upload (no PyMuPDF needed, <20 MB)
    """
    # --- Tier 1: PyMuPDF ---
    if PYMUPDF_AVAILABLE:
        try:
            pages = extract_text_pymupdf(pdf_path)
            avg_chars = sum(len(p["text"]) for p in pages) / max(len(pages), 1)
            if avg_chars >= 50:
                print(f"  PyMuPDF OK — {len(pages)} pages, {avg_chars:.0f} avg chars/page")
                return pages
            print(f"  PyMuPDF quality low ({avg_chars:.0f} avg chars) → Gemini fallback")
        except Exception as exc:
            print(f"  PyMuPDF error: {exc} → Gemini fallback")

    # --- Tier 2: Gemini page images (needs PyMuPDF for rendering) ---
    if PYMUPDF_AVAILABLE and gemini_api_key:
        try:
            print("  Trying Gemini page-by-page extraction…")
            return extract_text_gemini_pages(pdf_path, gemini_api_key)
        except Exception as exc:
            print(f"  Gemini page-image failed: {exc}")

    # --- Tier 3: Gemini direct PDF ---
    if gemini_api_key:
        try:
            print("  Trying Gemini direct PDF upload…")
            return extract_text_gemini_direct(pdf_path, gemini_api_key)
        except Exception as exc:
            print(f"  Gemini direct failed: {exc}")

    raise RuntimeError(
        f"Cannot extract text from {pdf_path}. "
        "Install PyMuPDF (pip install PyMuPDF) or provide a GEMINI_API_KEY."
    )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text, source_path, chunk_size=500, overlap=50):
    """Split text into overlapping chunks of ~chunk_size words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    chunk_id = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_content = " ".join(chunk_words)

        chunks.append(
            {
                "id": f"{Path(source_path).stem}_chunk_{chunk_id:04d}",
                "source": str(source_path),
                "chunk_index": chunk_id,
                "content": chunk_content,
                "word_count": len(chunk_words),
                "type": "rag",
            }
        )

        step = max(chunk_size - overlap, 1)
        i += step
        chunk_id += 1

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_pdf(pdf_path, gemini_api_key, chunk_size=500, overlap=50, md_output_dir=None):
    """Extract text from PDF, save as .md, return chunks."""
    pages = extract_text_from_pdf(pdf_path, gemini_api_key)

    # Save extracted markdown — use custom dir or default beside the PDF
    if md_output_dir:
        md_dir = Path(md_output_dir)
    else:
        md_dir = Path(pdf_path).parent / "extracted_md"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / (Path(pdf_path).stem + ".md")

    parts = [f"<!-- Page {p['page']} -->\n{p['text']}" for p in pages]
    full_text = "\n\n".join(parts)
    md_path.write_text(full_text, encoding="utf-8")
    print(f"  Saved markdown → {md_path}")

    chunks = chunk_text(full_text, pdf_path, chunk_size, overlap)
    return chunks, full_text, len(pages)


def chunk_markdown(md_path, chunk_size=500, overlap=50):
    """Read a markdown file and chunk it."""
    text = Path(md_path).read_text(encoding="utf-8")
    chunks = chunk_text(text, md_path, chunk_size, overlap)
    return chunks, text


def count_words(file_path):
    """Count words in a document."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                text = "".join(page.get_text() for page in doc)
                doc.close()
                return len(text.split())
            except Exception:
                pass
        return 10_000  # assume large if unreadable
    if ext in (".md", ".txt"):
        return len(Path(file_path).read_text(encoding="utf-8").split())
    return 0


def get_page_count(file_path):
    """Get page count for PDFs."""
    if Path(file_path).suffix.lower() == ".pdf" and PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(file_path)
            n = len(doc)
            doc.close()
            return n
        except Exception:
            return 0
    return 0


# ---------------------------------------------------------------------------
# Embeddings via Gemini gemini-embedding-2-preview (3072 dims)
# ---------------------------------------------------------------------------

EMBED_MODEL = "models/gemini-embedding-2-preview"
EMBED_MODEL_SHORT = "gemini-embedding-2-preview"
EMBED_DIMS = 3072
EMBED_BATCH_SIZE = 100  # Gemini batchEmbedContents limit

# gemini-embedding-2-preview uses instruction prefix instead of taskType
DOC_PREFIX = "Represent this document for retrieval: "
QUERY_PREFIX = "Represent this query for retrieval: "


def get_embedding(text, gemini_api_key, is_query=False):
    """Get a single 3072-dim embedding from gemini-embedding-2-preview."""
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    truncated = prefix + " ".join(text.split()[:2048])
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EMBED_MODEL_SHORT}:embedContent?key={gemini_api_key}"
    )
    payload = {
        "content": {"parts": [{"text": truncated}]},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def get_embeddings_batch(texts, gemini_api_key, is_query=False, batch_pause=0.1):
    """
    Get embeddings using Gemini's batchEmbedContents API.
    Sends up to 100 texts per request — dramatically faster than one-at-a-time.
    """
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    embeddings = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]

        requests_payload = [
            {
                "model": EMBED_MODEL,
                "content": {"parts": [{"text": prefix + " ".join(t.split()[:2048])}]},
            }
            for t in batch
        ]

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{EMBED_MODEL_SHORT}:batchEmbedContents?key={gemini_api_key}"
        )

        for attempt in range(3):
            try:
                resp = requests.post(url, json={"requests": requests_payload}, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                for emb_obj in result["embeddings"]:
                    embeddings.append(emb_obj["values"])
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  WARN: batch {start//EMBED_BATCH_SIZE} failed: {exc}, filling zeros")
                    embeddings.extend([[0.0] * EMBED_DIMS] * len(batch))
                else:
                    time.sleep(2 ** attempt)

        done = min(start + EMBED_BATCH_SIZE, total)
        print(f"  Embedded {done}/{total} ({done*100//total}%)")

        if batch_pause and done < total:
            time.sleep(batch_pause)

    return embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk documents for RAG")
    parser.add_argument("file_path", help="Path to the document")
    parser.add_argument("--output", "-o", default="data/chunks.json")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--embed", action="store_true", help="Generate Gemini embeddings")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    fp = args.file_path
    ext = Path(fp).suffix.lower()

    if ext == ".pdf":
        chunks, text, page_count = chunk_pdf(fp, gemini_key, args.chunk_size, args.overlap)
    elif ext in (".md", ".txt"):
        chunks, text = chunk_markdown(fp, args.chunk_size, args.overlap)
    else:
        print(f"Unsupported file type: {ext}")
        sys.exit(1)

    # Optionally embed
    if args.embed and gemini_key:
        print(f"  Generating embeddings for {len(chunks)} chunks…")
        texts = [c["content"] for c in chunks]
        embs = get_embeddings_batch(texts, gemini_key)
        for chunk, emb in zip(chunks, embs):
            chunk["embedding"] = emb

    # Merge into output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))

    # Remove old chunks from same source
    existing = [c for c in existing if c.get("source") != str(fp)]
    existing.extend(chunks)

    output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"  {len(chunks)} chunks from {fp}")
    print(f"  Total chunks in index: {len(existing)}")
