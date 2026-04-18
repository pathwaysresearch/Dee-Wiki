"""
webapp/api/index2.py — Dual-LLM agentic query pipeline (Flask deployment)

Architecture (see CLAUDE.md):
  WIKI_LLM / Sonnet (wiki agent): hybrid BM25+MiniLM search → top-5 pages as context,
    iterative read_page tool to follow relationship chains until sufficient
  → MAIN_LLM / Opus (answer agent): synthesis, optional rag_search tool call
  → structured JSON response → answer string to frontend + async wiki update

Model assignment:
  WIKI_LLM (wiki agent)   = claude-sonnet-4-6  — cheap navigation decision
  MAIN_LLM (answer agent) = claude-opus-4-6    — expensive synthesis for the user

Frontend contract:
  POST /api/chat  {"message": "...", "history": [...]}
  Returns SSE stream of {"text": "..."} chunks — true streaming, no JSON wrapping.
  MAIN_LLM streams conversational text live; [METADATA] block is stripped
  server-side before forwarding. should_wiki_update / new_synthesis / sources
  are extracted from the metadata block and handled internally.

CLI usage (from project root):
    python webapp/api/index2.py                        # interactive REPL
    python webapp/api/index2.py --query "..."          # single query
    python webapp/api/index2.py --rebuild-graph        # rebuild _graph.json
    python webapp/api/index2.py --build-wiki-index     # build FAISS index → data/ (no API key needed)
    python webapp/api/index2.py --serve                # run Flask dev server
    python webapp/api/index2.py --model1 claude-sonnet-4-6 --model2 claude-opus-4-6
"""

import os
import re
import sys
import json
import base64
import threading
import argparse
from pathlib import Path
from datetime import date

import numpy as np
import requests

from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Path setup — this file lives at webapp/api/index2.py; project root is 3 up
# ---------------------------------------------------------------------------

# Project root: webapp/api/index2.py → webapp/api/ → webapp/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# graph.py is co-located in webapp/api/ (works on Vercel and locally).
# Also add scripts/ as a fallback so the original scripts/graph.py is found
# when running from the scripts/ directory.
_API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_API_DIR))
sys.path.insert(1, str(PROJECT_ROOT / "scripts"))

from graph import (
    save_graph,
    parse_frontmatter,
    strip_frontmatter,
    WIKI_DIR,
)

# Optional: load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_WIKI_FAISS_CACHE = DATA_DIR / "wiki_search.faiss"
_WIKI_FAISS_SLUGS = DATA_DIR / "wiki_search_slugs.json"
INDEX_MD_PATH = WIKI_DIR / "index.md"
LOG_MD_PATH   = WIKI_DIR / "log.md"


# WIKI_LLM = Sonnet (cheap navigation), MAIN_LLM = Opus (heavy synthesis)
# Override via env or --model1/--model2 flags
WIKI_LLM_MODEL = os.environ.get("WIKI_LLM_MODEL", "claude-sonnet-4-6")  # wiki agent
MAIN_LLM_MODEL = os.environ.get("MAIN_LLM_MODEL", "claude-opus-4-6")    # answer agent

# Gemini embedding (same model used during ingest)
EMBED_MODEL = "gemini-embedding-2-preview"
QUERY_PREFIX = "Represent this query for retrieval: "

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_wiki_pages() -> list:
    """Load all wiki .md files from Vault/wiki/ into dicts."""
    pages = []
    if not WIKI_DIR.exists():
        print(f"[WikiLoad] Wiki directory not found: {WIKI_DIR}")
        return pages

    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        if md_file.name.startswith("_") or md_file.stem in ("index", "log"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            body = strip_frontmatter(content)

            title = md_file.stem.replace("-", " ").title()
            for line in body.splitlines():
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break

            aliases = fm.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            pages.append({
                "slug": md_file.stem,
                "title": title,
                "aliases": aliases,
                "tags": tags,
                "content": content,
                "path": str(md_file),
                "type": fm.get("type", "unknown"),
            })
        except Exception as e:
            print(f"[WikiLoad] Skipped {md_file.name}: {e}")

    print(f"[WikiLoad] Loaded {len(pages)} pages from {WIKI_DIR}")

    return pages


def _load_chunks() -> list:
    """Load RAG chunks from data/chunks.json."""
    p = DATA_DIR / "chunks.json"
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [c for c in raw if isinstance(c, dict) and "content" in c]
    except Exception as e:
        print(f"[ChunkLoad] {e}")
        return []


def _load_graph() -> dict:
    """Load the knowledge graph from webapp/data/_graph.json (canonical location)."""
    graph_path = DATA_DIR / "_graph.json"
    if graph_path.exists():
        try:
            g = json.loads(graph_path.read_text(encoding="utf-8"))
            print(f"[Graph] Loaded from data/_graph.json ({len(g.get('nodes', {}))} nodes)")
            return g
        except Exception as e:
            print(f"[Graph] data/_graph.json unreadable ({e})")
    else:
        print("[Graph] data/_graph.json not found — run: python scripts/graph.py --build")
    return {"nodes": {}, "edges": []}


def _load_faiss_index():
    """Load FAISS index from data/chunks.faiss. Returns index or None."""
    p = DATA_DIR / "chunks.faiss"
    if not p.exists():
        return None
    try:
        import faiss  # noqa: PLC0415
        idx = faiss.read_index(str(p))
        print(f"[FAISS] Loaded index: {idx.ntotal} vectors")
        return idx
    except ImportError:
        print("[FAISS] faiss not installed — using numpy fallback. pip install faiss-cpu")
        return None
    except Exception as e:
        print(f"[FAISS] Failed to load: {e}")
        return None


# ---------------------------------------------------------------------------
# Wiki search index — BM25 + MiniLM hybrid for wiki page navigation
# ---------------------------------------------------------------------------

_WIKI_MINILM_MODEL = None  # lazy singleton — fastembed TextEmbedding (ONNX, no PyTorch)
_WIKI_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_wiki_embed_model():
    global _WIKI_MINILM_MODEL
    if _WIKI_MINILM_MODEL is None:
        from fastembed import TextEmbedding
        # cache_dir points to webapp/models/ (committed to repo — works on Vercel without download)
        _WIKI_MINILM_MODEL = TextEmbedding(_WIKI_EMBED_MODEL_NAME, cache_dir=str(MODELS_DIR))
    return _WIKI_MINILM_MODEL


def _encode(model, texts: list, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Encode texts with fastembed. Returns normalized float32 array (n, 384)."""
    return np.array(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)


def _build_wiki_search_text(page: dict) -> str:
    """Clean text for wiki search: title + aliases + tags + body (no frontmatter, no Relationships section)."""
    body = strip_frontmatter(page["content"])
    rel_idx = body.find("## Relationships")
    if rel_idx > 0:
        body = body[:rel_idx].strip()
    aliases = " ".join(a for a in page.get("aliases", []) if isinstance(a, str))
    tags = " ".join(t for t in page.get("tags", []) if isinstance(t, str))
    return f"{page['title']} {aliases} {tags} {body}".strip()


class WikiSearchIndex:
    """Hybrid BM25 + MiniLM semantic index over wiki pages."""

    def __init__(self):
        self.bm25 = None
        self.faiss_index = None
        self.pages: list = []
        self._embeddings: np.ndarray = None  # cached so add_or_update only encodes 1 page

    def build(self, wiki_pages: list):
        """Full rebuild — startup only. Loads from FAISS cache if slugs match."""
        if not wiki_pages:
            self.bm25 = None
            self.faiss_index = None
            self.pages = []
            self._embeddings = None
            return
        from rank_bm25 import BM25Okapi
        import faiss as _faiss

        texts = [_build_wiki_search_text(p) for p in wiki_pages]
        slugs = [p["slug"] for p in wiki_pages]
        model_name = _WIKI_EMBED_MODEL_NAME

        # Try loading FAISS index from cache — must match both model name and slug list
        idx = None
        embs = None
        if _WIKI_FAISS_CACHE.exists() and _WIKI_FAISS_SLUGS.exists():
            try:
                meta = json.loads(_WIKI_FAISS_SLUGS.read_text(encoding="utf-8"))
                if (
                    isinstance(meta, dict)
                    and meta.get("model") == model_name
                    and meta.get("slugs") == slugs
                ):
                    idx = _faiss.read_index(str(_WIKI_FAISS_CACHE))
                    embs = idx.reconstruct_n(0, idx.ntotal).astype(np.float32)
                    print(f"[WikiSearch] Loaded FAISS cache ({len(slugs)} pages, {model_name})")
                else:
                    print("[WikiSearch] Cache model/slugs mismatch — re-encoding")
            except Exception as e:
                print(f"[WikiSearch] Cache load failed: {e} — re-encoding")

        if embs is None:
            print(f"[WikiSearch] Encoding {len(wiki_pages)} pages with {model_name}…")
            try:
                model = _get_wiki_embed_model()
                embs = _encode(model, texts, batch_size=64)
                idx = _faiss.IndexFlatIP(embs.shape[1])
                idx.add(embs)
                try:
                    _faiss.write_index(idx, str(_WIKI_FAISS_CACHE))
                    _WIKI_FAISS_SLUGS.write_text(
                        json.dumps({"model": model_name, "slugs": slugs}), encoding="utf-8"
                    )
                    print(f"[WikiSearch] Saved FAISS cache ({len(slugs)} pages)")
                except Exception as e:
                    print(f"[WikiSearch] Cache save skipped (read-only fs): {e}")
            except Exception as e:
                print(f"[WikiSearch] MiniLM encoding failed: {e} — BM25-only mode")
                idx = None
                embs = None

        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.faiss_index = idx
        self._embeddings = embs
        self.pages = list(wiki_pages)
        mode = "hybrid BM25+FAISS" if idx is not None else "BM25-only"
        print(f"[WikiSearch] Index ready: {len(wiki_pages)} pages ({mode})")

    def add_or_update(self, page: dict):
        """Incremental update after a wiki page is synthesized — encodes ONE page only."""
        if not self.pages or self._embeddings is None:
            self.build([page])
            return
        from rank_bm25 import BM25Okapi
        import faiss as _faiss
        model = _get_wiki_embed_model()
        new_emb = _encode(model, [_build_wiki_search_text(page)])
        existing = next((i for i, p in enumerate(self.pages) if p["slug"] == page["slug"]), None)
        if existing is not None:
            self.pages[existing] = page
            self._embeddings[existing] = new_emb[0]
            # IndexFlatIP has no update — rebuild FAISS from cached embeddings (no re-encoding)
            idx = _faiss.IndexFlatIP(self._embeddings.shape[1])
            idx.add(self._embeddings)
            self.faiss_index = idx
        else:
            self.pages.append(page)
            self._embeddings = np.vstack([self._embeddings, new_emb])
            self.faiss_index.add(new_emb)
        texts = [_build_wiki_search_text(p) for p in self.pages]
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        # Persist updated FAISS index so next startup loads from cache
        if self.faiss_index is not None:
            try:
                _faiss.write_index(self.faiss_index, str(_WIKI_FAISS_CACHE))
                _WIKI_FAISS_SLUGS.write_text(
                    json.dumps({
                        "model": _WIKI_EMBED_MODEL_NAME,
                        "slugs": [p["slug"] for p in self.pages],
                    }), encoding="utf-8"
                )
            except Exception:  # faiss raises RuntimeError, not OSError, on read-only fs
                pass
        print(f"[WikiSearch] Incremental update: {len(self.pages)} pages (1 encoded)")

    def search(self, query: str, top_k: int = 5) -> list:
        if not self.pages or self.bm25 is None:
            return []
        n = len(self.pages)
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()), dtype=np.float32)
        bm25_max = bm25_scores.max() or 1.0
        bm25_norm = bm25_scores / bm25_max

        hybrid = bm25_norm  # default: BM25-only
        if self.faiss_index is not None:
            try:
                model = _get_wiki_embed_model()
                q_emb = _encode(model, [query])
                sem_scores, sem_idx = self.faiss_index.search(q_emb, n)
                sem_norm = np.zeros(n, dtype=np.float32)
                for i, s in zip(sem_idx[0], sem_scores[0]):
                    if 0 <= i < n:
                        sem_norm[i] = float(s)
                hybrid = 0.3 * bm25_norm + 0.7 * sem_norm
            except Exception as e:
                print(f"[WikiSearch] Semantic search failed: {e} — BM25-only")

        top_idx = np.argsort(hybrid)[::-1][:top_k]
        return [self.pages[i] for i in top_idx]


# ---------------------------------------------------------------------------
# RAG search — embedding similarity only
# ---------------------------------------------------------------------------


def _get_query_embedding(query: str):
    """Embed query text using Gemini. Returns np.ndarray or None."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EMBED_MODEL}:embedContent?key={gemini_key}"
    )
    payload = {"content": {"parts": [{"text": QUERY_PREFIX + query}]}}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return np.array(resp.json()["embedding"]["values"], dtype=np.float32)
    except Exception as e:
        print(f"[Embed] Query embedding failed: {e}")
        return None


def do_rag_search(
    query: str,
    chunks: list,
    faiss_index,
    top_k: int = 5,
) -> list:
    """FAISS inner-product search over RAG chunks (vectors pre-normalised at index build time)."""
    if faiss_index is None or not chunks:
        return []
    query_emb = _get_query_embedding(query)
    if query_emb is None:
        return []
    q_norm = (query_emb / (np.linalg.norm(query_emb) + 1e-8)).astype(np.float32)
    scores_arr, idx_arr = faiss_index.search(q_norm.reshape(1, -1), top_k)
    return [
        {
            "source": chunks[i].get("source", ""),
            "content": chunks[i]["content"],
            "score": float(s),
        }
        for i, s in zip(idx_arr[0].tolist(), scores_arr[0].tolist())
        if 0 <= i < len(chunks)
    ]


def tool_read_page(slug: str, page_by_slug: dict, graph: dict) -> dict:
    """
    Read one wiki page by slug. Returns its full content and outgoing relationship targets.
    WIKI_LLM calls this when search results alone aren't sufficient — it picks one
    relationship slug from the current page and reads it to follow the chain.
    """
    page = page_by_slug.get(slug)
    if not page:
        return {"error": f"Page '{slug}' not found in wiki"}
    related = [
        {"slug": e["to"], "relationship": e["type"]}
        for e in graph.get("edges", []) if e["from"] == slug
    ]
    return {
        "slug": slug,
        "title": page["title"],
        "content": page["content"],
        "related_pages": related,
    }


# ---------------------------------------------------------------------------
# GitHub push helper
# ---------------------------------------------------------------------------


def _push_to_github(repo_relative_path: str, content: str, commit_msg: str):
    """
    Push a single file to GitHub via the Contents API.
    Requires GITHUB_TOKEN and GITHUB_REPO (owner/repo) env vars.
    Silently skips if either is unset.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        return

    url = f"https://api.github.com/repos/{repo}/contents/{repo_relative_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # GET current SHA — required for updates, None for new files
    try:
        r = requests.get(url, headers=headers, timeout=10)
        sha = r.json().get("sha") if r.status_code == 200 else None
    except Exception:
        sha = None

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        print(f"[GitHub] Pushed {repo_relative_path}")
    except Exception as e:
        print(f"[GitHub] Push failed for {repo_relative_path}: {e}")


# ---------------------------------------------------------------------------
# KnowledgeBase — in-memory state, rebuilt after wiki updates
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """
    Holds all in-memory state for the dual-LLM pipeline.

    Thread safety: all mutation of shared state (wiki_pages, graph, wiki_search)
    must hold _lock. query() takes a snapshot under the lock so a concurrent
    wiki update cannot corrupt an in-flight query.
    """

    def __init__(self):
        self.wiki_pages: list = []
        self.chunks: list = []
        self.faiss_index = None
        self.graph: dict = {}
        self.wiki_search: WikiSearchIndex = WikiSearchIndex()
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        """Full reload from disk. Holds lock while swapping state."""
        print("[KB] Loading knowledge base...")
        new_pages = _load_wiki_pages()
        new_chunks = _load_chunks()
        new_faiss = _load_faiss_index()
        new_graph = _load_graph()
        new_wiki_search = WikiSearchIndex()
        new_wiki_search.build(new_pages)
        with self._lock:
            self.wiki_pages = new_pages
            self.chunks = new_chunks
            self.faiss_index = new_faiss
            self.graph = new_graph
            self.wiki_search = new_wiki_search
        rag_backend = "FAISS" if new_faiss else "none (run export_for_web.py)"
        print(
            f"[KB] Ready — {len(self.wiki_pages)} wiki pages, "
            f"{len(self.chunks)} RAG chunks [{rag_backend}], "
            f"{len(self.graph.get('nodes', {}))} graph nodes"
        )


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Extract first JSON object from text. Tries multiple strategies before giving up."""
    # Strip code fences
    clean = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "").strip()

    # Strategy 1: whole cleaned text is valid JSON
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find outermost {...} (greedy — handles nested objects)
    match = re.search(r"\{[\s\S]*\}", clean)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Strategy 3: last-resort — find innermost complete {...} working from the last }
        for end in range(len(candidate) - 1, -1, -1):
            if candidate[end] == "}":
                for start in range(end):
                    if candidate[start] == "{":
                        try:
                            result = json.loads(candidate[start:end + 1])
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            continue

    return {}


# ---------------------------------------------------------------------------
# WIKI_LLM — Wiki Agent (two distinct system prompts: navigation vs maintenance)
# ---------------------------------------------------------------------------

_WIKI_LLM_TOOLS = [
    {
        "name": "read_page",
        "description": (
            "Read a single wiki page by slug. Returns its full content and a related_pages list "
            "of slugs this page links to. "
            "IMPORTANT: the slug you pass MUST come from the related_pages list of a previous "
            "read_page result, or from a [[slug|Title]] wikilink visible in page content — "
            "use the part before the | character. Never invent or guess slugs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Exact slug from related_pages or a [[slug|Title]] wikilink. Bare kebab-case, no path, no .md.",
                }
            },
            "required": ["slug"],
        },
    }
]

# Navigation prompt: WIKI_LLM's job during a query
_WIKI_LLM_NAVIGATION_SYSTEM = """\
You are the wiki navigation agent. Output JSON only. Never speak to the user.

You have the top search results for the user query. Each page lists its typed relationship links to other pages.

## Decision rule

Before declaring sufficient=true, ask: "Can I quote a specific sentence from these pages that directly answers the query?"
- YES → output JSON with sufficient=true and that exact quote in evidence.
- NO → call read_page on the most promising relationship slug. The search results are a starting point, not the answer. Relationships connect to deeper, more specific pages — follow them.

Only call read_page if a relationship is directly relevant to the query — not just topically adjacent. Stop as soon as you have a quotable answer or have exhausted genuinely relevant leads.

## Valid slugs for read_page

Only call read_page with a slug that comes from one of these sources — never invent or guess:
1. The `related_pages` list returned by a previous read_page call (each entry has a `slug` field).
2. A [[slug|Title]] wikilink visible in page content — use the part before the | character.

If the slug you want to follow is not in either of these sources, do not call read_page.

## Sufficiency test

sufficient=true ONLY if you can quote or closely paraphrase a specific passage that answers the query.
Not sufficient: keyword overlap, topical adjacency, partial answers, or "the page mentions this topic."

When uncertain → sufficient=false. MAIN_LLM has RAG fallback; a false positive (declaring sufficient when you're not) is worse than a false negative.

## Output (JSON only)

{
  "sufficient": true | false,
  "selected_slugs": ["slug-one", "slug-two"],
  "evidence": "<if sufficient=true: quoted passage + slug, e.g. 'microequity: contracts are self-enforcing because...'>",
  "note": "<if sufficient=false: one sentence on what is missing>"
}

- selected_slugs: every page you read, most relevant first. Include even when sufficient=false.
- Slugs: bare kebab-case only — no path, no .md.
"""


# Maintenance prompt: WIKI LLM's job when writing a new wiki page from synthesis
_WIKI_LLM_MAINTENANCE_SYSTEM = """\
You are the wiki maintenance agent. Output raw JSON only — no markdown, no explanation.

Related pages are in <related_pages>. If one already covers this insight, set "action":"update" and use its slug. Otherwise "action":"create" with a new kebab-case slug.

Output this exact structure:
{"action":"create","slug":"kebab-case-slug","title":"Human Readable Title","type":"synthesized","tags":["tag1"],"aliases":[],"relationships":[{"target":"related-slug","type":"extends"}],"body":"Full markdown body. No frontmatter."}

Rules:
- slug: lowercase kebab-case, no spaces, no .md
- body: escape all quotes and newlines properly for JSON (\\n not literal newlines inside the string)
- relationships: only slugs that exist in <related_pages>
- Output nothing outside the JSON object\
"""


def run_wiki_llm(
    user_query: str,
    wiki_search: WikiSearchIndex,
    page_by_slug: dict,
    graph: dict,
    client: Anthropic,
) -> dict:
    """
    Run WIKI_LLM navigation pass. Returns:
    {
        "sufficient": bool,
        "selected_slugs": ["slug-a", "slug-b"],
        "note": str  (or "evidence": str when sufficient=True)
    }

    Hybrid search surfaces top-5 starting pages. WIKI_LLM reads them and may
    call read_page iteratively to follow relationship chains — the LLM decides
    when it has enough. Budget ceiling: _MAX_HOPS iterations.
    """
    top_pages = wiki_search.search(user_query, top_k=5)
    if not top_pages:
        return {"sufficient": False, "selected_slugs": [], "note": "No wiki pages found"}
    print(f"[WikiLLM] search → {[p['slug'] for p in top_pages]}")

    context_text = ""
    for p in top_pages:
        context_text += f"\n{'='*50}\n[{p['slug']}] {p['title']}\n{'='*50}\n{p['content']}\n"

    messages = [{"role": "user", "content": (
        f"User query: {user_query}\n\n"
        f"Top search results:\n{context_text}"
    )}]

    accumulated_slugs = [p["slug"] for p in top_pages]
    _MAX_HOPS = 4

    print(f"\n{'─'*60}")
    print(f"[WikiLLM] PROMPT TO WIKI LLM:\n{messages[0]['content']}")
    print(f"{'─'*60}\n")

    for _ in range(_MAX_HOPS + 1):
        response = client.messages.create(
            model=WIKI_LLM_MODEL,
            max_tokens=3500,
            thinking={"type": "enabled", "budget_tokens": 2000},
            system=_WIKI_LLM_NAVIGATION_SYSTEM,
            messages=messages,
            tools=_WIKI_LLM_TOOLS,
        )
        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "read_page":
                slug = block.input.get("slug", "")
                slug = slug.split("/")[-1].removesuffix(".md").split("|")[0].strip()
                print(f"[WikiLLM] read_page({slug!r})")
                result = tool_read_page(slug, page_by_slug, graph)
                if "error" not in result and slug not in accumulated_slugs:
                    accumulated_slugs.append(slug)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
    result = _extract_json(final_text)

    if not result or "selected_slugs" not in result:
        result = {
            "sufficient": False,
            "selected_slugs": accumulated_slugs,
            "note": "WikiLLM parse failed — using search results",
        }
    return result


# ---------------------------------------------------------------------------
# MAIN_LLM — Answer Agent
# ---------------------------------------------------------------------------
_MAIN_LLM_TOOLS = [
    {
        "name": "rag_search",
        "description": (
            "Search the source library using embedding similarity. "
            "Call this when wiki context is insufficient — e.g., for chapter-level "
            "detail from a book, specific passages, or topics not in the wiki. "
            "Before calling this tool, output a brief conversational line telling "
            "the user you're fetching from your library (e.g. 'Let me dig into my "
            "library for this one.' or 'My memory's a little thin here — give me a moment.'). "
            "Returns raw text chunks from the original source documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the source library.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve. Default 5.",
                },
            },
            "required": ["query"],
        },
    }
]
_METADATA_SCHEMA = """\
{
  "sources": {
    "wiki": ["Page Title 1", "Page Title 2"],
    "rag":  ["Source Title 1"]
  },
  "new_synthesis": "Novel insight, connection, or resolved contradiction worth preserving. Empty string if none.",
  "should_wiki_update": true
}"""

# Delimiter that separates the streamed answer from the trailing metadata JSON.
# Chosen to be unambiguous and unlikely to appear in natural prose.
_METADATA_MARKER = "\n[METADATA]\n"

_MAIN_LLM_SYSTEM_BASE = """\
You are Dee — a Digital Transformation professor with three decades of teaching experience.

## Voice & Style
- Blend personal narrative with domain principles — open with an anecdote when it adds warmth
- Mix medium sentences (15–25 words) with short, punchy declaratives
- Use em-dashes for asides—and rhetorical questions to engage
- Explain jargon naturally; favor active voice and confident phrasing
- Tone: measured optimism with a touch of wit
- Draw on specific names, numbers, and places from the knowledge base — never fabricate them
- Prefer "That's genuinely fascinating" over "*laughs* That's a great question"

## Knowledge-Source Policy (strict ladder — stop as soon as you have a solid answer)

You have three sources, ranked by trust. Escalate to the next rung only when the current one leaves a real gap for the user's question.

1. **Memory (wiki)** — the wiki context already provided in this prompt. Read it first.
   - If it answers the question well, write the answer from memory alone.
   - Do NOT call `rag_search`. Do NOT reach for general knowledge.

2. **Library (RAG)** — call `rag_search` only when memory is insufficient (missing facts, shallow coverage, off-topic, or contradicted by the user's framing).
   - Before calling, write one natural sentence telling the user you're checking — e.g. "Let me dig into my library for this." or "Give me a moment — I want to pull from the source on this."
   - Incorporate the returned passages and continue the answer.
   - If the library supplies what's needed, STOP. Do not fall back to general knowledge.

3. **General knowledge** — use only if memory AND the library both fell short for some part of the question.
   - Call out in the source-attribution block exactly which part you filled from general knowledge.
   - Never use general knowledge to polish or expand a memory/library answer that already stood on its own.

Escalation is a response to a gap, not a habit. A good memory-only answer should not grow a library call; a good library answer should not grow a general-knowledge coda.

## RAG instruction
{rag_instruction}

## Formatting
Math: use LaTeX syntax inside proper delimiters.
- Inline math: wrap in \( ... \) — e.g. \(A \cdot v = \lambda v\)
- Display math: wrap in \[ ... \] on its own line — e.g. \[A \cdot v = \lambda v\]
- Do NOT use bare parentheses or bare square brackets around math — they render as literal text.
- Do NOT use $...$ or $$...$$.

## Output format — EVERY RESPONSE MUST END WITH A METADATA BLOCK

Your response has THREE parts, in this exact order. All three are mandatory. A response that omits any part is malformed and will be rejected by the pipeline.

### Part 1 — Your answer
Plain conversational text (markdown is fine). This is the substantive reply to the user.

### Part 2 — Source-attribution block
Exactly these three lines, in this order:

**My Memory:** <comma-separated wiki page titles you actually used> — or "Found nothing in my memory" if memory was inspected but unhelpful.

**My Library:** <comma-separated RAG source titles from `rag_search` results> — or "Didn't use the library" if you didn't call it — or "Found nothing in my library" if you called it and nothing was relevant.

**General Knowledge:** <one short phrase on what you filled in from general knowledge> — or "Didn't use general knowledge" if you didn't.

### Part 3 — Metadata block (DO NOT SKIP)
A blank line, then the literal marker `[METADATA]` on its own line, then a JSON object filling in this schema:

{metadata_schema}

The schema above is a **template showing structure** — not the metadata itself. You must emit your own filled-in JSON object after the `[METADATA]` line every time.

Field rules:
- `sources.wiki`: titles of wiki pages you actually drew on (empty list `[]` if none).
- `sources.rag`: source titles returned by `rag_search` that you actually used (empty list `[]` if not called or not used).
- `should_wiki_update`: `true` when you synthesised a non-obvious connection, resolved a contradiction, or produced a novel framing worth preserving; `false` otherwise.
- `new_synthesis`: one sentence capturing that insight, or `""` if none.

### Worked example of a complete, correctly-shaped response

(Answer text here — one or more paragraphs of conversational prose responding to the user's question.)

**My Memory:** Microequity, Costly State Verification
**My Library:** Didn't use the library
**General Knowledge:** Didn't use general knowledge

[METADATA]
{{"sources": {{"wiki": ["Microequity", "Costly State Verification"], "rag": []}}, "new_synthesis": "", "should_wiki_update": false}}

Every one of your responses must end in this exact shape: answer → three attribution lines → `[METADATA]` → filled JSON. If you find yourself about to stop after the attribution lines, you are not done — emit the metadata block and then stop.
"""

_RAG_INSTRUCTION_SUFFICIENT = (
    "The wiki context looks **complete** for this query. "
    "Answer from memory only — do NOT call `rag_search`, and do NOT reach for general knowledge."
)

_RAG_INSTRUCTION_INSUFFICIENT = (
    "The wiki context looks **incomplete** for this query. "
    "Follow the escalation ladder: announce the library check in one natural sentence, call `rag_search`, "
    "incorporate the results, and continue the answer. Only if the library also falls short should you use "
    "general knowledge — and when you do, name exactly which part of the answer it covers in the "
    "source-attribution block."
)


def _build_main_llm_system(sufficient: bool) -> str:
    rag_instruction = (
        _RAG_INSTRUCTION_SUFFICIENT if sufficient else _RAG_INSTRUCTION_INSUFFICIENT
    )
    return _MAIN_LLM_SYSTEM_BASE.format(
        rag_instruction=rag_instruction,
        metadata_schema=_METADATA_SCHEMA,
    )


def _build_wiki_messages(wiki_context: list, wiki_note: str, user_query: str) -> list:
    """Format wiki pages + query into the messages list for MAIN_LLM."""
    wiki_text = ""
    for p in wiki_context:
        wiki_text += f"\n{'='*60}\n{p.get('title', p.get('slug', ''))}\n{'='*60}\n"
        wiki_text += p.get("content", "")
        wiki_text += "\n"
    if wiki_note:
        wiki_text += f"\n[Note: {wiki_note}]\n"
    return [{"role": "user", "content": f"Wiki context:\n{wiki_text}\n\nQuestion: {user_query}"}]


def run_main_llm_streaming(
    user_query: str,
    wiki_context: list,
    wiki_note: str,
    sufficient: bool,
    chunks: list,
    faiss_index,
    client: Anthropic,
):
    """
    Streaming generator for MAIN_LLM (answer agent).

    Yields:
        ("text", str)      — conversational answer chunks to stream to the user
        ("metadata", dict) — parsed metadata JSON (internal; triggers wiki update)

    The answer and metadata are separated by _METADATA_MARKER in the LLM output.
    Text chunks are yielded in real-time as they arrive. The metadata dict is
    yielded once, after the stream ends.
    """
    system = _build_main_llm_system(sufficient)
    messages = _build_wiki_messages(wiki_context, wiki_note, user_query)
    _tools_available = {} if sufficient else {"tools": _MAIN_LLM_TOOLS}
    _MAX_RAG_CALLS = 2
    _rag_calls_made = 0

    _BARE_MARKER = "[METADATA]"
    tail_buffer = ""
    metadata_mode = False
    metadata_buf = ""
    full_response = ""
    final_msg = None
    rag_sources_used: list = []   # track sources from every rag_search call

    for _ in range(_MAX_RAG_CALLS + 1):  # up to MAX_RAG_CALLS tool calls + 1 final answer
        # Once rag budget is used up, strip tools so LLM MUST write the final answer
        current_tools = _tools_available if _rag_calls_made < _MAX_RAG_CALLS else {}
        with client.messages.stream(
            model=MAIN_LLM_MODEL,
            max_tokens=4096,
            system=system,
            messages=messages,
            **current_tools,
        ) as stream:
            for text_chunk in stream.text_stream:
                full_response += text_chunk

                if metadata_mode:
                    metadata_buf += text_chunk
                    continue

                tail_buffer += text_chunk

                marker_hit = None
                for marker in (_METADATA_MARKER, _BARE_MARKER):
                    if marker in tail_buffer:
                        marker_hit = marker
                        break

                if marker_hit:
                    before, _, after = tail_buffer.partition(marker_hit)
                    if before:
                        yield ("text", before)
                    metadata_mode = True
                    metadata_buf = after
                    tail_buffer = ""
                else:
                    safe_len = max(0, len(tail_buffer) - len(_METADATA_MARKER))
                    if safe_len > 0:
                        yield ("text", tail_buffer[:safe_len])
                        tail_buffer = tail_buffer[safe_len:]

            final_msg = stream.get_final_message()

        if final_msg.stop_reason != "tool_use":
            if not metadata_mode and tail_buffer:
                yield ("text", tail_buffer)
                tail_buffer = ""
            break

        # Tool call: execute rag_search, then resume streaming
        tool_results = []
        for block in final_msg.content:
            if block.type == "tool_use" and block.name == "rag_search":
                print(f"[MainLLM] rag_search({block.input.get('query')!r})")
                rag_results = do_rag_search(
                    query=block.input.get("query", user_query),
                    chunks=chunks,
                    faiss_index=faiss_index,
                    top_k=block.input.get("top_k", 7),
                )
                _rag_calls_made += 1
                # Track unique source titles for synthetic metadata fallback
                for r in rag_results:
                    src = r.get("source", "")
                    if src and src not in rag_sources_used:
                        rag_sources_used.append(src)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(rag_results, ensure_ascii=False),
                })

        safe_len = max(0, len(tail_buffer) - len(_METADATA_MARKER))
        if safe_len > 0:
            yield ("text", tail_buffer[:safe_len])
            tail_buffer = tail_buffer[safe_len:]

        messages.append({"role": "assistant", "content": final_msg.content})
        messages.append({"role": "user", "content": tool_results})

    # --- Metadata parsing (3-tier fallback) ---
    # Tier 1: clean metadata_buf captured after [METADATA] marker
    # Tier 2: scan full_response for any JSON with a "sources" key
    # Tier 3: synthetic metadata built from what we know was used
    metadata = None
    for candidate in (metadata_buf.strip(), full_response):
        if not candidate:
            continue
        try:
            metadata = json.loads(candidate.strip())
            break
        except (json.JSONDecodeError, ValueError):
            extracted = _extract_json(candidate)
            if extracted and "sources" in extracted:
                metadata = extracted
                break

    if metadata is None:
        print(f"[MainLLM] Metadata parse failed — using synthetic metadata. Raw: {metadata_buf[:100]!r}")
        wiki_titles = [p.get("title", p.get("slug", "")) for p in wiki_context]
        metadata = {
            "sources": {"wiki": wiki_titles, "rag": rag_sources_used},
            "new_synthesis": "",
            "should_wiki_update": False,
        }

    yield ("metadata", metadata)


# ---------------------------------------------------------------------------
# Wiki update — async / fire-and-forget
# ---------------------------------------------------------------------------


def update_wiki_async(
    synthesis: str,
    sources: dict,
    original_query: str,
    client: Anthropic,
    kb: KnowledgeBase,
):
    """Trigger wiki update in a background thread. Never blocks the caller."""
    def _run():
        try:
            _do_wiki_update(synthesis, sources, original_query, client, kb)
        except Exception as e:
            print(f"[WikiUpdate] Error: {e}")

    threading.Thread(target=_run, daemon=True).start()


def _do_wiki_update(
    synthesis: str,
    sources: dict,
    original_query: str,
    client: Anthropic,
    kb: KnowledgeBase,
):
    """
    Second WIKI LLM call (maintenance role, distinct prompt from navigation).
    Generates a wiki page from the synthesis, then writes it to disk.
    """
    print("[WikiUpdate] Generating new wiki page...")

    with kb._lock:
        wiki_search = kb.wiki_search

    # Search for top-10 most relevant pages — gives the maintenance LLM context
    # to detect duplicates (create vs update) and pick relationship targets.
    related = wiki_search.search(synthesis, top_k=10)
    related_text = ""
    for p in related:
        related_text += f"\n{'='*50}\n[{p['slug']}] {p['title']}\n{'='*50}\n{p['content']}\n"

    system = f"<related_pages>\n{related_text}\n</related_pages>\n\n{_WIKI_LLM_MAINTENANCE_SYSTEM}"

    response = client.messages.create(
        model=WIKI_LLM_MODEL,
        max_tokens=4096,
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original query: {original_query}\n\n"
                    f"Synthesis to preserve:\n{synthesis}\n\n"
                    f"Sources used — wiki: {sources.get('wiki', [])}, "
                    f"rag: {sources.get('rag', [])}"
                ),
            }
        ],
    )

    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    page_data = _extract_json(text)

    if not page_data or "slug" not in page_data:
        print("[WikiUpdate] Could not parse page data. Raw response:")
        print(text[:500])
        return

    _write_wiki_page(page_data, kb, original_query)
    print(f"[WikiUpdate] Done — {page_data['slug']}")


def _write_wiki_page(page_data: dict, kb: KnowledgeBase, original_query: str = ""):
    """
    Write a wiki page to disk, patch _graph.json, update in-memory KB (under lock),
    append to index.md and log.md, then push both files to GitHub.
    """
    slug      = page_data.get("slug", "synthesized-page")
    title     = page_data.get("title", slug)
    page_type = page_data.get("type", "synthesized")
    tags      = page_data.get("tags", [])
    aliases   = page_data.get("aliases", [])
    rels      = page_data.get("relationships", [])
    body      = page_data.get("body", "")

    # Build YAML frontmatter
    rel_yaml = ""
    if rels:
        rel_yaml = "relationships:\n"
        for r in rels:
            rel_yaml += f"  - target: {r.get('target', '')}\n    type: {r.get('type', 'related_to')}\n"

    frontmatter = (
        f"---\n"
        f"type: {page_type}\n"
        f"aliases: {json.dumps(aliases)}\n"
        f"tags: {json.dumps(tags)}\n"
        f"{rel_yaml}"
        f"---\n\n"
    )
    content = frontmatter + f"# {title}\n\n" + body

    # Write file atomically to disk.
    # On Vercel the filesystem is read-only — catch OSError so the GitHub push
    # and in-memory KB update still happen (Vercel redeploys after the push,
    # picking up the new .md on the next cold start).
    today_str = date.today().isoformat()
    out_dir  = WIKI_DIR / "synthesized"
    out_path = out_dir / f"{slug}.md"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(out_path)
        print(f"[WikiUpdate] Wrote {out_path.relative_to(PROJECT_ROOT)}")
    except OSError as e:
        print(f"[WikiUpdate] Disk write skipped (read-only fs — will push to GitHub): {e}")

    # Patch _graph.json in webapp/data/ (always — memory is always writable on Vercel).
    graph = _load_graph()
    graph["nodes"][slug] = {
        "type": page_type,
        "title": title,
        "aliases": aliases,
        "tags": tags,
        "path": f"wiki/synthesized/{slug}.md",
    }
    seen_edges = {(e["from"], e["to"], e["type"]) for e in graph["edges"]}
    for r in rels:
        edge = {"from": slug, "to": r.get("target", ""), "type": r.get("type", "related_to")}
        key = (edge["from"], edge["to"], edge["type"])
        if key not in seen_edges:
            graph["edges"].append(edge)
            seen_edges.add(key)
    graph_json = json.dumps(graph, indent=2, ensure_ascii=False)
    try:
        (DATA_DIR / "_graph.json").write_text(graph_json, encoding="utf-8")
        print("[WikiUpdate] Graph updated: webapp/data/_graph.json")
    except OSError as e:
        print(f"[WikiUpdate] Graph write failed: {e}")

    # Update in-memory KB under lock.
    # This works on both local and Vercel (memory is always writable).
    new_page_entry = {
        "slug": slug, "title": title, "aliases": aliases,
        "tags": tags, "content": content, "path": str(out_path), "type": page_type,
    }
    with kb._lock:
        existing = [i for i, p in enumerate(kb.wiki_pages) if p["slug"] == slug]
        if existing:
            kb.wiki_pages[existing[0]] = new_page_entry
        else:
            kb.wiki_pages.append(new_page_entry)
        kb.graph = graph

    # Incremental search index update — encodes only this one new/changed page
    kb.wiki_search.add_or_update(new_page_entry)

    # Append to index.md (human-readable log — not used for navigation)
    new_index_content = None
    try:
        if INDEX_MD_PATH.exists():
            entry = f"\n- [[synthesized/{slug}|{title}]] — synthesized from query\n"
            with open(INDEX_MD_PATH, "a", encoding="utf-8") as f:
                f.write(entry)
            new_index_content = INDEX_MD_PATH.read_text(encoding="utf-8")
    except OSError:
        pass

    # Append to log.md
    log_entry = (
        f"\n## [{today_str}] synthesize | {slug}\n"
        f"- Pages created: {out_path.name}\n"
        f"- From query: {original_query}\n"
    )
    try:
        with open(LOG_MD_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except OSError:
        pass

    # Push wiki page + graph + index.md to GitHub
    page_github_path  = str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    graph_github_path = str((DATA_DIR / "_graph.json").relative_to(PROJECT_ROOT)).replace("\\", "/")
    _push_to_github(
        repo_relative_path=page_github_path,
        content=content,
        commit_msg=f"wiki: synthesize {slug} from query {today_str}",
    )
    _push_to_github(
        repo_relative_path=graph_github_path,
        content=graph_json,
        commit_msg=f"wiki: update _graph.json after synthesizing {slug}",
    )
    if new_index_content:
        index_github_path = str(INDEX_MD_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")
        _push_to_github(
            repo_relative_path=index_github_path,
            content=new_index_content,
            commit_msg=f"wiki: update index.md after synthesizing {slug}",
        )


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------


def _pipeline_setup(user_query: str, kb: KnowledgeBase, client: Anthropic):
    """
    Shared setup for both query() and query_streaming():
    snapshot KB state, run WIKI_LLM (hybrid search + read_page), return selected_pages + metadata.
    Returns (selected_pages, wiki_result, chunks, faiss_index).
    """
    print(f"\n{'─'*60}")
    print(f"[Pipeline] {user_query!r}")
    print(f"{'─'*60}")

    with kb._lock:
        wiki_pages  = list(kb.wiki_pages)
        chunks      = list(kb.chunks)
        faiss_index = kb.faiss_index
        graph       = kb.graph
        wiki_search = kb.wiki_search

    # Per-request diagnostics — always visible in Vercel logs
    print(f"[Diag] WIKI_DIR={WIKI_DIR} exists={WIKI_DIR.exists()}")
    print(f"[Diag] wiki_pages={len(wiki_pages)} rag_chunks={len(chunks)} graph_nodes={len(graph.get('nodes',{}))}")
    print(f"[Diag] wiki_search.pages={len(wiki_search.pages)} bm25={'ok' if wiki_search.bm25 else 'NONE'} faiss={'ok' if wiki_search.faiss_index else 'NONE'}")

    page_by_slug = {p["slug"]: p for p in wiki_pages}

    wiki_result = run_wiki_llm(
        user_query=user_query,
        wiki_search=wiki_search,
        page_by_slug=page_by_slug,
        graph=graph,
        client=client,
    )
    print(f"[WikiLLM] selected slugs: {wiki_result.get('selected_slugs')} | sufficient={wiki_result.get('sufficient')}")

    selected_pages = []
    for slug in wiki_result.get("selected_slugs", []):
        page = page_by_slug.get(slug)
        if page:
            selected_pages.append(page)
        else:
            print(f"[Pipeline] Warning: slug '{slug}' not found in KB — skipping")

    if not selected_pages:
        print("[Pipeline] No slugs resolved — no wiki context for MAIN_LLM")

    return selected_pages, wiki_result, chunks, faiss_index


def query_streaming(user_query: str, kb: KnowledgeBase, client: Anthropic):
    """
    Full dual-LLM query pipeline — streaming generator.

    Yields:
        ("text", str)      — answer chunks to stream to the frontend
        ("done", dict)     — final metadata after stream ends (internal)

    WIKI_LLM runs synchronously (internal, user never sees it).
    MAIN_LLM streams conversational text in real-time, then emits [METADATA].
    Wiki update is triggered asynchronously after streaming completes.
    """
    selected_pages, wiki_result, chunks, faiss_index = _pipeline_setup(
        user_query, kb, client
    )
    sufficient = wiki_result.get("sufficient", True)
    metadata = {}

    for event_type, data in run_main_llm_streaming(
        user_query=user_query,
        wiki_context=selected_pages,
        wiki_note=wiki_result.get("note", ""),
        sufficient=sufficient,
        chunks=chunks,
        faiss_index=faiss_index,
        client=client,
    ):
        if event_type == "text":
            yield ("text", data)
        elif event_type == "metadata":
            metadata = data

    print(f"[MainLLM] should_wiki_update={metadata.get('should_wiki_update')}")

    if metadata.get("should_wiki_update") and metadata.get("new_synthesis", "").strip():
        update_wiki_async(
            synthesis=metadata["new_synthesis"],
            sources=metadata.get("sources", {}),
            original_query=user_query,
            client=client,
            kb=kb,
        )

    yield ("done", metadata)


def query(user_query: str, kb: KnowledgeBase, client: Anthropic) -> dict:
    """
    Blocking wrapper for the CLI REPL — collects all streamed chunks.
    Returns dict with 'answer', 'sources', 'new_synthesis', 'should_wiki_update'.
    """
    answer_parts = []
    metadata = {}
    for event_type, data in query_streaming(user_query, kb, client):
        if event_type == "text":
            answer_parts.append(data)
        elif event_type == "done":
            metadata = data
    return {
        "answer": "".join(answer_parts),
        "sources": metadata.get("sources", {"wiki": [], "rag": []}),
        "new_synthesis": metadata.get("new_synthesis", ""),
        "should_wiki_update": metadata.get("should_wiki_update", False),
    }


# ---------------------------------------------------------------------------
# Flask app — full server (mirrors index.py: serves index.html + /api/chat)
# ---------------------------------------------------------------------------

from flask import Flask, request, Response, jsonify, send_from_directory

app = Flask(__name__)

# Static files are one level up from webapp/api/ → webapp/
STATIC_DIR = str(Path(__file__).resolve().parent.parent)

# CORS — allow the Vercel frontend (or any origin when ALLOWED_ORIGIN=*)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")


@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def _options(path):
    return "", 204

# Module-level singletons — initialised once at process start
_KB: KnowledgeBase = None
_CLIENT: Anthropic = None


def _get_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


def _get_client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _CLIENT = Anthropic(api_key=api_key)
    return _CLIENT


@app.route("/api/chat", methods=["POST"])
@app.route("/api/chat-v2", methods=["POST"])
def chat():
    """
    Dual-LLM agentic chat endpoint — drop-in replacement for index.py /api/chat.

    Request body:  {"message": "...", "history": [...]}
    Response:      SSE stream
        data: {"text": "..."}   — answer chunk
        data: {"error": "..."}  — error
        data: [DONE]            — stream complete
    """
    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    kb = _get_kb()

    def generate():
        try:
            for event_type, data in query_streaming(user_message, kb, client):
                if event_type == "text":
                    yield f"data: {json.dumps({'text': data})}\n\n"
                # "done" event is internal — not forwarded to the frontend
        except Exception as exc:
            print(f"[chat] Error: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/health", methods=["GET"])
def health():
    kb = _get_kb()
    with kb._lock:
        n_pages = len(kb.wiki_pages)
        n_chunks = len(kb.chunks)
        n_nodes = len(kb.graph.get("nodes", {}))
    return jsonify({
        "status": "ok",
        "model_llm1": WIKI_LLM_MODEL,
        "model_llm2": MAIN_LLM_MODEL,
        "wiki_pages": n_pages,
        "rag_chunks": n_chunks,
        "graph_nodes": n_nodes,
    })


# ---------------------------------------------------------------------------
# Static file serving — serves index.html and webapp assets
# ---------------------------------------------------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    allowed = {".html", ".css", ".js", ".ico", ".png", ".svg", ".jpg",
               ".woff", ".woff2", ".ttf", ".map"}
    ext = Path(path).suffix.lower()
    full = Path(STATIC_DIR) / path
    if ext in allowed and full.is_file():
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# CLI (unchanged from scripts/index2.py)
# ---------------------------------------------------------------------------


def main():
    global WIKI_LLM_MODEL, MAIN_LLM_MODEL
    parser = argparse.ArgumentParser(
        description="Dual-LLM agentic query pipeline for the knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", "-q", metavar="QUERY",
                        help="Run a single query and exit")
    parser.add_argument("--rebuild-graph", action="store_true",
                        help="Rebuild _graph.json from wiki pages and exit")
    parser.add_argument("--build-wiki-index", action="store_true",
                        help="Build wiki search FAISS index and save to data/ then exit (no API key needed)")
    parser.add_argument("--model1", metavar="MODEL",
                        help=f"WIKI LLM model override (default: {WIKI_LLM_MODEL})")
    parser.add_argument("--model2", metavar="MODEL",
                        help=f"MAIN LLM model override (default: {MAIN_LLM_MODEL})")
    args = parser.parse_args()

    if args.rebuild_graph:
        graph = save_graph()
        print(f"Built graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        print(f"Saved to: {DATA_DIR / '_graph.json'}")
        return

    if args.build_wiki_index:
        pages = _load_wiki_pages()
        if not pages:
            print("[Error] No wiki pages found — check WIKI_DIR")
            sys.exit(1)
        idx = WikiSearchIndex()
        idx.build(pages)
        print(f"[Done] {len(pages)} pages → {_WIKI_FAISS_CACHE}")
        return
    if args.model1:
        WIKI_LLM_MODEL = args.model1
    if args.model2:
        MAIN_LLM_MODEL = args.model2

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[Error] ANTHROPIC_API_KEY is not set.")
        print("        Set it in .env or export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    kb = KnowledgeBase()

    print(f"\nModels — WIKI LLM (wiki agent / navigation): {WIKI_LLM_MODEL}")
    print(f"         MAIN LLM (answer agent / synthesis):  {MAIN_LLM_MODEL}")
    print(f"GitHub push: {'enabled' if os.environ.get('GITHUB_TOKEN') else 'disabled (GITHUB_TOKEN not set)'}")

    if args.query:
        result = query(args.query, kb, client)
        print(f"\n{'='*60}")
        print(result.get("answer", "[No answer returned]"))
        print(f"{'='*60}")
        if result.get("sources"):
            print(f"\nSources — wiki: {result['sources'].get('wiki', [])}")
            print(f"          rag:  {result['sources'].get('rag', [])}")
        return

    print("\nDual-LLM Knowledge Base REPL")
    print("Type your question and press Enter. Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        result = query(user_input, kb, client)
        print(f"\nAssistant:\n{result.get('answer', '[No answer returned]')}\n")


if __name__ == "__main__":
    import argparse as _argparse
    _p = _argparse.ArgumentParser(add_help=False)
    _p.add_argument("--serve", action="store_true",
                    help="Run Flask dev server instead of REPL")
    _p.add_argument("--host", default="127.0.0.1",
                    help="Bind host (use 0.0.0.0 for Cloud Run)")
    _p.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", 5001)),
                    help="Bind port (Cloud Run sets $PORT=8080)")
    _known, _rest = _p.parse_known_args()

    if _known.serve or _known.host != "127.0.0.1" or _known.port != 5001:
        try:
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        print(f"[index2] Starting Flask server on http://{_known.host}:{_known.port}")
        print("[index2] Initializing knowledge base...")
        _get_kb()  # eager init — load wiki + build FAISS before first request
        print("[index2] Knowledge base ready.")
        app.run(host=_known.host, port=_known.port, debug=False, use_reloader=False)
    else:
        sys.argv = [sys.argv[0]] + _rest
        main()
