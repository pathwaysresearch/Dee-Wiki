"""
webapp/api/index2.py — Dual-LLM agentic query pipeline (Flask deployment)

Architecture (see CLAUDE.md):
  BM25 (wiki only, full content: title + aliases + tags + body) → top 2 pages
  → WIKI_LLM / Sonnet (wiki agent): navigation, optional graph_traverse tool call
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
    load_graph,
    save_graph,
    traverse,
    parse_frontmatter,
    strip_frontmatter,
    WIKI_DIR,
    VAULT,
    GRAPH_PATH,
)

# Optional: load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Optional: BM25
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    BM25Okapi = None
    _HAS_BM25 = False
    print("[Warning] rank-bm25 not installed. BM25 disabled — install with: pip install rank-bm25")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Deployed assets live in webapp/data/ (written by export_for_web.py)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_MD_PATH = WIKI_DIR / "index.md"
LOG_MD_PATH = WIKI_DIR / "log.md"

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
    """
    Load the knowledge graph with Vercel-aware path fallback.

    Priority:
      1. DATA_DIR/_graph.json  — copied here by export_for_web.py; present on Vercel
      2. GRAPH_PATH (WIKI_DIR/_graph.json) — local dev path from graph.py
      3. Build from wiki pages   — local dev only (requires writable filesystem)
      4. Empty graph             — silent fallback; graph_traverse returns no results
    """
    # 1. Vercel-deployed copy
    vercel_path = DATA_DIR / "_graph.json"
    if vercel_path.exists():
        try:
            g = json.loads(vercel_path.read_text(encoding="utf-8"))
            print(f"[Graph] Loaded from data/_graph.json ({len(g.get('nodes', {}))} nodes)")
            return g
        except Exception as e:
            print(f"[Graph] data/_graph.json unreadable ({e}), trying wiki path")

    # 2 & 3. Local dev path (load_graph builds from wiki if missing, writes to WIKI_DIR)
    try:
        g = load_graph()
        print(f"[Graph] Loaded from wiki ({len(g.get('nodes', {}))} nodes)")
        return g
    except Exception as e:
        print(f"[Graph] Unavailable ({e}) — graph_traverse will return no results")
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
# BM25 index — wiki pages only, full content (title + aliases + tags + body)
# ---------------------------------------------------------------------------


def _build_wiki_bm25(pages: list):
    """Build BM25Okapi over wiki pages using full content: title + aliases + tags + body text."""
    if not _HAS_BM25 or not pages:
        return None
    tokenized = []
    for p in pages:
        body = strip_frontmatter(p.get("content", ""))
        full_text = " ".join([
            p.get("title", ""),
            " ".join(p.get("aliases", [])),
            " ".join(p.get("tags", [])),
            body,
        ])
        tokenized.append(full_text.lower().split())
    return BM25Okapi(tokenized)


def bm25_wiki_search(query: str, pages: list, bm25, top_k: int = 2) -> list:
    """Return top_k wiki pages scored by BM25 over full content (title + aliases + tags + body)."""
    if bm25 is None or not pages:
        return [{"page": p, "score": 0.0} for p in pages[:top_k]]
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"page": pages[i], "score": float(scores[i])} for i in top_idx]


# ---------------------------------------------------------------------------
# RAG search — embedding similarity only, no BM25 on chunks
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


# ---------------------------------------------------------------------------
# Tool: graph_traverse (used by WIKI LLM during navigation)
# ---------------------------------------------------------------------------


def tool_graph_traverse(slug: str, hops: int = 1, max_nodes: int = 5, graph: dict = None) -> list:
    """
    BFS from slug, return neighbor pages with full markdown content.
    WIKI LLM reads this content to decide which slugs to select — it does NOT
    copy this content into its output JSON.
    """
    if graph is None:
        graph = _load_graph()

    results = traverse(graph, slug, hops=hops)[:max_nodes]
    output = []

    for r in results:
        node = r["node"]
        node_data = graph["nodes"].get(node, {})
        rel_path = node_data.get("path", "")
        content = ""
        if rel_path:
            full_path = VAULT / rel_path
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""

        edge_str = " → ".join(
            f"{p['from']} --[{p['type']}]--> {p['to']}"
            for p in r.get("path", [])
        )

        output.append({
            "slug": node,
            "title": r.get("title", node),
            "type": r.get("type", "unknown"),
            "edge": edge_str,
            "content": content,
        })

    return output


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

    Thread safety: all mutation of shared state (wiki_pages, bm25, graph,
    index_md) must hold _lock. query() takes a snapshot under the lock so
    a concurrent wiki update cannot corrupt an in-flight query.
    """

    def __init__(self):
        self.wiki_pages: list = []
        self.bm25 = None
        self.chunks: list = []
        self.faiss_index = None
        self.graph: dict = {}
        self.index_md: str = ""
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        """Full reload from disk. Holds lock while swapping state."""
        print("[KB] Loading knowledge base...")
        new_pages = _load_wiki_pages()
        new_bm25 = _build_wiki_bm25(new_pages) if _HAS_BM25 else None
        new_chunks = _load_chunks()
        new_faiss = _load_faiss_index()
        new_graph = _load_graph()
        new_index = (
            INDEX_MD_PATH.read_text(encoding="utf-8")
            if INDEX_MD_PATH.exists() else ""
        )
        # Single lock acquisition for the full state swap
        with self._lock:
            self.wiki_pages = new_pages
            self.bm25 = new_bm25
            self.chunks = new_chunks
            self.faiss_index = new_faiss
            self.graph = new_graph
            self.index_md = new_index
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
    """
    Extract the first JSON object from text, handling markdown code fences.
    Returns {} on failure.
    """
    clean = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", clean)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# WIKI_LLM — Wiki Agent (two distinct system prompts: navigation vs maintenance)
# ---------------------------------------------------------------------------

_WIKI_LLM_TOOLS = [
    {
        "name": "graph_traverse",
        "description": (
            "BFS traversal from a wiki page slug to find related pages. "
            "Use when the BM25 results are insufficient and index.md reveals "
            "a more relevant page that BM25 missed. "
            "Returns neighbor pages with their full markdown content for you to read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Wiki page slug to BFS from (kebab-case filename without .md).",
                },
                "hops": {
                    "type": "integer",
                    "description": "Number of BFS hops. Default 1.",
                },
                "max_nodes": {
                    "type": "integer",
                    "description": "Max neighbor nodes to return. Default 5.",
                },
            },
            "required": ["slug"],
        },
    }
]

# Navigation prompt: WIKI LLM's job during a query
_WIKI_LLM_NAVIGATION_SYSTEM = """\
You are the wiki navigation agent. You never talk to the user directly.
You have the complete wiki catalog in <index_md> above.

---
## STEP 1 — Evaluate BM25 pages

Read the BM25 page content carefully.
**Sufficient** means: the pages directly and completely answer the user's query
from established knowledge — no raw source excerpts or chapter-level detail needed.

**If BM25 pages are sufficient → respond immediately, do NOT call graph_traverse:**
```json
{"sufficient": true, "selected_slugs": ["<bm25-slug-1>", "<bm25-slug-2>"], "note": ""}
```

---
## STEP 2 — Only if BM25 is NOT sufficient: call graph_traverse

Call `graph_traverse` on the most relevant BM25 slug(s) to surface related pages.
You may call it on multiple slugs in parallel in a single response.

After reading the returned pages, select the best ones and set:
- `"sufficient": true`  — wiki now covers the query
- `"sufficient": false` — wiki is still incomplete; MAIN_LLM should use RAG

---
## Output schema (JSON only, no other text)

```json
{
  "sufficient": true,
  "selected_slugs": ["slug-one", "slug-two"],
  "note": ""
}
```

`note` is required only when `sufficient: false` or graph_traverse was called — one sentence max.

## SLUG FORMAT — CRITICAL
A slug is the **bare filename without extension and without any directory prefix**.
- ✅ CORRECT: `"thinking-patterns"`, `"microequity"`, `"prof-bhagwan-chowdhry"`
- ❌ WRONG: `"persona/thinking-patterns"`, `"concepts/microequity"`, `"wiki/stub-foo.md"`

Index.md uses wikilink syntax like `[[persona/thinking-patterns|Title]]` — extract only the
part after the last `/` and before any `|` or `.md`. Never include path separators in slugs.\
"""

# Maintenance prompt: WIKI LLM's job when writing a new wiki page from synthesis
_WIKI_LLM_MAINTENANCE_SYSTEM = """\
You are the wiki maintenance agent.

Write a new wiki page (or update an existing one) to permanently preserve the
insight provided. The page should capture novel connections, resolved contradictions,
or reusable framings that are not yet in the wiki.

You have the complete wiki catalog in <index_md> above. Check it before deciding
whether to create a new page or update an existing slug.

Respond with raw JSON (no markdown, no explanation outside the JSON):
{
  "action": "create",
  "slug": "kebab-case-slug",
  "title": "Human Readable Title",
  "type": "synthesized",
  "tags": ["tag1", "tag2"],
  "aliases": [],
  "relationships": [
    {"target": "related-page-slug", "type": "extends"}
  ],
  "body": "Full markdown body. Write naturally — frontmatter will be added automatically."
}

If an existing page should be updated instead, set "action": "update" and use its slug.\
"""


def run_wiki_llm(
    user_query: str,
    bm25_pages: list,
    index_md: str,
    graph: dict,
    client: Anthropic,
) -> dict:
    """
    Run WIKI_LLM navigation pass. Returns:
    {
        "sufficient": bool,
        "selected_slugs": ["slug-a", "slug-b"],
        "note": str
    }

    WIKI_LLM reads full page content to make its decision but outputs only slugs.
    The backend fetches full content for those slugs to pass to MAIN_LLM.
    One API call — WIKI_LLM may issue multiple parallel graph_traverse calls in that
    single response. All results are returned in one follow-up, then WIKI_LLM produces
    its final JSON. Max 2 API calls total.
    """
    # Show WIKI_LLM the full BM25 page content so it can make a good decision
    bm25_block = ""
    for i, r in enumerate(bm25_pages, 1):
        p = r["page"]
        bm25_block += (
            f"\n--- BM25 Result {i} (score: {r['score']:.3f}) ---\n"
            f"Slug: {p['slug']} | Title: {p['title']} | Type: {p['type']}\n\n"
            f"{p['content']}\n"
        )

    # Full index.md — never truncated (200K context window)
    system = f"<index_md>\n{index_md}\n</index_md>\n\n{_WIKI_LLM_NAVIGATION_SYSTEM}"

    messages = [
        {
            "role": "user",
            "content": (
                f"User query: {user_query}\n\n"
                f"BM25 search results:{bm25_block}"
            ),
        }
    ]

    # Single API call — WIKI_LLM may return multiple parallel graph_traverse tool_use blocks
    response = client.messages.create(
        model=WIKI_LLM_MODEL,
        max_tokens=1024,   # output is just a small JSON slug list
        system=system,
        messages=messages,
        tools=_WIKI_LLM_TOOLS,
    )

    # If WIKI LLM used tools, execute all of them and do one follow-up call
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "graph_traverse":
                print(f"[WikiLLM] graph_traverse({block.input.get('slug')})")
                result = tool_graph_traverse(
                    slug=block.input.get("slug", ""),
                    hops=block.input.get("hops", 1),
                    max_nodes=block.input.get("max_nodes", 5),
                    graph=graph,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # One follow-up call with all tool results
        response = client.messages.create(
            model=WIKI_LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=_WIKI_LLM_TOOLS,
        )

    # Extract final JSON from response
    final_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )

    result = _extract_json(final_text)
    if not result or "selected_slugs" not in result:
        # Fallback: use BM25 page slugs directly
        result = {
            "sufficient": True,
            "selected_slugs": [r["page"]["slug"] for r in bm25_pages],
            "note": "WIKI LLM parse failed — using BM25 slugs directly",
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
}\
"""

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
- Draw on specific names, numbers, and places from the knowledge base
- ❌ "*laughs* That's a great question" → ✅ "That's genuinely fascinating"

## Source attribution
End your answer with this block (before [METADATA]):

My Memory: [wiki page titles you drew on, or 'Found Nothing in My Memory']
My Library: [RAG source titles from rag_search results, or 'Found Nothing in My Library']
General Knowledge: [note any inferences beyond the provided sources]

## RAG instruction
{rag_instruction}

## Formatting
Math: use (...) for inline, [...] for display. Never $...$. Use LaTeX syntax.

## Output format — CRITICAL
Write your full answer as plain conversational text (markdown is fine).
After your complete answer, output EXACTLY this on its own line, then the metadata JSON:

[METADATA]
{metadata_schema}

Rules:
- [METADATA] must be the very last thing you output — never mid-response
- sources.wiki: titles of wiki pages you actually used
- sources.rag: source titles from rag_search results (empty list if not called)
- should_wiki_update: true when you synthesise non-obvious connections, resolve a contradiction, or produce a novel framing
- new_synthesis: one sentence capturing the insight, or "" if none\
"""

_RAG_INSTRUCTION_SUFFICIENT = (
    "The wiki context is **complete** for this query. "
    "Answer from wiki only — do NOT call `rag_search`."
)
_RAG_INSTRUCTION_INSUFFICIENT = (
    "The wiki context is **incomplete**. "
    "Before calling `rag_search`, write one natural sentence telling the user you are checking your source library "
    "(e.g. 'Let me dig into my library for this.' or 'Give me a moment — I want to pull from the source on this.'). "
    "Then call `rag_search` ONCE, incorporate the results, and continue your answer."
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
    tools_arg = {} if sufficient else {"tools": _MAIN_LLM_TOOLS}

    # Sliding tail buffer: we hold back up to len(_METADATA_MARKER) chars so we
    # can detect the marker even if it straddles two chunks.
    tail_buffer = ""
    metadata_mode = False
    metadata_buf = ""
    final_msg = None

    for _ in range(2):  # at most one tool call (rag_search)
        with client.messages.stream(
            model=MAIN_LLM_MODEL,
            max_tokens=4096,
            system=system,
            messages=messages,
            **tools_arg,
        ) as stream:
            for text_chunk in stream.text_stream:
                if metadata_mode:
                    metadata_buf += text_chunk
                    continue

                tail_buffer += text_chunk

                if _METADATA_MARKER in tail_buffer:
                    before, _, after = tail_buffer.partition(_METADATA_MARKER)
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
            # Flush any remaining tail (marker never appeared)
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
                    top_k=block.input.get("top_k", 5),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(rag_results, ensure_ascii=False),
                })

        # Flush safe portion of tail before continuing with tool results
        safe_len = max(0, len(tail_buffer) - len(_METADATA_MARKER))
        if safe_len > 0:
            yield ("text", tail_buffer[:safe_len])
            tail_buffer = tail_buffer[safe_len:]

        messages.append({"role": "assistant", "content": final_msg.content})
        messages.append({"role": "user", "content": tool_results})

    # Parse and yield metadata
    try:
        metadata = json.loads(metadata_buf.strip())
    except (json.JSONDecodeError, ValueError):
        print(f"[MainLLM] Metadata parse failed. Raw: {metadata_buf[:200]!r}")
        metadata = {"sources": {"wiki": [], "rag": []}, "new_synthesis": "", "should_wiki_update": False}

    yield ("metadata", metadata)


def run_main_llm(
    user_query: str,
    wiki_context: list,
    wiki_note: str,
    sufficient: bool,
    chunks: list,
    faiss_index,
    client: Anthropic,
) -> dict:
    """
    Blocking wrapper around run_main_llm_streaming — used by the CLI REPL.
    Collects all streamed text and returns a dict with an 'answer' key.
    """
    answer_parts = []
    metadata = {}
    for event_type, data in run_main_llm_streaming(
        user_query, wiki_context, wiki_note, sufficient, chunks, faiss_index, client
    ):
        if event_type == "text":
            answer_parts.append(data)
        elif event_type == "metadata":
            metadata = data
    return {
        "answer": "".join(answer_parts),
        "sources": metadata.get("sources", {"wiki": [], "rag": []}),
        "new_synthesis": metadata.get("new_synthesis", ""),
        "should_wiki_update": metadata.get("should_wiki_update", False),
    }


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
        index_md_snapshot = kb.index_md  # full index.md, never truncated

    # Use the maintenance system prompt (different from navigation prompt)
    system = f"<index_md>\n{index_md_snapshot}\n</index_md>\n\n{_WIKI_LLM_MAINTENANCE_SYSTEM}"

    response = client.messages.create(
        model=WIKI_LLM_MODEL,
        max_tokens=2048,
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
        print("[WikiUpdate] Could not parse page data from WIKI LLM maintenance response.")
        return

    _write_wiki_page(page_data, kb, original_query)
    print(f"[WikiUpdate] Done — {page_data['slug']}")


def _write_wiki_page(page_data: dict, kb: KnowledgeBase, original_query: str = ""):
    """
    Write a wiki page to disk, patch _graph.json, rebuild BM25 (under lock),
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

    # Write file atomically (write → rename avoids partial reads)
    out_dir = WIKI_DIR / "synthesized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.md"
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"[WikiUpdate] Wrote {out_path.relative_to(PROJECT_ROOT)}")

    # Patch _graph.json on disk
    graph = load_graph()
    graph["nodes"][slug] = {
        "type": page_type,
        "title": title,
        "aliases": aliases,
        "tags": tags,
        "path": str(out_path.relative_to(VAULT)),
    }
    for r in rels:
        graph["edges"].append({
            "from": slug,
            "to": r.get("target", ""),
            "type": r.get("type", "related_to"),
        })
    GRAPH_PATH.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Update in-memory KB under lock — BM25 rebuild is also inside the lock
    # so concurrent queries see either the old or the new index, never a partial rebuild
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
        # BM25 rebuild inside the lock — no query can see a half-rebuilt index
        kb.bm25 = _build_wiki_bm25(kb.wiki_pages) if _HAS_BM25 else None

    # Append to index.md and refresh KB cache
    new_index_content = None
    if INDEX_MD_PATH.exists():
        entry = f"\n- [[synthesized/{slug}|{title}]] — synthesized from query\n"
        with open(INDEX_MD_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        new_index_content = INDEX_MD_PATH.read_text(encoding="utf-8")
        with kb._lock:
            kb.index_md = new_index_content

    # Append to log.md
    today = date.today().isoformat()
    log_entry = (
        f"\n## [{today}] synthesize | {slug}\n"
        f"- Pages created: {out_path.name}\n"
        f"- From query: {original_query}\n"
    )
    with open(LOG_MD_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Push wiki page + index.md to GitHub (push on every write)
    today_str = date.today().isoformat()
    _push_to_github(
        repo_relative_path=f"Vault/wiki/synthesized/{slug}.md",
        content=content,
        commit_msg=f"wiki: synthesize {slug} from query {today_str}",
    )
    if new_index_content:
        _push_to_github(
            repo_relative_path="Vault/wiki/index.md",
            content=new_index_content,
            commit_msg=f"wiki: update index.md after synthesizing {slug}",
        )


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------


def _pipeline_setup(user_query: str, kb: KnowledgeBase, client: Anthropic):
    """
    Shared setup for both query() and query_streaming():
    snapshot KB state, run BM25 + WIKI_LLM, return selected_pages + metadata.
    Returns (selected_pages, wiki_result, chunks, faiss_index).
    """
    print(f"\n{'─'*60}")
    print(f"[Pipeline] {user_query!r}")
    print(f"{'─'*60}")

    with kb._lock:
        wiki_pages  = list(kb.wiki_pages)
        bm25        = kb.bm25
        chunks      = list(kb.chunks)
        faiss_index = kb.faiss_index
        graph       = kb.graph
        index_md    = kb.index_md

    page_by_slug = {p["slug"]: p for p in wiki_pages}

    bm25_results = bm25_wiki_search(user_query, wiki_pages, bm25, top_k=2)
    print(f"[BM25]   top pages: {[r['page']['title'] for r in bm25_results]}")

    wiki_result = run_wiki_llm(
        user_query=user_query,
        bm25_pages=bm25_results,
        index_md=index_md,
        graph=graph,
        client=client,
    )
    print(f"[WikiLLM] selected slugs: {wiki_result.get('selected_slugs')} | sufficient={wiki_result.get('sufficient')}")

    selected_pages = []
    for raw_slug in wiki_result.get("selected_slugs", []):
        # Normalise: WIKI_LLM sometimes returns "persona/slug" or "wiki/slug.md"
        # (copied from wikilink syntax in index.md). Strip any directory prefix
        # and extension so we always look up the bare stem.
        slug = raw_slug.split("/")[-1]          # drop "persona/", "concepts/", etc.
        slug = slug.removesuffix(".md")          # drop ".md" if present
        slug = slug.split("|")[0].strip()        # drop "|Title" if wikilink leaked

        page = page_by_slug.get(slug)
        if page:
            if slug != raw_slug:
                print(f"[Pipeline] Slug normalised: '{raw_slug}' → '{slug}'")
            selected_pages.append(page)
        else:
            print(f"[Pipeline] Warning: slug '{raw_slug}' (normalised: '{slug}') not found — skipping")

    if not selected_pages:
        print("[Pipeline] No slugs resolved — falling back to BM25 pages")
        selected_pages = [r["page"] for r in bm25_results]

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
        "bm25": _HAS_BM25,
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
    parser.add_argument("--model1", metavar="MODEL",
                        help=f"WIKI LLM model override (default: {WIKI_LLM_MODEL})")
    parser.add_argument("--model2", metavar="MODEL",
                        help=f"MAIN LLM model override (default: {MAIN_LLM_MODEL})")
    args = parser.parse_args()

    if args.rebuild_graph:
        graph = save_graph()
        print(f"Built graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        print(f"Saved to: {GRAPH_PATH}")
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
    _p.add_argument("--port", type=int, default=5001)
    _known, _rest = _p.parse_known_args()

    if _known.serve:
        try:
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        print(f"[index2] Starting Flask dev server on http://localhost:{_known.port}")
        app.run(debug=True, port=_known.port, use_reloader=False)
    else:
        sys.argv = [sys.argv[0]] + _rest
        main()
