"""
webapp/api/index2.py — Dual-LLM agentic query pipeline (Flask deployment)

Architecture (see CLAUDE.md):
  WIKI_LLM / Sonnet (wiki agent): reads index.md, calls graph_traverse to fetch pages
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


# ---------------------------------------------------------------------------
# Tool: graph_traverse (used by WIKI LLM during navigation)
# ---------------------------------------------------------------------------


def tool_graph_traverse(slug: str, hops: int = 1, max_nodes: int = 8, graph: dict = None) -> list:
    """
    Return the seed page itself + BFS neighbor pages, each with full markdown content.
    WIKI LLM calls this to fetch a page it identified from index.md, then reads the
    content to decide relevance. It does NOT copy content into its output JSON.
    If the slug is not found exactly, falls back to a partial-match on graph node keys.
    """
    if graph is None:
        graph = _load_graph()

    output = []

    # Resolve slug — exact match first, then word-overlap fuzzy match as fallback
    resolved_slug = slug
    seed_data = graph["nodes"].get(slug, {})
    if not seed_data:
        slug_lower = slug.lower()
        # Significant words: split on '-' or '_', keep words longer than 3 chars
        sig_words = [w for w in re.split(r"[-_]", slug_lower) if len(w) > 3]

        candidates = []
        for k in graph["nodes"]:
            k_lower = k.lower()
            # Accept if: the slug is a long-enough direct substring of the key (or vice-versa),
            # OR at least 2 significant words from the slug appear in the key.
            if (len(slug_lower) > 5 and slug_lower in k_lower) or \
               (len(k_lower) > 5 and k_lower in slug_lower):
                candidates.append(k)
            elif sig_words:
                n_match = sum(1 for w in sig_words if w in k_lower)
                if n_match >= min(2, len(sig_words)):
                    candidates.append(k)

        if candidates:
            # Prefer shortest key (most specific match)
            resolved_slug = min(candidates, key=len)
            seed_data = graph["nodes"][resolved_slug]
            print(f"[GraphTraverse] Slug '{slug}' not found → fuzzy match '{resolved_slug}' (from {len(candidates)} candidates)")
        else:
            print(f"[GraphTraverse] Slug '{slug}' not found in graph — no fuzzy match")
            return [{"slug": slug, "title": slug, "type": "unknown", "edge": "seed",
                     "content": f"[Page '{slug}' not found in wiki]"}]

    def _read_node(node_data: dict, label: str) -> str:
        """Read a wiki page from disk, normalising path separators for cross-platform use."""
        rel_path = node_data.get("path", "").replace("\\", "/")
        if not rel_path:
            print(f"[GraphTraverse] Node '{label}' has no path in graph")
            return ""
        full_path = VAULT / rel_path
        if not full_path.exists():
            print(f"[GraphTraverse] File not found on disk: {full_path}")
            return ""
        try:
            return full_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[GraphTraverse] Read error for {full_path}: {e}")
            return ""

    # Always include the seed node's own content first
    seed_content = _read_node(seed_data, resolved_slug)
    print(f"[GraphTraverse] Seed '{resolved_slug}' — content {'found' if seed_content else 'EMPTY'}")
    output.append({
        "slug": resolved_slug,
        "title": seed_data.get("title", resolved_slug),
        "type": seed_data.get("type", "unknown"),
        "edge": "seed",
        "content": seed_content,
    })

    # BFS neighbors
    results = traverse(graph, resolved_slug, hops=hops)[:max_nodes]
    seen = {resolved_slug}
    for r in results:
        node = r["node"]
        if node in seen:
            continue
        seen.add(node)
        node_data = graph["nodes"].get(node, {})
        content = _read_node(node_data, node)

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

    titles = [p["title"] for p in output]
    print(f"[GraphTraverse] Returning {len(output)} pages: {titles}")
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

    Thread safety: all mutation of shared state (wiki_pages, graph,
    index_md) must hold _lock. query() takes a snapshot under the lock so
    a concurrent wiki update cannot corrupt an in-flight query.
    """

    def __init__(self):
        self.wiki_pages: list = []
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
        new_chunks = _load_chunks()
        new_faiss = _load_faiss_index()
        new_graph = _load_graph()
        new_index = (
            INDEX_MD_PATH.read_text(encoding="utf-8")
            if INDEX_MD_PATH.exists() else ""
        )
        with self._lock:
            self.wiki_pages = new_pages
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
            "Fetch a wiki page and its neighbors. "
            "Call this on slugs you identify from index.md to read their full content. "
            "Returns the seed page itself plus related neighbor pages with full markdown content."
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
You are the wiki navigation agent. Output JSON only. Never speak to the user.

You have the wiki catalog in <index_md> above, but **no page content yet**.
You must call `graph_traverse` to fetch content before you can judge relevance.

---

## STEP 1 — Identify candidates from index.md and fetch content

Scan index.md for entries whose title or description relates to the query.

**Extract the slug EXACTLY as it appears in index.md** — do NOT invent or guess slugs.
index.md uses wikilink syntax: `[[path/to/slug|Title]]`
The slug = the part after the last `/` and before `|` or `.md`.
Example: `[[concepts/microequity|Microequity]]` → slug is `microequity`

Call `graph_traverse` on the slugs you found (parallel calls allowed).
Each call returns that page + its neighbors with full content.

**If the query topic does not appear anywhere in index.md** → skip fetching,
output `{"sufficient": false, "selected_slugs": [], "note": "Topic not in wiki"}`.

---

## STEP 2 — Judge the fetched content (STRICT — this is where you must not cheat)

Your job is **not** to decide whether the fetched pages are "about the right topic."
Your job is to decide whether they **contain the answer** to the user's specific question.

### The sufficiency test

Before declaring `sufficient: true`, you must be able to do this:

> Point to a specific passage (a sentence or paragraph) on one of the fetched pages
> and say: "This passage directly answers the user's question because it states X."

If you cannot identify that passage — if the best you can do is "this page discusses
the general area" or "this page mentions the keyword" — then `sufficient: false`.

### Not sufficient (common failure modes — watch for these)

- **Keyword overlap only.** The page uses the same terms as the query but does not
  answer what was asked. ("The query asks *why* X happens; the page defines X.")
- **Topical adjacency.** The page is about a neighboring concept. ("Query asks about
  Random Forests; the page covers Decision Trees and mentions ensembles in passing.")
- **Tangential mention.** The topic appears in one sentence as an example, with no
  substantive treatment.
- **Wrong level of specificity.** Query asks for a concrete mechanism, number, or
  example; page gives only high-level framing — or vice versa.
- **Asserts without explaining.** Query asks *why* or *how*; page only states *that*.
- **Partial answer.** Page addresses one part of a multi-part question; the rest is
  absent.

In all of these → `sufficient: false`, even though the pages are worth passing along.

### When in doubt → false

Default to `sufficient: false`. MAIN_LLM has a RAG fallback and can recover gracefully
from a false negative. A false positive, on the other hand, makes MAIN_LLM answer from
thin wiki context and skip the library — which is the failure you are trying to avoid.

### Even when `sufficient: false`

Include the closest slugs found in `selected_slugs` so MAIN_LLM has partial wiki
context to work with alongside its RAG results.

---

## Output (JSON only — no other text)

```json
{
  "sufficient": <true|false>,
  "selected_slugs": ["slug-one", "slug-two"],
  "evidence": "<required when sufficient is true: the specific passage (≤2 sentences, quoted or closely paraphrased) from one of the fetched pages that directly answers the query, prefixed with the slug it came from — e.g. 'microequity: Microequity contracts are self-enforcing without costly state verification because...'>",
  "note": "<required when sufficient is false: one sentence on what is missing — e.g. 'Pages cover decision trees generally but do not explain bagging or variance reduction.'>"
}
```

Rules:
- If `sufficient: true`, `evidence` is mandatory and must quote or closely paraphrase
  an actual passage from a fetched page. If you cannot fill this field honestly,
  flip `sufficient` to false.
- If `sufficient: false`, `note` is mandatory.
- Include `selected_slugs` in both cases.

## SLUG FORMAT — CRITICAL

- Only use slugs you found in index.md. **Never invent a slug.**
- Bare filename only — no path prefix, no extension.
- ✅ `"thinking-patterns"`, `"microequity"`, `"prof-bhagwan-chowdhry"`
- ❌ `"persona/thinking-patterns"`, `"random-forest"` (unless that exact string appears in index.md)
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

    WIKI_LLM reads index.md to identify candidate slugs, calls graph_traverse to
    fetch their content, then judges relevance. No BM25 pre-filter.
    Up to 2 API calls total (first call fetches pages via tool; second produces JSON).
    """
    # Full index.md — never truncated (200K context window)
    system = f"<index_md>\n{index_md}\n</index_md>\n\n{_WIKI_LLM_NAVIGATION_SYSTEM}"

    messages = [{"role": "user", "content": f"User query: {user_query}"}]

    # First call — force at least one graph_traverse call so the LLM always
    # fetches page content before judging relevance (tool_choice "any").
    response = client.messages.create(
        model=WIKI_LLM_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=_WIKI_LLM_TOOLS,
        tool_choice={"type": "any"},
    )

    # Execute all graph_traverse tool calls, then do one follow-up call
    # Also collect every slug that came back so we can use them as a fallback.
    traversed_slugs: list = []
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "graph_traverse":
                print(f"[WikiLLM] graph_traverse({block.input.get('slug')})")
                pages = tool_graph_traverse(
                    slug=block.input.get("slug", ""),
                    hops=block.input.get("hops", 1),
                    max_nodes=block.input.get("max_nodes", 8),
                    graph=graph,
                )
                        # Track all returned slugs — even content-empty ones are valid
                # wiki pages that _pipeline_setup can look up from kb.wiki_pages.
                for p in pages:
                    if p["slug"] not in traversed_slugs:
                        traversed_slugs.append(p["slug"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(pages, ensure_ascii=False),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

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

    # If LLM returned empty selected_slugs but we fetched pages with content,
    # fall back to those slugs so MAIN_LLM isn't left without wiki context.
    if result and result.get("selected_slugs") == [] and traversed_slugs:
        print(f"[WikiLLM] selected_slugs was empty — using traversed slugs as fallback: {traversed_slugs}")
        result["selected_slugs"] = traversed_slugs

    if not result or "selected_slugs" not in result:
        result = {
            "sufficient": False,
            "selected_slugs": traversed_slugs,
            "note": "Wiki LLM parse failed — using traversed slugs",
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
    out_dir  = WIKI_DIR / "synthesized"
    out_path = out_dir / f"{slug}.md"
    disk_ok  = False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(out_path)
        print(f"[WikiUpdate] Wrote {out_path.relative_to(PROJECT_ROOT)}")
        disk_ok = True
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

    # Append to index.md and refresh KB cache
    new_index_content = None
    try:
        if INDEX_MD_PATH.exists():
            entry = f"\n- [[synthesized/{slug}|{title}]] — synthesized from query\n"
            with open(INDEX_MD_PATH, "a", encoding="utf-8") as f:
                f.write(entry)
            new_index_content = INDEX_MD_PATH.read_text(encoding="utf-8")
            with kb._lock:
                kb.index_md = new_index_content
    except OSError:
        pass

    # Append to log.md
    today = date.today().isoformat()
    log_entry = (
        f"\n## [{today}] synthesize | {slug}\n"
        f"- Pages created: {out_path.name}\n"
        f"- From query: {original_query}\n"
    )
    try:
        with open(LOG_MD_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except OSError:
        pass

    # Push wiki page + graph + index.md to GitHub
    today_str = date.today().isoformat()
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
    snapshot KB state, run WIKI_LLM (index.md + graph_traverse), return selected_pages + metadata.
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
        index_md    = kb.index_md

    page_by_slug = {p["slug"]: p for p in wiki_pages}

    wiki_result = run_wiki_llm(
        user_query=user_query,
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
