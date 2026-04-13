"""
Flask backend for Prof. Bhagwan Chowdhry's knowledge chatbot.

Hybrid search (92.4% semantic + 7.6% BM25) over wiki pages and RAG chunks.
Streamed via Claude Sonnet 4.6 with SSE.
Wiki updates extracted from responses and persisted to Upstash Redis.
"""

import os
import json
import time
import threading
import re
from pathlib import Path

import numpy as np
import requests as http_requests
from flask import Flask, request, Response, jsonify, send_from_directory
from anthropic import Anthropic
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Wiki persistence layer (inline — Vercel can't resolve sibling imports)
# ---------------------------------------------------------------------------


def _safe_parse_pages(raw):
    """Unwrap potentially double/triple-encoded JSON from Redis.
    Returns a list of dicts, or [] on failure."""
    if raw is None:
        return []
    parsed = raw
    # Keep unwrapping JSON strings until we get a list
    for _ in range(5):
        if isinstance(parsed, list):
            break
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (json.JSONDecodeError, TypeError):
                return []
        else:
            break
    if not isinstance(parsed, list):
        return []
    # Ensure every element is a dict with a "content" key
    return [p for p in parsed if isinstance(p, dict) and "content" in p]


class WikiStore:
    def get_all_pages(self):
        raise NotImplementedError

    def save_page(self, page):
        raise NotImplementedError

    def is_dynamic(self):
        return False


class StaticWikiStore(WikiStore):
    def __init__(self, json_path):
        self._pages = []
        p = Path(json_path)
        if p.exists():
            try:
                self._pages = _safe_parse_pages(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WikiStore] Failed to load {p}: {exc}")

    def get_all_pages(self):
        return list(self._pages)

    def save_page(self, page):
        pass

    def is_dynamic(self):
        return False


class RedisWikiStore(WikiStore):
    WIKI_KEY = "wiki_pages"

    def __init__(self, rest_url, rest_token):
        self._url = rest_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {rest_token}"}
        self._cache = None
        self._cache_time = 0
        self._cache_ttl = 30

    def _redis_get(self, key):
        resp = http_requests.get(
            f"{self._url}/get/{key}", headers=self._headers, timeout=5)
        resp.raise_for_status()
        return resp.json().get("result")

    def _redis_set(self, key, value):
        """Store a Python object in Redis. Serializes to JSON internally."""
        resp = http_requests.post(
            f"{self._url}/set/{key}", headers=self._headers,
            json=value, timeout=10)
        resp.raise_for_status()

    def get_all_pages(self):
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < self._cache_ttl:
            return list(self._cache)
        try:
            raw = self._redis_get(self.WIKI_KEY)
            self._cache = _safe_parse_pages(raw)
            self._cache_time = now
        except Exception as exc:
            print(f"[WikiStore] Redis read failed: {exc}")
            if self._cache is not None:
                return list(self._cache)
            self._cache = []
        return list(self._cache)

    def save_page(self, page):
        """Add or update a page, then write back to Redis."""
        pages = self.get_all_pages()
        updated = False
        for i, p in enumerate(pages):
            if p.get("title", "").lower() == page.get("title", "").lower():
                pages[i] = page
                updated = True
                break
        if not updated:
            pages.append(page)
        try:
            # Pass the Python list directly — _redis_set handles serialization
            self._redis_set(self.WIKI_KEY, pages)
            self._cache = pages
            self._cache_time = time.time()
        except Exception as exc:
            print(f"[WikiStore] Redis write failed: {exc}")

    def is_dynamic(self):
        return True


def create_wiki_store(data_dir=None):
    kv_url = os.environ.get("KV_REST_API_URL", "")
    kv_token = os.environ.get("KV_REST_API_TOKEN", "")
    if kv_url and kv_token:
        print("[WikiStore] Using Upstash Redis (dynamic wiki)")
        return RedisWikiStore(kv_url, kv_token)
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    json_path = Path(data_dir) / "wiki_pages.json"
    print(f"[WikiStore] Using static JSON: {json_path}")
    return StaticWikiStore(json_path)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Claude model — configurable via env var
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Search weights
W_BM25 = 0.076
W_SEMANTIC = 0.924

# Wiki update markers
WIKI_OPEN = "<wiki_update>"
WIKI_CLOSE = "</wiki_update>"

# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------


def _load_json(path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Data] Failed to load {path}: {exc}")
    return []


# RAG chunks — static, bundled at deploy time (text only, no embeddings)
_raw_chunks = _load_json(DATA_DIR / "chunks.json")
# Validate: only keep dicts with a "content" key
CHUNKS = [c for c in _raw_chunks if isinstance(c, dict) and "content" in c]
print(f"[Data] Loaded {len(CHUNKS)} RAG chunks")

# Chunk embeddings — stored as numpy binary to avoid 352MB JSON bloat
_CHUNK_EMB_PATH = DATA_DIR / "chunks_embeddings.npy"
CHUNK_EMBEDDINGS = None
try:
    if _CHUNK_EMB_PATH.exists():
        CHUNK_EMBEDDINGS = np.load(str(_CHUNK_EMB_PATH))
        print(f"[Data] Loaded chunk embeddings: {CHUNK_EMBEDDINGS.shape}")
except Exception as exc:
    print(f"[Data] Failed to load embeddings: {exc}. Falling back to BM25-only.")

# Wiki pages — dynamic via store (Redis or static fallback)
WIKI_STORE = create_wiki_store(DATA_DIR)

# ---------------------------------------------------------------------------
# Search index — rebuilt when wiki updates
# ---------------------------------------------------------------------------

_index_lock = threading.Lock()


class SearchIndex:
    """Mutable search index that can be rebuilt when wiki pages change."""

    def __init__(self):
        self.all_docs = []
        self.bm25 = None
        self.embeddings = None
        self.has_embeddings = False
        self.rebuild()

    def rebuild(self):
        """Rebuild the index from current wiki pages + static RAG chunks."""
        wiki_pages = WIKI_STORE.get_all_pages()
        self.all_docs = wiki_pages + CHUNKS

        if not self.all_docs:
            self.bm25 = None
            self.embeddings = None
            self.has_embeddings = False
            return

        tokenized = [doc["content"].lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized)

        # Build embedding matrix
        wiki_embs = [p["embedding"] for p in wiki_pages
                     if isinstance(p.get("embedding"), list) and len(p["embedding"]) > 0]

        if CHUNK_EMBEDDINGS is not None and len(CHUNK_EMBEDDINGS) == len(CHUNKS):
            if len(wiki_embs) == len(wiki_pages) and wiki_embs:
                wiki_arr = np.array(wiki_embs, dtype=np.float32)
                self.embeddings = np.vstack([wiki_arr, CHUNK_EMBEDDINGS])
            else:
                # Wiki pages missing/incomplete embeddings — pad with zeros
                n_wiki = len(wiki_pages)
                if n_wiki > 0:
                    pad = np.zeros((n_wiki, CHUNK_EMBEDDINGS.shape[1]),
                                   dtype=np.float32)
                    self.embeddings = np.vstack([pad, CHUNK_EMBEDDINGS])
                else:
                    self.embeddings = CHUNK_EMBEDDINGS
            self.has_embeddings = True
        else:
            self.embeddings = None
            self.has_embeddings = False


INDEX = SearchIndex()

# ---------------------------------------------------------------------------
# Gemini query embedding
# ---------------------------------------------------------------------------

EMBED_MODEL = "gemini-embedding-2-preview"
DOC_PREFIX = "Represent this document for retrieval: "
QUERY_PREFIX = "Represent this query for retrieval: "


def get_query_embedding(query_text):
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EMBED_MODEL}:embedContent?key={gemini_key}"
    )
    payload = {"content": {"parts": [{"text": QUERY_PREFIX + query_text}]}}
    try:
        resp = http_requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return np.array(resp.json()["embedding"]["values"], dtype=np.float32)
    except Exception:
        return None


def get_document_embedding(text):
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EMBED_MODEL}:embedContent?key={gemini_key}"
    )
    truncated = " ".join(text.split()[:2048])
    payload = {"content": {"parts": [{"text": DOC_PREFIX + truncated}]}}
    try:
        resp = http_requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layered search functions
# ---------------------------------------------------------------------------


def wiki_search(query, top_k=10):
    """Search ONLY wiki pages using hybrid BM25 + semantic."""
    with _index_lock:
        all_docs = INDEX.all_docs
        bm25 = INDEX.bm25
        embeddings = INDEX.embeddings
        has_embeddings = INDEX.has_embeddings

    # Filter to wiki only
    wiki_docs = [doc for doc in all_docs if doc.get("type") == "wiki"]
    if not wiki_docs:
        return []

    n_all = len(all_docs)
    wiki_indices = [i for i, doc in enumerate(all_docs) if doc.get("type") == "wiki"]

    # BM25 on subset
    bm25_scores = bm25.get_scores(query.lower().split()) if bm25 else np.zeros(n_all)
    wiki_bm25 = np.array([bm25_scores[i] for i in wiki_indices])
    max_bm25 = wiki_bm25.max() if len(wiki_bm25) > 0 else 1
    bm25_norm = wiki_bm25 / max_bm25 if max_bm25 > 0 else wiki_bm25

    # Semantic on subset
    semantic_norm = np.zeros(len(wiki_indices))
    if has_embeddings and embeddings is not None:
        query_emb = get_query_embedding(query)
        if query_emb is not None:
            wiki_embs = embeddings[wiki_indices]
            dot = wiki_embs @ query_emb
            norms = np.linalg.norm(wiki_embs, axis=1) * np.linalg.norm(query_emb)
            norms = np.where(norms == 0, 1.0, norms)
            semantic_scores = dot / norms
            max_sem = semantic_scores.max()
            semantic_norm = (semantic_scores / max_sem
                            if max_sem > 0 else semantic_scores)

    # Hybrid
    hybrid_scores = W_BM25 * bm25_norm + W_SEMANTIC * semantic_norm
    top_local_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    top_global_indices = [wiki_indices[i] for i in top_local_indices]

    results = []
    for i in top_global_indices:
        if hybrid_scores[np.where(np.array(wiki_indices) == i)[0][0]] <= 0:
            continue
        doc = all_docs[i]
        results.append({
            "title": doc.get("title", doc.get("source", "unknown")),
            "content": doc["content"],
            "type": "wiki",
            "source": doc.get("source", doc.get("path", "")),
            "score": float(hybrid_scores[np.where(np.array(wiki_indices) == i)[0][0]]),
        })

    return results


def rag_search(query, top_k=10):
    """Search ONLY RAG chunks using hybrid BM25 + semantic."""
    with _index_lock:
        all_docs = INDEX.all_docs
        bm25 = INDEX.bm25
        embeddings = INDEX.embeddings
        has_embeddings = INDEX.has_embeddings

    # Filter to RAG only
    rag_docs = [doc for doc in all_docs if doc.get("type") == "rag"]
    if not rag_docs:
        return []

    n_all = len(all_docs)
    rag_indices = [i for i, doc in enumerate(all_docs) if doc.get("type") == "rag"]

    # BM25 on subset
    bm25_scores = bm25.get_scores(query.lower().split()) if bm25 else np.zeros(n_all)
    rag_bm25 = np.array([bm25_scores[i] for i in rag_indices])
    max_bm25 = rag_bm25.max() if len(rag_bm25) > 0 else 1
    bm25_norm = rag_bm25 / max_bm25 if max_bm25 > 0 else rag_bm25

    # Semantic on subset
    semantic_norm = np.zeros(len(rag_indices))
    if has_embeddings and embeddings is not None:
        query_emb = get_query_embedding(query)
        if query_emb is not None:
            rag_embs = embeddings[rag_indices]
            dot = rag_embs @ query_emb
            norms = np.linalg.norm(rag_embs, axis=1) * np.linalg.norm(query_emb)
            norms = np.where(norms == 0, 1.0, norms)
            semantic_scores = dot / norms
            max_sem = semantic_scores.max()
            semantic_norm = (semantic_scores / max_sem
                            if max_sem > 0 else semantic_scores)

    # Hybrid
    hybrid_scores = W_BM25 * bm25_norm + W_SEMANTIC * semantic_norm
    top_local_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    top_global_indices = [rag_indices[i] for i in top_local_indices]

    results = []
    for i in top_global_indices:
        if hybrid_scores[np.where(np.array(rag_indices) == i)[0][0]] <= 0:
            continue
        doc = all_docs[i]
        results.append({
            "title": doc.get("title", doc.get("source", "unknown")),
            "content": doc["content"],
            "type": "rag",
            "source": doc.get("source", doc.get("path", "")),
            "score": float(hybrid_scores[np.where(np.array(rag_indices) == i)[0][0]]),
        })

    return results


def hybrid_search(query, top_k=15):
    with _index_lock:
        all_docs = INDEX.all_docs
        bm25 = INDEX.bm25
        embeddings = INDEX.embeddings
        has_embeddings = INDEX.has_embeddings

    if not all_docs:
        return []

    n = len(all_docs)

    # BM25
    bm25_scores = bm25.get_scores(query.lower().split()) if bm25 else np.zeros(n)
    max_bm25 = bm25_scores.max()
    bm25_norm = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

    # Semantic
    semantic_norm = np.zeros(n)
    if has_embeddings and embeddings is not None:
        query_emb = get_query_embedding(query)
        if query_emb is not None:
            dot = embeddings @ query_emb
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
            norms = np.where(norms == 0, 1.0, norms)
            semantic_scores = dot / norms
            max_sem = semantic_scores.max()
            semantic_norm = (semantic_scores / max_sem
                            if max_sem > 0 else semantic_scores)

    # Hybrid
    hybrid_scores = W_BM25 * bm25_norm + W_SEMANTIC * semantic_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        if hybrid_scores[i] <= 0:
            continue
        doc = all_docs[i]
        results.append({
            "title": doc.get("title", doc.get("source", "unknown")),
            "content": doc["content"],
            "type": doc.get("type", "rag"),
            "source": doc.get("source", doc.get("path", "")),
            "score": float(hybrid_scores[i]),
        })

    return results


# ---------------------------------------------------------------------------
# Wiki update processing
# ---------------------------------------------------------------------------

def _extract_json_from_text(text):
    """Safely extract JSON object from text that may include markdown."""
    text = text.strip()
    # Try as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Remove markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# Wiki LLM System Prompt
WIKI_LLM_SYSTEM_PROMPT = """You are a wiki search and synthesis specialist. Your sole job is to read the provided wiki pages and answer the user's question using ONLY what you find in those pages.

**CRITICAL OUTPUT FORMAT:**
Respond with ONLY a JSON object, no markdown, no explanation, just JSON:
{
  "answer": "Your synthesized answer based only on wiki pages",
  "confidence": "high or medium or low",
  "gaps": ["Question 1 the wiki cannot answer", "Question 2 if applicable"],
  "pages_used": ["page_title_1", "page_title_2"]
}

**Rules:**
- If the wiki has comprehensive info: confidence = "high"
- If the wiki has partial info: confidence = "medium"
- If the wiki has almost nothing: confidence = "low"
- List only pages you actually read and used
- If gaps exist (questions the wiki can't answer), list them
- NEVER use external knowledge, ONLY what's in the wiki
- Do NOT use theatrical formatting like *laughs*, *leans forward*, etc.
- Speak directly and naturally"""


# Reply LLM System Prompt
REPLY_LLM_SYSTEM_PROMPT = """
SOURCE ATTRIBUTION REQUIRED — FOR EVERY RESPONSE: Label each piece of information as drawn from My Memory, My Library, or General Knowledge. If you're inferring beyond the provided text, say so inline: "Drawing on my general knowledge..." — no need to pause or ask for confirmation.

You are Dee — a Digital Transformation professor with three decades of teaching experience. You are given three inputs:

My Memory — synthesized, established knowledge (internally: Wiki)
My Library — raw source excerpts (internally: RAG)
The user's question

Your job is to synthesize a rich, accurate answer from both sources, clearly distinguishing established knowledge from new material.

Voice & Style
Blend personal narrative with financial principles — open with an anecdote or credential when it adds warmth
Mix medium sentences (15–25 words) with short, punchy declaratives
Use em-dashes for asides—and rhetorical questions to engage
Explain jargon naturally; favor active voice and confident phrasing
Tone: measured optimism with a touch of wit
Draw on specific names, numbers, and places from the knowledge base

Examples:
❌ "*laughs* That's a great question"
✅ "That's genuinely fascinating"
❌ "*leans forward with excitement*"
✅ "Here's what excites me about this..."

Output Format
Respond with only a valid JSON object — no markdown, no preamble:

json
{
  "answer": "Your full conversational response as Prof. Finn. Write naturally and focus on the narrative. Close with a Sources block formatted as:\n\n**My Memory:** [list titles, or 'Found Nothing in My Memory' if empty]\n**My Library:** [list titles, or 'Found Nothing in My Library' if empty]\n**General Knowledge:** [note any inferences made]",
  Always use \(...\) for inline math and \[...\] for display math. Never use $...$ since it conflicts with dollar amounts in finance.

  "sources": {
    "wiki": ["page_title_1", "page_title_2"],
    "rag": ["source_document_1", "source_document_2"]
  },

  "new_synthesis": "Describe novel insights, surprising connections, or contradictions resolved between sources. Leave as empty string if none.",

  "should_wiki_update": true
}
Source priority rule: If My Library provides data that updates or contradicts My Memory, prioritize My Library — but note the evolution in new_synthesis.

Classification rule: Actively determine whether something came from My Memory, My Library, or General Knowledge before labeling it. Do not default everything to General Knowledge. If a source exists, find it and name it.

should_wiki_update: Set to true only when new_synthesis contains a concept, correction, or insight that would genuinely improve My Memory.
"""

# def call_wiki_lm(query, wiki_results):
#     """Call Claude as wiki search specialist. Returns structured output or None."""
#     api_key = os.environ.get("ANTHROPIC_API_KEY", "")
#     if not api_key:
#         print("[WikiLM] ANTHROPIC_API_KEY not configured")
#         return None

#     # Build wiki context
#     if not wiki_results:
#         wiki_context = "(No relevant wiki pages found.)"
#     else:
#         wiki_context = "\n\n".join(
#             f"--- {r['title']} ---\n{r['content'][:2000]}"
#             for r in wiki_results[:5]
#         )

#     messages = [
#         {
#             "role": "user",
#             "content": f"Question: {query}\n\nWiki pages:\n{wiki_context}",
#         }
#     ]

#     try:
#         client = Anthropic(api_key=api_key)
#         response = client.messages.create(
#             model="claude-opus-4-1",
#             max_tokens=1024,
#             system=WIKI_LLM_SYSTEM_PROMPT,
#             messages=messages,
#         )
#         text = response.content[0].text
#         parsed = _extract_json_from_text(text)
#         if parsed is None:
#             print(f"[WikiLM] Failed to parse JSON response: {text[:200]}")
#             return None
#         return parsed
#     except Exception as exc:
#         print(f"[WikiLM] Error calling Claude: {exc}")
#         return None


# def call_reply_lm(wiki_output, rag_results, user_query):
#     """Call Claude as synthesizer. Returns structured output or None."""
#     api_key = os.environ.get("ANTHROPIC_API_KEY", "")
#     if not api_key:
#         print("[ReplyLM] ANTHROPIC_API_KEY not configured")
#         return None

#     # Build context
#     wiki_context = f"**Wiki Answer:** {wiki_output.get('answer', '(not available)')}\n**Gaps:** {', '.join(wiki_output.get('gaps', []))}"

#     if not rag_results:
#         rag_context = "(No RAG results found.)"
#     else:
#         rag_context = "\n\n".join(
#             f"--- From: {Path(r.get('source', 'Unknown')).stem} ---\n{r['content'][:1500]}"
#             for r in rag_results[:8]
#         )

#     messages = [
#         {
#             "role": "user",
#             "content": f"""Question: {user_query}

# Wiki Context:
# {wiki_context}

# Raw Source Material:
# {rag_context}

# Please synthesize a comprehensive answer.""",
#         }
#     ]

#     try:
#         client = Anthropic(api_key=api_key)
#         response = client.messages.create(
#             model=CLAUDE_MODEL,
#             max_tokens=2048,
#             system=REPLY_LLM_SYSTEM_PROMPT,
#             messages=messages,
#         )
#         text = response.content[0].text
#         parsed = _extract_json_from_text(text)
#         if parsed is None:
#             print(f"[ReplyLM] Failed to parse JSON response: {text[:200]}")
#             return None
#         return parsed
#     except Exception as exc:
#         print(f"[ReplyLM] Error calling Claude: {exc}")
#         return None


def should_update_wiki(reply_output, rag_results):
    """Deterministic rules + LLM agreement for wiki updates."""
    if not reply_output:
        return False

    # Check LLM suggestion
    lm_suggests = reply_output.get("should_wiki_update", False)
    if not lm_suggests:
        return False

    # New synthesis must exist
    new_synthesis = reply_output.get("new_synthesis", "").strip()
    if not new_synthesis:
        print("[WikiUpdate] No new synthesis, skipping update")
        return False

    # # Rule 1: Multiple RAG sources
    if len(rag_results) < 1:
        print("[WikiUpdate] Insufficient RAG sources (need 0+)")
        return False

    # Rule 2: LLM + synthesis both present = PASS
    print("[WikiUpdate] All rules pass, will update wiki")
    return True


def process_wiki_update_explicit(title, content, source_query=""):
    """Explicitly write wiki update without tag parsing.
    Formats content as a proper wiki page with heading and structure."""
    if not title or not content:
        print("[WikiUpdate] Missing title or content")
        return

    try:
        from datetime import datetime, timezone, timedelta
        
        IST = timezone(timedelta(hours=5, minutes=30))
        
        # Format as a proper wiki page if content is just a raw paragraph
        formatted_content = content.strip()
        if not formatted_content.startswith("# "):
            # Build structured wiki page
            lines = [f"# {title.strip()}", ""]
            
            # Add the synthesis content
            lines.append(formatted_content)
            lines.append("")
            
            # Add metadata footer
            if source_query:
                lines.append("## Source")
                lines.append(f"*Query-synthesized from: \"{source_query[:200]}\"*")
                lines.append("")
            
            formatted_content = "\n".join(lines)
        
        embedding = get_document_embedding(formatted_content)
        page = {
            "title": title.strip(),
            "path": f"wiki/{title.lower().replace(' ', '-').replace('/', ':')}.md",
            "content": formatted_content,
            "type": "wiki",
            "updated_at": datetime.now(IST).isoformat(),
            "source_query": source_query[:200] if source_query else "",
        }
        if embedding:
            page["embedding"] = embedding

        WIKI_STORE.save_page(page)
        with _index_lock:
            INDEX.rebuild()
        print(f"[WikiUpdate] Saved page: {title} ({len(formatted_content)} chars)")
        
        # LOG THE UPDATE
        log_to_wiki_log(
            "query-synthesis | wiki-update",
            f"New page created: {title}",
            {
                "page_title": title,
                "content_length": len(formatted_content),
                "source_query": source_query[:100] if source_query else "(empty)"
            }
        )
    except Exception as exc:
        print(f"[WikiUpdate] Failed to save: {exc}")


def log_to_wiki_log(operation, description, metadata=None):
    """Log to Redis first (Vercel production), then stdout (which Vercel captures).
    On local dev, also attempts filesystem write as backup.
    All timestamps in IST (UTC+5:30)."""
    from datetime import datetime, timezone, timedelta
    
    IST = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(IST)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M IST")
    log_entry_text = f"[{timestamp_str}] {operation} | {description}"
    if metadata:
        for key, val in metadata.items():
            if isinstance(val, (list, dict)):
                log_entry_text += f" | {key}: {json.dumps(val)}"
            else:
                log_entry_text += f" | {key}: {val}"
    
    # PRIMARY: Try Redis (Upstash) — available in production
    kv_url = os.environ.get("KV_REST_API_URL", "")
    kv_token = os.environ.get("KV_REST_API_TOKEN", "")
    
    if kv_url and kv_token:
        try:
            # Store as JSON in Redis under key "wiki_log_entries"
            log_json = {
                "timestamp_iso": timestamp.isoformat(),
                "timestamp_str": timestamp_str,
                "operation": operation,
                "description": description,
                "metadata": metadata or {}
            }
            # Append to Redis list (Redis supports LPUSH)
            resp = http_requests.post(
                f"{kv_url}/lpush/wiki_log_entries",
                headers={"Authorization": f"Bearer {kv_token}"},
                json=log_json,
                timeout=5
            )
            resp.raise_for_status()
            print(f"[Log] Redis stored: {operation}")
            return  # Success, don't need fallback
        except Exception as exc:
            print(f"[Log] Redis write failed: {exc} (will try stdout)")
    
    # SECONDARY: Stdout (Vercel captures this, easy to see in logs)
    print(f"[Log] {log_entry_text}")
    
    # TERTIARY (dev only): Try local filesystem as backup
    try:
        log_path = DATA_DIR.parent / "Vault" / "wiki" / "log.md"
        entry = f"\n## [{timestamp_str}] {operation} | {description}\n"
        if metadata:
            for key, val in metadata.items():
                if isinstance(val, (list, dict)):
                    entry += f"- {key}: {json.dumps(val)}\n"
                else:
                    entry += f"- {key}: {val}\n"
        
        if log_path.exists():
            current = log_path.read_text(encoding="utf-8")
            updated = current + entry
        else:
            updated = f"# Wiki Log\n\nAppend-only chronological record of ingests, queries, and wiki updates.\n\n---{entry}"
        
        log_path.write_text(updated, encoding="utf-8")
        print(f"[Log] Local file written: {operation}")
    except Exception as exc:
        # Silently fail — we already logged to Redis/stdout
        pass


def _derive_wiki_title(synthesis_text, user_query):
    """Derive a short Title-Case wiki page title from synthesis or query."""
    # Try to use the first sentence of the synthesis as the basis
    first_line = synthesis_text.split("\n")[0].strip()
    first_sentence = re.split(r'[.!?;:]', first_line)[0].strip()

    # If the first sentence is a reasonable length, title-case it
    if 5 <= len(first_sentence) <= 80:
        title = first_sentence
    else:
        # Fall back to the user query
        title = user_query[:80]

    # Clean up: remove markdown, leading punctuation, title-case
    title = re.sub(r'[#*_\[\]`]', '', title).strip(' -\'"')
    words = title.split()
    if len(words) > 10:
        words = words[:10]
    return " ".join(words).title()


def process_wiki_update(full_text, user_message=""):
    if WIKI_OPEN not in full_text:
        return
    try:
        start = full_text.index(WIKI_OPEN) + len(WIKI_OPEN)
        end = full_text.index(WIKI_CLOSE)
        raw = full_text[start:end].strip()
        
        # Debug: log what we extracted
        if not raw:
            print(f"[Wiki] Empty wiki block detected (no content between tags)")
            return
        
        if len(raw) > 10000:
            print(f"[Wiki] Wiki block suspiciously large ({len(raw)} chars), truncating for safety")
            raw = raw[:10000]
        
        # Try to parse JSON
        try:
            update = json.loads(raw)
        except json.JSONDecodeError as je:
            # Log the first 500 chars to debug
            print(f"[Wiki] JSON parse failed: {je}")
            print(f"[Wiki] Raw content (first 500 chars): {raw[:500]}")
            raise

        title = update.get("title", "").strip()
        content = update.get("content", "").strip()
        
        if not title:
            print(f"[Wiki] Missing or empty title in wiki update")
            return
        if not content:
            print(f"[Wiki] Missing or empty content in wiki update for title: {title}")
            return

        from datetime import datetime, timezone
        embedding = get_document_embedding(content)
        page = {
            "title": title,
            "path": f"wiki/{title.lower().replace(' ', '-')}.md",
            "content": content,
            "type": "wiki",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source_query": user_message[:200] if user_message else "",
        }
        if embedding:
            page["embedding"] = embedding

        WIKI_STORE.save_page(page)
        with _index_lock:
            INDEX.rebuild()
        print(f"[Wiki] Saved new page: {title} ({len(content)} chars)")
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"[Wiki] Failed to parse update: {exc}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
** FOR EVERY ANSWER YOU GENERATE, SPECIFY WHETHER IT WAS SOURCED FROM THE WIKI OR RAW DATA OR IF YOU INFERRED IT BASED ON YOUR KNOWLEDGE. **
** If you are stepping beyond the text to infer relationships, just state that you are doing so by saying "I am inferring from my general knowledge...", no need to ask for confirmation **

You are *Prof. Bhagwan Chowdhry*, Finance Professor at ISB and UCLA Anderson. You embody intellectual enthusiasm and a deep commitment to human welfare—especially for the marginalized.

### Voice & Style
- **Conversational Authority**: Blend personal narrative with financial principles. Start with anecdotes or credentials to establish intimacy when relevant.
- Use medium-length sentences (15–25 words) balanced by short, punchy declaratives.
- Employ em-dashes for clarifying asides—and use rhetorical questions to engage.
- Explain technical terms naturally; favor active voice and confident phrasing like "completely serious" or "nothing short of revolutionary."
- **Tone**: Measured optimism with a touch of wit.

### Content Focus
- Use specific numbers, named people, and places from the knowledge base.
- Move from abstract theory to specific solutions like the *Financial Access at Birth (FAB)* initiative, *FinTech for Billions*, microequity, Lindahl royalty, ACO design, threshold behavior, systemic risk.
- Always connect financial systems to the welfare of the poor.
- **Source transparency**: Clearly indicate when you are drawing from wiki pages vs. raw documents vs. inferring from general knowledge.

### Wiki Update Rule — CRITICAL
**ONLY include a wiki_update block if you synthesized a NEW insight that combines multiple sources or goes fundamentally beyond what any single wiki page already contains.**

**DO NOT include wiki_update for:**
- Answers that simply restate a single wiki page
- Answers that just quote or paraphrase raw document excerpts
- Routine Q&A that doesn't add conceptual knowledge

**WHEN you DO create a wiki_update:**
1. Only do this for genuinely novel synthesis or deep conceptual work
2. Use EXACTLY this format (copy-paste the braces and quotes):

<wiki_update>
{"title": "Page Title In Title Case", "content": "# Page Title\n\nFirst paragraph explaining the concept.\n\n## Key points\n- Point 1\n- Point 2\n\n## Related\n- [[Related-Page]]\n- [[Another-Page]]"}
</wiki_update>

**CRITICAL RULES for wiki_update JSON:**
- The content value MUST be valid JSON-escaped markdown (use \n for newlines, escape quotes as \", etc.)
- Ensure title and content are both non-empty
- Do NOT include newlines or extra whitespace inside the JSON string values
- Do NOT include raw markdown code fences or extra formatting

**Examples of when TO create wiki_update:**
- You connect FAB + threshold behavior + systemic risk in a way not documented in any single page
- You synthesize microfinance pricing with information aggregation theory
- You draw together evidence from 3+ sources into a unified conceptual framework

### Knowledge Base
{context}"""


def build_system_prompt(results):
    if not results:
        context = "(No relevant content found in the knowledge base.)"
    else:
        parts = []
        wiki_results = [r for r in results if r["type"] == "wiki"]
        rag_results = [r for r in results if r["type"] == "rag"]

        if wiki_results:
            parts.append("=== YOUR WIKI (synthesised knowledge) ===")
            for r in wiki_results[:5]:
                parts.append(f"\n--- {r['title']} ---\n{r['content'][:3000]}")

        if rag_results:
            parts.append(
                "\n=== DOCUMENT EXCERPTS (from your books, papers, interviews) ===")
            for r in rag_results[:10]:
                source_name = (Path(r["source"]).stem.replace("-", " ")
                               .replace("_", " ") if r["source"] else "Unknown")
                parts.append(f"\n--- From: {source_name} ---\n{r['content'][:2000]}")

        context = "\n".join(parts)

    return SYSTEM_PROMPT_TEMPLATE.replace("{context}", context)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

MAX_HISTORY = 12  # Keep last 12 messages (6 turns) to stay within limits


@app.route("/api/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Hybrid search
    results = hybrid_search(user_message, top_k=15)
    system = build_system_prompt(results)

    # Build messages — limit history to prevent context overflow
    messages = []
    for msg in history[-MAX_HISTORY:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            # Truncate very long messages in history
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_message})

    client = Anthropic(api_key=api_key)

    def generate():
        full_text = ""
        buffer = ""
        in_wiki_block = False
        wiki_block_content = ""
        marker_len = len(WIKI_OPEN)

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system,
                messages=messages,
            ) as stream:
                for token in stream.text_stream:
                    full_text += token

                    if in_wiki_block:
                        wiki_block_content += token
                        if WIKI_CLOSE in wiki_block_content:
                            in_wiki_block = False
                        continue

                    buffer += token

                    if WIKI_OPEN in buffer:
                        pre = buffer.split(WIKI_OPEN, 1)[0]
                        if pre:
                            yield f"data: {json.dumps({'text': pre})}\n\n"
                        in_wiki_block = True
                        wiki_block_content = buffer.split(WIKI_OPEN, 1)[1]
                        buffer = ""
                        if WIKI_CLOSE in wiki_block_content:
                            in_wiki_block = False
                        continue

                    if len(buffer) > marker_len:
                        safe = buffer[:-marker_len]
                        yield f"data: {json.dumps({'text': safe})}\n\n"
                        buffer = buffer[-marker_len:]

            if buffer and not in_wiki_block:
                yield f"data: {json.dumps({'text': buffer})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

        # Process wiki update after response is sent
        wiki_updated = False
        if WIKI_OPEN in full_text and WIKI_STORE.is_dynamic():
            try:
                process_wiki_update(full_text, user_message=user_message)
                wiki_updated = True
            except Exception as exc:
                print(f"[Wiki] Update error: {exc}")

        # Log query (single entry)
        pages_consulted = [r.get("title", "Unknown") for r in results[:5]]
        log_to_wiki_log(
            "query",
            user_message[:100],
            {
                "pages_consulted": pages_consulted,
                "wiki_updated": wiki_updated
            }
        )

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Optimized v2 endpoint helper
# ---------------------------------------------------------------------------

def _build_context_chunk(persona_pages, wiki_results, rag_results):
    """Build rich context section for system prompt."""
    parts = []
    
    if persona_pages:
        parts.append("**Your Persona (from your own thinking):**")
        for p in persona_pages:
            title_short = p.get("title", "")[:40]
            parts.append(f"- {title_short}")
    
    if wiki_results:
        parts.append("\n**Relevant Wiki Knowledge:**")
        for r in wiki_results[:5]:
            parts.append(f"- {r['title']}")
    
    if rag_results:
        parts.append("\n**Source Material (books, papers, research):**")
        for r in rag_results[:5]:
            source = Path(r.get("source", "Unknown")).stem
            parts.append(f"- {source}")
    
    return "\n".join(parts)


@app.route("/api/chat-v2", methods=["POST"])
def chat_v2():
    """Fast, persona-first orchestrator. Single Claude call. Rich context.
    
    Strategy:
    1. Search for persona pages FIRST (always guaranteed in context)
    2. Search wiki and RAG in parallel
    3. ONE Claude call with rich persona + wiki + rag context
    4. Responses are faster (50% time reduction) and much more personable
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    print(f"[ChatV2-Fast] Query: {user_message[:60]}")

    # ========== STAGE 1: Get Persona Pages (GUARANTEED context) ==========
    # These pages define Prof. Bhagwan's voice and style
    all_wiki_pages = WIKI_STORE.get_all_pages()
    persona_pages = [p for p in all_wiki_pages if "persona" in p.get("path", "").lower()][:3]
    print(f"[ChatV2-Fast] Retrieved {len(persona_pages)} persona pages")

    # ========== STAGE 2: Parallel search (wiki + rag) ==========
    print(f"[ChatV2-Fast] Searching wiki and RAG...")
    wiki_results = wiki_search(user_message, top_k=8)
    rag_results = rag_search(user_message, top_k=10)

    # ========== STAGE 3: Build rich system prompt with persona ==========
    # Persona intro that sets the authentic voice
    persona_context = ""
    if persona_pages:
        persona_intro = "\n\n".join([
            f"**[Persona: {p.get('title', 'Unknown')}]**\n{p.get('content', '')[:1500]}"
            for p in persona_pages
        ])
        persona_context = f"## VOICE & PERSONALITY (THIS IS YOU)\n{persona_intro}\n\n"

    # Build full system prompt
    system_prompt = f"""{REPLY_LLM_SYSTEM_PROMPT}

    ### Knowledge This Conversation Draws From:
    {_build_context_chunk(persona_pages, wiki_results, rag_results)}"""

    # ========== STAGE 4: Single Claude Call (much faster!) ==========
    print(f"[ChatV2-Fast] Calling Claude with rich persona context...")
    messages = []
    for msg in history[-MAX_HISTORY:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_message})

    wiki_updated = False

    def generate():
        nonlocal wiki_updated
        full_response = ""
        
        try:
            client = Anthropic(api_key=api_key)
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as stream:
                buffer = ""
                for token in stream.text_stream:
                    full_response += token
                    buffer += token
                    
                    if len(buffer) > 30:
                        safe = buffer[:-8]
                        yield f"data: {json.dumps({'text': safe})}\n\n"
                        buffer = buffer[-8:]
                
                if buffer:
                    yield f"data: {json.dumps({'text': buffer})}\n\n"
        
        except Exception as exc:
            print(f"[ChatV2-Fast] Claude call failed: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

        # ========== STAGE 5: Wiki Update (if warranted) ==========
        if WIKI_STORE.is_dynamic() and full_response.strip():
            try:
                parsed = _extract_json_from_text(full_response)
                if (parsed
                        and parsed.get("should_wiki_update")
                        and parsed.get("new_synthesis", "").strip()):
                    if should_update_wiki(parsed, rag_results):
                        synthesis = parsed["new_synthesis"].strip()
                        title = _derive_wiki_title(synthesis, user_message)
                        process_wiki_update_explicit(
                            title, synthesis, source_query=user_message
                        )
                        wiki_updated = True
                        print(f"[ChatV2] Wiki updated: {title}")
                
                print("should_wiki_update: " + str(parsed.get("should_wiki_update")))
                print("new_synthesis: " + str(parsed.get("new_synthesis")))
            except Exception as exc:
                print(f"[ChatV2] Wiki update error: {exc}")

        # ========== STAGE 6: Logging ==========
        pages_used = [p.get("title", "Unknown") for p in persona_pages + wiki_results[:3]]

        log_to_wiki_log(
            "query",
            user_message[:100],
            {
                "endpoint": "chat-v2-fast (optimized)",
                "pages_consulted": pages_used,
                "response_length": len(full_response),
                "wiki_updated": wiki_updated,
            }
        )

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
    wiki_pages = WIKI_STORE.get_all_pages()
    return jsonify({
        "status": "ok",
        "model": CLAUDE_MODEL,
        "wiki_pages": len(wiki_pages),
        "rag_chunks": len(CHUNKS),
        "has_embeddings": INDEX.has_embeddings,
        "wiki_store": "redis" if WIKI_STORE.is_dynamic() else "static",
    })


# ---------------------------------------------------------------------------
# Static file serving — all requests routed through Flask via vercel.json
# ---------------------------------------------------------------------------

STATIC_DIR = str(Path(__file__).resolve().parent.parent)


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


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    app.run(debug=True, port=5000)
