# Prof. Bhagwan Chowdhry — Dual-LLM Wiki System

## Project Overview
A hybrid knowledge architecture for Prof. Bhagwan Chowdhry (Finance Professor, ISB & UCLA Anderson). Two LLMs collaborate: one maintains the wiki, one answers the user. BM25 over wiki pages (title + tags + aliases) for fast navigation; embedding similarity over RAG chunks for deep source lookup. Deployed as a Flask chatbot on Vercel / Cloud Run.

---

## Directory Layout
```
Vault/                            ← Obsidian vault
  wiki/                           ← Tier 1: LLM-maintained pages
    index.md                      ← Master catalog (always in LLM 1 system prompt)
    log.md                        ← Append-only chronological log
    _graph.json                   ← In-memory knowledge graph (nodes + typed edges)
    concepts/                     ← Concept pages
    entities/                     ← Entity pages
    persona/                      ← Persona pages
    stubs/                        ← RAG stub pages
    synthesized/                  ← Pages created from query synthesis
  raw/                            ← Tier 2: Immutable source files
    research_papers/              ← Paper PDFs
    books/                        ← Book .md files (large)
    profile/                      ← Digital profile .md
scripts/                          ← Ingest + export pipeline
webapp/                           ← Flask chatbot
data/                             ← Generated RAG chunks + embeddings
  chunks.json                     ← RAG text chunks
  chunks_embeddings.npy           ← Embedding vectors (one per chunk)
```

---

## Routing Rules

### Threshold
- **Wiki (full ingest)**: Files under 20 pages AND under 5,000 words
- **RAG + wiki stub**: Files over 20 pages OR over 5,000 words
- **RAG only**: Datasets, spreadsheets, raw numerical data

### Routing Table
| Source type | Route | Action |
|---|---|---|
| Digital profile (.md) | Wiki — full ingest | Create entity pages for Prof. Chowdhry, publications, research areas, contributions |
| Research paper PDF (<5000 words) | Wiki — full ingest | Extract text (PyMuPDF → Gemini fallback), create concept pages, cross-reference |
| Research paper PDF (≥5000 words) | RAG + wiki stub | Extract text, chunk for RAG, create stub page with title, abstract, 3–5 key claims, pointer |
| Book .md (typically large) | RAG + wiki stub | Chunk for RAG, create stub page |
| Interview / transcript | RAG + wiki stub | Chunk for RAG, create stub with key topics |
| Your own theoretical contributions | Wiki — full ingest | Core knowledge — needs cross-referencing and synthesis |
| Synthesised insight from a query | Wiki — file back | Update or create wiki page with new insight |

---

## LLM Roles

There are two LLMs with distinct, non-overlapping jobs.

### LLM 1 — Wiki Agent
- **Never talks to the user.**
- Navigates the wiki: given BM25 results and `index.md`, decides if context is sufficient or calls `graph_traverse`.
- Maintains the wiki: when `should_wiki_update: true` arrives from LLM 2, calls `update_wiki`.
- Tools: `graph_traverse(slug, hops, max_nodes)`, `update_wiki(structured page data)`

### LLM 2 — Answer Agent
- Talks to the user exclusively.
- Synthesises an answer from wiki context passed by LLM 1.
- Calls `rag_search` only when wiki context is insufficient.
- Always returns a structured JSON response (see schema below).
- Tool: `rag_search(query, top_k)`

---

## Query Pipeline

```
User query
  │
  ▼
BM25 search over wiki pages (title + tags + aliases) → top 2 pages (full content)
  │
  ▼
LLM 1 — wiki agent
  context: index.md (always) + top 2 pages + user query
  │
  ├─ SUFFICIENT (or no better page in index.md)
  │     └─► pass context to LLM 2
  │
  └─ NOT SUFFICIENT (index.md shows a better page BM25 missed)
        └─► graph_traverse(slug, hops=1, max_nodes=5)
              └─► neighbor pages (full content)
                    └─► pass all context to LLM 2
  │
  ▼
LLM 2 — answer agent
  context: wiki pages from LLM 1 + user query
  │
  ├─ SUFFICIENT → structured JSON response
  │
  └─ NOT SUFFICIENT → rag_search(query) [embedding similarity only]
        └─► RAG chunks → structured JSON response
  │
  ▼
answer → user (immediate)
  │
  └─ should_wiki_update: true → async update_wiki (fire-and-forget, never blocks user)
```

---

## Structured JSON Response (LLM 2)

LLM 2 always returns this exact schema — never plain text:

```json
{
  "answer": "Full conversational response. Close with a Sources block:\n\nMy Memory: [wiki page titles, or 'Found Nothing in My Memory']\nMy Library: [RAG source titles, or 'Found Nothing in My Library']\nGeneral Knowledge: [note any inferences beyond sources]",

  "sources": {
    "wiki": ["Page Title 1", "Page Title 2"],
    "rag":  ["Source Title 1"]
  },

  "new_synthesis": "Novel insight, connection, or resolved contradiction. Empty string if none.",

  "should_wiki_update": true
}
```

`should_wiki_update` is `true` when:
- The answer synthesises two or more wiki pages in a non-obvious way
- A contradiction was found and resolved
- RAG revealed something that updates or extends a wiki page
- The query produced a novel framing worth preserving

`should_wiki_update` is `false` when:
- Simple factual lookup from a single wiki page
- Answer is largely from general knowledge with no source synthesis

**Math formatting:** Use `(...)` for inline math, `[...]` for display math. Never use `$...$`.

---

## Wiki Page Format

Every wiki page is a `.md` file with YAML frontmatter encoding its graph relationships:

```markdown
---
type: concept
aliases: [Alternate Name]
relationships:
  - target: related-page-slug
    type: proposed_by
  - target: another-page-slug
    type: discussed_in
tags: [finance, theory]
---

# Page Title

Body text here.

## Relationships

- **proposed_by**: [[related-page-slug|Related Page]]
- **discussed_in**: [[another-page-slug|Another Page]]

---
*Extracted from: Source Title*
```

### Page Types
| Type | Description | Example |
|---|---|---|
| `entity` | Person, institution, initiative | `prof-bhagwan-chowdhry.md`, `fab-initiative.md` |
| `concept` | Idea, theory, mechanism | `microequity.md`, `systemic-risk.md` |
| `stub` | Pointer to RAG-indexed large document | `stub-driving-digital-strategy.md` |
| `synthesized` | Created from query synthesis | `wiki/synthesized/henrich-business-strategy.md` |
| `persona` | Core intellectual identity pages | `wiki/persona/deepa-mani.md` |

### Stub Page Format
```markdown
---
type: stub
tags: [...]
---

# [Document Title]

**Source**: `Vault/raw/books/filename.md`
**Pages**: N | **Words**: ~N

## Abstract
3–5 sentence summary.

## Key Claims
- Claim 1
- Claim 2
- Claim 3

## Relationships
- [[Related Wiki Page 1]]
- [[Related Wiki Page 2]]
```

### File Naming
- Use kebab-case: `financial-access-at-birth.md`
- Prefix stubs: `stub-title.md`
- Prefix source summaries: `summary-author-year.md`
- Synthesized pages live in `wiki/synthesized/`

---

## update_wiki Behavior

When LLM 1 calls `update_wiki`, the backend:
1. Writes `wiki/synthesized/{slug}.md` (atomic write)
2. Patches `_graph.json` in memory — adds new node + edges, no full rebuild needed
3. Rebuilds BM25 index (milliseconds)
4. Rebuilds `index.md`
5. Pushes `.md` + `index.md` to GitHub
6. Appends to `wiki/log.md`

Commit message format: `wiki: synthesize {slug} from query {date}`

---

## BM25 Search

- Runs over **wiki pages only** — RAG chunks are never BM25-indexed
- Full page content per wiki page: `title + aliases + tags + body` (entire `.md` content minus YAML frontmatter)
- Treats each wiki page as one atomic document — no chunking of wiki pages
- Returns top 2 pages with full content to LLM 1
- Rebuilt after every `update_wiki` call

RAG search uses **embedding similarity only** (cosine over `chunks_embeddings.npy`). BM25 is not applied to chunks.

---

## Ingest Workflow (offline pipeline)

When the user says **"ingest"**:

1. **Read this file** (CLAUDE.md) for routing rules
2. **Scan** `Vault/raw/` for all files
3. **Check** `Vault/wiki/log.md` for already-ingested files
4. **For each new file**:
   a. Count words / pages to determine routing
   b. **If wiki route**: Read the file, create/update wiki pages with YAML frontmatter, update cross-references and `_graph.json`
   c. **If RAG route**: Run `python scripts/ingest.py <file_path>` to chunk and embed, then create a stub wiki page
5. **Update** `Vault/wiki/index.md`
6. **Append** to `Vault/wiki/log.md`:
   ```
   ## [YYYY-MM-DD] ingest | Source Title
   - Route: wiki / RAG+stub
   - Pages created/updated: page1.md, page2.md, ...
   - Chunks: N (if RAG)
   ```

---

## Server Startup Sequence

At startup, held in memory for the lifetime of the instance:

1. Load `_graph.json` → graph dict (build from `wiki/*.md` if missing)
2. Build BM25 index from wiki pages (title + tags + aliases per page)
3. Load `data/chunks.json` → chunk list
4. Load `data/chunks_embeddings.npy` → numpy array
5. Load `Vault/wiki/index.md` → string (injected into every LLM 1 system prompt)
6. Ready to serve requests

---

## Two-Output Rule

**Every query produces two outputs:**
1. The answer to the user
2. A wiki update (if the answer contains insight not yet recorded)

The wiki update is always async — the user gets their answer immediately.

---

## Contradiction Handling

When a new source contradicts an existing wiki claim:
- Flag with ■ CONTRADICTION marker
- Record both views with sources
- Do NOT silently overwrite

---

## Periodic Lint

Run these health checks periodically:
- Find wiki pages with fewer than 2 inbound links (orphans)
- Identify claims that newer sources may have superseded
- Find concepts mentioned in multiple pages but lacking their own page
- List stub pages — do any now have enough cross-references to warrant full ingest?
- Check `_graph.json` for edges pointing to non-existent nodes
