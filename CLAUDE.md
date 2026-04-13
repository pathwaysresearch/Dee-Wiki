# Prof. Bhagwan Chowdhry — Two-Tier LLM Wiki

## Project Overview
A hybrid knowledge architecture for Prof. Bhagwan Chowdhry (Finance Professor, ISB & UCLA Anderson). LLM-maintained wiki for synthesis, vector RAG for large raw sources. Deployed as a Flask chatbot on Vercel.

---

## Directory Layout
```
prof-bhagwan-hybrid-demo/         ← Obsidian vault
  wiki/                           ← Tier 1: LLM-maintained pages
    index.md                      ← Master catalog (read before every query)
    log.md                        ← Append-only chronological log
  raw/                            ← Tier 2: Immutable source files
    research_papers/              ← Paper PDFs
    books/                        ← Book .md files (large)
    profile/                      ← Digital profile .md
scripts/                          ← Ingest + export pipeline
webapp/                           ← Vercel-deployable Flask chatbot
data/                             ← Generated RAG chunks + embeddings
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

## Wiki Conventions

### Page Types
- **Entity pages**: People, institutions, initiatives (e.g., `Prof-Bhagwan-Chowdhry.md`, `FAB-Initiative.md`)
- **Concept pages**: Ideas, theories, mechanisms (e.g., `Microequity.md`, `Systemic-Risk.md`)
- **Source pages**: Summaries of ingested sources (e.g., `Summary-Acemoglu-2024.md`)
- **Stub pages**: Pointers to RAG-indexed large documents

### Stub Page Format
```markdown
# [Document Title]
**Type**: RAG stub
**Source**: `raw/books/filename.md` or `raw/research_papers/filename.pdf`
**Pages**: N | **Words**: ~N

## Abstract
3–5 sentence summary of the document.

## Key Claims
- Claim 1
- Claim 2
- Claim 3

## Cross-References
- [[Related Wiki Page 1]]
- [[Related Wiki Page 2]]
```

### File Naming
- Use kebab-case: `financial-access-at-birth.md`
- Prefix source summaries: `summary-author-year.md`
- Prefix stubs: `stub-title.md`

---

## Ingest Workflow

When the user says **"ingest"**, follow these steps:

1. **Read this file** (CLAUDE.md) for routing rules
2. **Scan** `prof-bhagwan-hybrid-demo/raw/` for all files
3. **Check** `wiki/log.md` for already-ingested files
4. **For each new file**:
   a. Count words / pages to determine routing
   b. **If wiki route**: Read the file, create/update 5–15 wiki pages (entity, concept, source pages), update cross-references
   c. **If RAG route**: Run `python scripts/ingest.py <file_path>` to chunk and embed, then create a stub wiki page
5. **Update** `wiki/index.md` with new page entries
6. **Append** to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] ingest | Source Title
   - Route: wiki / RAG+stub
   - Pages created/updated: page1.md, page2.md, ...
   - Chunks: N (if RAG)
   ```

---

## Query Workflow (for Claude Code sessions)

1. **Read** `wiki/index.md` to find relevant pages
2. **Read** relevant wiki pages
3. **Attempt** to answer from wiki content
4. **If insufficient**: search RAG chunks (`python scripts/ingest.py --search "query"`)
5. **Synthesise** answer from wiki + RAG
6. **File back**: If answer contains insight not in the wiki, update or create the relevant page
7. **Append** to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] query | Question summary
   - Pages consulted: page1.md, page2.md
   - Wiki updated: yes/no
   ```

---

## Two-Output Rule

**Every query produces two outputs:**
1. The answer to the user
2. A wiki update (if the answer contains insight not yet recorded)

This is what makes the knowledge base compound over time.

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
- List RAG stub pages — do any now have enough cross-references to warrant full ingest?
