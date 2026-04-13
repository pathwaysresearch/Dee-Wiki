# Two-Tier LLM Wiki + RAG Knowledge Chatbot

A hybrid knowledge architecture that combines an LLM-maintained wiki (for synthesised insights) with vector RAG (for large source documents). Deployed as a streaming chatbot on Vercel.

Built for **Prof. Bhagwan Chowdhry** (Finance, ISB & UCLA Anderson) — adaptable to any professor or domain expert.

---

## Architecture

```
                   User Question
                        │
                   Hybrid Search
                  (92.4% semantic + 7.6% BM25)
                   ┌────┴────┐
              Wiki Pages   RAG Chunks
            (synthesised    (raw source
             knowledge)      excerpts)
                   └────┬────┘
                  Claude Sonnet 4.6
                  (streaming SSE)
                        │
                 ┌──────┴──────┐
              Answer     Wiki Update
            (to user)   (filed back)
```

**Two outputs per query**: the answer + a wiki update if new insight was synthesised. The knowledge base compounds over time.

### Two Tiers

| Tier | What | Where | Purpose |
|------|------|-------|---------|
| **Wiki** | LLM-maintained markdown pages | `wiki/` (Obsidian vault) | Synthesised knowledge, cross-referenced |
| **RAG** | Chunked + embedded source documents | `data/chunks.json` | Raw retrieval from books, papers, interviews |

### Routing Rule

- **Under 5,000 words** → Wiki (full ingest as markdown pages)
- **Over 5,000 words** → RAG chunks + wiki stub page

---

## Adapting for Another Professor

To build this for a different domain expert:

### 1. Gather Source Materials

Place files in `Vault/raw/`:

```
raw/
  profile/       ← Digital profile, CV, bio (.md)
  research_papers/  ← Published papers (.pdf)
  books/          ← Textbooks or monographs (.md)
```

The profile is the most important source — it should contain interviews, op-eds, talks, opinions, and biographical material. The richer the profile, the better the chatbot captures the person's voice.

### 2. Run Ingest

```bash
# Set up environment
conda create -n wiki-rag python=3.11
conda activate wiki-rag
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Ingest source materials into RAG chunks
python scripts/ingest.py raw/books/*.md
python scripts/ingest.py raw/research_papers/*.pdf

# Or use Claude Code to ingest — it reads CLAUDE.md for routing rules
# and creates wiki pages automatically
```

### 3. Build Wiki Pages

This is where the real value lives. Use Claude Code (or any LLM) to:

1. Read the profile and source materials
2. Extract **tacit knowledge** — not just facts, but thinking patterns, rhetorical style, intellectual evolution
3. Create cross-referenced wiki pages in `wiki/`

The `CLAUDE.md` file contains detailed instructions for wiki page creation, routing rules, and the two-output query workflow.

### 4. Export and Deploy

```bash
# Export wiki + RAG data for the webapp
python scripts/export_for_web.py

# Deploy to Vercel (see Deployment section below)
```

### 5. Customise the Chatbot Voice

Edit the system prompt in [webapp/api/index.py](webapp/api/index.py) (`SYSTEM_PROMPT_TEMPLATE`). Replace:
- Name, title, institution
- Voice and style guidelines
- Content focus areas
- The persona description

---

## Project Structure

```
.
├── CLAUDE.md                          # LLM instructions for wiki maintenance
├── Vault/
│   └── wiki/                          # Tier 1: LLM-maintained wiki (Obsidian vault)
│       ├── index.md                   # Master catalog
│       ├── log.md                     # Chronological ingest/query log
│       ├── prof-bhagwan-chowdhry.md   # Main persona page
│       └── persona/                   # Tacit knowledge pages
├── scripts/
│   ├── ingest.py                      # PDF/MD → chunks + embeddings
│   ├── chunker.py                     # Text chunking + Gemini embeddings
│   ├── export_for_web.py              # Wiki + chunks → webapp/data/
│   └── sync_wiki.py                   # Sync wiki ↔ Upstash Redis
├── webapp/                            # Vercel-deployable Flask app
│   ├── api/
│   │   ├── index.py                   # Flask backend (hybrid search + streaming)
│   │   └── wiki_store.py              # Redis (dynamic) / JSON (static) wiki store
│   ├── data/                          # Exported data (bundled at deploy)
│   │   ├── wiki_pages.json            # Wiki pages with embeddings (~700KB)
│   │   ├── chunks.json                # RAG chunks, text only (~19MB)
│   │   └── chunks_embeddings.npy      # Embeddings as numpy binary (~68MB)
│   ├── public/                        # Frontend
│   │   ├── index.html
│   │   ├── style.css
│   │   └── script.js
│   ├── vercel.json                    # Vercel routing config
│   └── requirements.txt               # Python dependencies
└── .env.example                       # Required environment variables
```

---

## Deployment on Vercel

### Prerequisites

- A [Vercel](https://vercel.com) account (free tier works)
- An [Anthropic API key](https://console.anthropic.com) (for Claude)
- A [Google AI Studio API key](https://aistudio.google.com/apikey) (for Gemini embeddings)

### Steps

1. **Push to GitHub** (if not already done)

2. **Import in Vercel**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your GitHub repository
   - Set **Root Directory** to `webapp`
   - Framework Preset: **Other**

3. **Set Environment Variables** in Vercel Dashboard → Settings → Environment Variables:

   | Variable | Required | Purpose |
   |----------|----------|---------|
   | `ANTHROPIC_API_KEY` | Yes | Claude API for chat responses |
   | `GEMINI_API_KEY` | Yes | Gemini embeddings for semantic search |
   | `KV_REST_API_URL` | Optional | Upstash Redis URL for dynamic wiki |
   | `KV_REST_API_TOKEN` | Optional | Upstash Redis token |

4. **Deploy** — Vercel builds and deploys automatically

### Dynamic Wiki (Optional)

Without Redis, the wiki is read-only (served from bundled JSON). To enable wiki updates from conversations:

1. Create a free [Upstash Redis](https://upstash.com) database
2. Add `KV_REST_API_URL` and `KV_REST_API_TOKEN` to Vercel env vars
3. Seed Redis with current wiki: `python scripts/sync_wiki.py --push`

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the Flask dev server
cd webapp
python api/index.py
# Open http://localhost:5000
```

---

## Tech Stack

- **LLM**: Claude Sonnet 4.6 (streaming via SSE)
- **Embeddings**: Gemini `gemini-embedding-2-preview` (3072 dimensions)
- **Search**: Hybrid — 92.4% cosine similarity + 7.6% BM25 (rank_bm25)
- **Backend**: Flask on Vercel Python serverless
- **Frontend**: Vanilla HTML/CSS/JS + marked.js for markdown
- **Wiki Storage**: Upstash Redis (dynamic) or static JSON (fallback)
- **Wiki Authoring**: Obsidian (local) + Claude Code (LLM maintenance)
