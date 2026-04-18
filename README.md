# Two-Tier LLM Wiki + RAG Knowledge Chatbot

A hybrid knowledge architecture combining an LLM-maintained wiki (synthesised insights) with vector RAG (large source documents). Deployed as a streaming chatbot: **frontend on Vercel, backend on Google Cloud Run**.

Built for **Prof. Bhagwan Chowdhry** (Finance, ISB & UCLA Anderson) — adaptable to any professor or domain expert.

---

## Architecture

```
Browser
  │
  ├─ GET /  →  Vercel (static files): index.html, script.js, style.css
  │
  └─ POST /api/chat-v2  →  Cloud Run (Flask)
         │
         ├─ WIKI_LLM (Sonnet): hybrid BM25 + MiniLM search → top-5 pages
         │       iterative read_page tool to follow relationship chains
         │
         └─ MAIN_LLM (Opus): synthesise answer, optional RAG fallback
                 │
                 ├─ Answer (SSE stream → user)
                 └─ Wiki update (async, pushed to GitHub)
```

**Two outputs per query**: the answer streamed live + a wiki update filed back if new insight was synthesised. The knowledge base compounds over time.

### Two Tiers

| Tier | What | Where | Purpose |
|------|------|-------|---------|
| **Wiki** | LLM-maintained markdown pages | `Vault/wiki/` | Synthesised knowledge, cross-referenced graph |
| **RAG** | Chunked + embedded source documents | `data/chunks.json` | Raw retrieval from books, papers, interviews |

### Routing Rule

- **Under 5,000 words** → Wiki (full ingest as markdown pages)
- **Over 5,000 words** → RAG chunks + wiki stub page

---

## Project Structure

```
.
├── CLAUDE.md                          # LLM instructions for wiki maintenance
├── requirements.txt                   # Local dev dependencies
├── scripts/
│   ├── ingest.py                      # PDF/MD → RAG chunks + Gemini embeddings
│   ├── chunker.py                     # Text chunking library
│   ├── export_for_web.py              # Wiki + chunks → webapp/data/ (FAISS)
│   ├── graph.py                       # Build _graph.json from wiki YAML frontmatter
│   └── download_models.py             # Pre-download fastembed ONNX model
└── webapp/                            # Deployed app (Vercel + Cloud Run)
    ├── Dockerfile                     # Cloud Run backend image
    ├── .dockerignore
    ├── vercel.json                    # Vercel static routing (frontend only)
    ├── requirements.txt               # Backend Python dependencies
    ├── index.html                     # Frontend (static, served by Vercel)
    ├── script.js                      # Frontend JS — SSE streaming, markdown
    ├── style.css                      # Frontend styles
    ├── api/
    │   ├── index2.py                  # Flask backend — dual-LLM pipeline
    │   └── graph.py                   # Knowledge graph (co-located for Vercel)
    ├── data/                          # Exported data (baked into Docker image)
    │   ├── _graph.json                # Knowledge graph nodes + edges
    │   ├── chunks.json                # RAG chunks (text only)
    │   ├── chunks.faiss               # FAISS vector index over chunks
    │   ├── wiki_search.faiss          # FAISS index over wiki pages
    │   └── wiki_search_slugs.json     # Slug → FAISS row mapping
    ├── models/                        # fastembed ONNX model (committed to repo)
    │   └── models--qdrant--all-MiniLM-L6-v2-onnx/
    └── Vault/                         # Obsidian knowledge vault
        └── wiki/                      # Tier 1: LLM-maintained pages
            ├── concepts/
            ├── entities/
            ├── persona/
            ├── stubs/
            └── synthesized/
```

---

## Deployment

### Frontend — Vercel (static)

1. Push to GitHub
2. Import at [vercel.com/new](https://vercel.com/new) — set **Root Directory** to `webapp`
3. No environment variables needed on Vercel (all secrets stay in Cloud Run)
4. Vercel serves `index.html`, `script.js`, `style.css` as static files

After deploying the backend, set `window.BACKEND_URL` in `webapp/index.html` to your Cloud Run URL.

### Backend — Google Cloud Run (via GCP Console)

1. Go to **Cloud Run → Create Service**
2. Choose **"Continuously deploy from a repository"** → connect `Dee-Wiki` GitHub repo
3. Set **Build type** → `Dockerfile`, **Dockerfile location** → `/webapp/Dockerfile`
4. Set **Region**, **Memory** → 2 GiB, enable **Allow unauthenticated invocations**
5. Under **Environment variables**, add:

   | Variable | Value |
   |----------|-------|
   | `ANTHROPIC_API_KEY` | your Claude API key |
   | `GEMINI_API_KEY` | your Gemini API key |
   | `GITHUB_TOKEN` | your GitHub PAT |
   | `GITHUB_REPO` | `owner/repo` |
   | `ALLOWED_ORIGIN` | `https://your-app.vercel.app` |

6. Deploy — GCP builds the image from `webapp/Dockerfile` and gives you a service URL
7. Every `git push` to main auto-rebuilds and redeploys

**Environment Variables**

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude API (Sonnet + Opus) |
| `GEMINI_API_KEY` | Yes | Gemini embeddings for RAG search |
| `GITHUB_TOKEN` | Recommended | Push wiki updates back to GitHub |
| `GITHUB_REPO` | Recommended | `owner/repo` for wiki push |
| `ALLOWED_ORIGIN` | Recommended | Vercel frontend URL for CORS |

---

## Adding New Material (Ingest Workflow)

All source material lives in the git repo — no external storage service needed.

```bash
# 1. Clone the repo locally
git clone https://github.com/pathwaysresearch/Dee-Wiki
cd Dee-Wiki

# 2. Add source files to Vault/raw/
cp new-paper.pdf Vault/raw/research_papers/

# 3. Run ingest (creates RAG chunks + wiki stub)
pip install -r requirements.txt
python scripts/ingest.py Vault/raw/research_papers/new-paper.pdf

# 4. Export updated data for the webapp
python scripts/export_for_web.py

# 5. Push to GitHub
git add . && git commit -m "ingest: new paper" && git push

# 6. Cloud Run auto-redeploys on push (if set up with continuous deployment)
#    Otherwise trigger manually: GCP Console → Cloud Run → your service → Edit & Deploy New Revision
```

Wiki pages synthesised at runtime are pushed to GitHub automatically. They are picked up in the next Cloud Run deploy.

---

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys

# Run Flask dev server (serves frontend + API on same origin)
python webapp/api/index2.py --serve
# Open http://localhost:5001
```

For Docker local testing:

```bash
cd webapp
docker build -t dee-backend .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=... \
  -e GEMINI_API_KEY=... \
  dee-backend
# Open http://localhost:8080
```

---

## Tech Stack

- **LLM**: Claude Sonnet 4.6 (wiki navigation) + Claude Opus 4.6 (answer synthesis)
- **Embeddings**: `all-MiniLM-L6-v2` via fastembed (ONNX, no PyTorch) + Gemini for RAG
- **Search**: Hybrid BM25 (30%) + MiniLM cosine (70%) over wiki; FAISS cosine over RAG chunks
- **Backend**: Flask on Google Cloud Run (containerised, 2GB RAM)
- **Frontend**: Vercel static hosting — vanilla HTML/CSS/JS, marked.js, KaTeX
- **Wiki Storage**: GitHub (source of truth) — baked into Docker image at deploy time
- **Wiki Authoring**: Obsidian (local) + Claude Code (LLM maintenance via `update_wiki`)
