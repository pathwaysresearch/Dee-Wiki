"""
auto_wiki_builder.py — Automates wiki page creation using an LLM API.

  --stubs   : Generate RAG stub pages for all ingested sources
  --index   : Rebuild wiki/index.md as a complete catalog
  --all     : Run stubs → index in sequence

Usage:
    python scripts/auto_wiki_builder.py --stubs
    python scripts/auto_wiki_builder.py --index
    python scripts/auto_wiki_builder.py --all
    python scripts/auto_wiki_builder.py --all --force
    python scripts/auto_wiki_builder.py --all --model claude-sonnet-4-6
    python scripts/auto_wiki_builder.py --all --provider openai --model gpt-4o
"""

import os
import sys
import json
import time
import re
import argparse
import concurrent.futures
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
VAULT = PROJECT_ROOT / os.environ.get("WIKI_VAULT_NAME", "webapp/Vault")
WIKI_DIR = VAULT / "wiki"
RAW_DIR = VAULT / "raw"
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_FILE = DATA_DIR / "chunks.json"
INGESTED_FILE = DATA_DIR / "ingested.json"

STUBS_DIR = WIKI_DIR / "stubs"
INDEX_FILE = WIKI_DIR / "index.md"

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# LLM configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-4-6"
MAX_TOKENS_STUB = 1200
STUB_WORKERS = 4
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


# ---------------------------------------------------------------------------
# LLM Client — model-agnostic abstraction
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Model-agnostic LLM client supporting Anthropic and OpenAI providers.
    Retry logic with exponential backoff is handled here.
    """

    SUPPORTED_PROVIDERS = {"anthropic", "openai"}

    def __init__(self, provider: str, model: str, api_key: str):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {sorted(self.SUPPORTED_PROVIDERS)}"
            )
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )

    def call(self, system: str, user: str, max_tokens: int) -> str:
        """Make an LLM call. Retries up to MAX_RETRIES on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._call_once(system, user, max_tokens)
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(
                        f"    [LLM] Retry {attempt + 1}/{MAX_RETRIES - 1} "
                        f"in {delay}s ({type(exc).__name__}: {exc})"
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts. Last: {last_exc}"
        )

    def _call_once(self, system: str, user: str, max_tokens: int) -> str:
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.provider}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sanitize_slug(text: str) -> str:
    """Convert any string to a safe kebab-case filesystem slug (max 80 chars)."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug[:80]


def _load_chunks() -> list:
    """Load all RAG chunks from data/chunks.json. Returns [] on any failure."""
    if not CHUNKS_FILE.exists():
        print(f"  [WARN] chunks.json not found at {CHUNKS_FILE}")
        return []
    try:
        return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Could not load chunks.json: {exc}")
        return []


def _write_page_atomic(path: Path, content: str, force: bool) -> bool:
    """
    Atomically write content to path (temp file → os.replace).
    Returns True if written, False if skipped (exists and not force).
    """
    if path.exists() and not force:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
        return True
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap output in."""
    text = text.strip()
    text = re.sub(r"^```(?:markdown|json|yaml)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _discover_rag_sources(
    skip_dirs: tuple = ("research_papers_md", "profile")
) -> list:
    """
    Build list of RAG source dicts: {slug, name, source_path, chunks, words}.
    Primary source: data/ingested.json. Fallback: unique source paths in chunks.json.
    Always skips profile sources and derived/converted directories.
    """
    sources = []
    seen_slugs: set = set()

    def _should_skip(path_str: str) -> bool:
        normalized = path_str.replace("\\", "/")
        return any(f"/{skip}/" in normalized or normalized.endswith(f"/{skip}")
                   for skip in skip_dirs)

    if INGESTED_FILE.exists():
        try:
            ingested = json.loads(INGESTED_FILE.read_text(encoding="utf-8"))
            for rel_path, info in ingested.items():
                if _should_skip(rel_path):
                    continue
                stem = Path(rel_path).stem
                slug = _sanitize_slug(stem)
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                name = re.sub(r"\s+", " ", stem.replace("_", " ").replace("-", " ").title()).strip()
                sources.append({
                    "slug": slug,
                    "name": name,
                    "source_path": rel_path,
                    "chunks": info.get("chunks", 0),
                    "words": info.get("words", 0),
                })
            if sources:
                return sources
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] ingested.json read failed ({exc}), falling back to chunks.json")

    all_chunks = _load_chunks()
    chunk_counts: dict = {}
    for c in all_chunks:
        src = c.get("source", "")
        if src:
            chunk_counts[src] = chunk_counts.get(src, 0) + 1

    for src_path, count in chunk_counts.items():
        if _should_skip(src_path):
            continue
        stem = Path(src_path).stem
        slug = _sanitize_slug(stem)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        name = stem.replace("_", " ").replace("-", " ").title()
        sources.append({
            "slug": slug,
            "name": name,
            "source_path": src_path,
            "chunks": count,
            "words": 0,
        })

    return sources


def _scan_wiki_inventory() -> dict:
    """
    Scan all wiki/**/*.md and return structured inventory by category.
    Excludes: index.md, log.md, _graph.json, files with stem starting with '_'.
    """
    inventory = {
        "persona": [],
        "concepts": [],
        "entities": [],
        "stubs": [],
        "synthesized": [],
        "other": [],
    }

    if not WIKI_DIR.exists():
        return inventory

    SKIP_STEMS = {"index", "log"}

    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        stem = md_file.stem
        if stem in SKIP_STEMS or stem.startswith("_"):
            continue

        try:
            content = md_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if len(content) < 30:
            continue

        title = stem.replace("-", " ").replace("_", " ").title()
        description = ""
        in_frontmatter = False
        frontmatter_done = False
        dash_count = 0

        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "---":
                dash_count += 1
                if dash_count == 1:
                    in_frontmatter = True
                elif dash_count == 2:
                    in_frontmatter = False
                    frontmatter_done = True
                continue
            if in_frontmatter:
                continue
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped.lstrip("# ").strip()
            elif (frontmatter_done and stripped
                  and not stripped.startswith("#")
                  and not stripped.startswith("-")
                  and not stripped.startswith("*")
                  and not stripped.startswith(">")
                  and len(stripped) > 20
                  and not description):
                description = stripped[:120]

        rel_path = str(md_file.relative_to(VAULT))
        parts = Path(rel_path).parts

        entry = {
            "title": title,
            "path": rel_path,
            "slug": stem,
            "description": description,
        }

        if "persona" in parts:
            inventory["persona"].append(entry)
        elif "concepts" in parts:
            inventory["concepts"].append(entry)
        elif "entities" in parts:
            inventory["entities"].append(entry)
        elif "stubs" in parts or stem.startswith("stub-"):
            inventory["stubs"].append(entry)
        elif "synthesized" in parts:
            inventory["synthesized"].append(entry)
        else:
            inventory["other"].append(entry)

    return inventory


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_stub_prompt(source_info: dict, first_chunks: list) -> tuple:
    """Build prompt for a single RAG stub page. Returns (system, user)."""
    chunks_text = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c.get('content', '')[:800]}"
        for i, c in enumerate(first_chunks[:5])
    )

    name = source_info["name"]
    source_path = source_info["source_path"]
    chunk_count = source_info["chunks"]

    system = (
        "You are creating a RAG stub wiki page — a catalog entry for a large document "
        "fully indexed in a vector search system but too long to summarize in full.\n\n"
        "The stub's purpose: let a query system know this source EXISTS and what it contains "
        "so it can decide whether to retrieve chunks from it.\n\n"
        "Output ONLY raw markdown with YAML frontmatter. "
        "No code fences. No preamble. No commentary after the final line."
    )

    user = (
        f'Create a RAG stub wiki page for: "{name}"\n\n'
        f"Source path: {source_path}\n"
        f"Chunks in RAG index: {chunk_count}\n\n"
        f"Based on these opening excerpts:\n{chunks_text}\n\n"
        f"Output this exact format (replace placeholder text):\n\n"
        f"---\n"
        f"type: stub\n"
        f"aliases: [{name}]\n"
        f"relationships: []\n"
        f"tags: [rag-stub]\n"
        f"rag_source: {source_path}\n"
        f"rag_chunks: {chunk_count}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"**Type**: RAG stub — full content in vector index, not in wiki\n\n"
        f"## Abstract\n\n"
        f"[3-4 sentences: what this source covers, its main argument, why it matters]\n\n"
        f"## Key Claims\n\n"
        f"- [Specific claim 1]\n"
        f"- [Specific claim 2]\n"
        f"- [Specific claim 3]\n"
        f"- [Specific claim 4]\n"
        f"- [Specific claim 5]\n\n"
        f"## Topics Covered\n\n"
        f"[8-12 specific topics as a comma-separated list]\n\n"
        f"## How to Query\n\n"
        f'> "Explain [main topic] from {name}"\n'
        f'> "What does {name} say about [concept]?"\n\n'
        f"---\n"
        f"*RAG stub — {chunk_count} chunks indexed. Source: `{source_path}`*\n\n"
        f"Rules:\n"
        f"- Ground ALL claims in the provided excerpts\n"
        f"- Do not invent content not supported by the text\n"
        f"- Replace every placeholder line above with real content"
    )

    return system, user


def _build_index(inventory: dict) -> str:
    """
    Generate wiki/index.md programmatically from the page inventory.
    No LLM call — deterministic, instant, and never truncates regardless of
    how many pages exist.
    """
    from datetime import date

    def _entry(p: dict) -> str:
        rel = p["path"].replace("\\", "/")
        if rel.startswith("wiki/"):
            rel = rel[5:]
        if rel.endswith(".md"):
            rel = rel[:-3]
        desc = f" — {p['description']}" if p.get("description") else ""
        return f"- [[{rel}|{p['title']}]]{desc}"

    def _section(header: str, pages: list, empty_note: str = "") -> str:
        """Return a section block ending with a single trailing newline."""
        lines = [header, ""]
        if pages:
            lines += [_entry(p) for p in pages]
        else:
            lines.append(f"<!-- {empty_note or 'None yet'} -->")
        return "\n".join(lines)

    today = date.today().isoformat()
    total = sum(len(v) for v in inventory.values())

    sections = [
        f"# Wiki Index",
        f"Master catalog — {total} pages · last rebuilt {today}",
    ]

    if inventory["persona"]:
        sections.append(_section(
            f"## Persona Pages ({len(inventory['persona'])})",
            inventory["persona"],
        ))

    sections.append(_section(
        f"## Extracted Concepts ({len(inventory['concepts'])})",
        inventory["concepts"],
        "No concept pages yet — run extract_entities.py",
    ))

    sections.append(_section(
        f"## Extracted Entities ({len(inventory['entities'])})",
        inventory["entities"],
        "No entity pages yet — run extract_entities.py",
    ))

    sections.append(_section(
        f"## Query-Synthesized Pages ({len(inventory['synthesized'])})",
        inventory["synthesized"],
        "No synthesized pages yet",
    ))

    if inventory["stubs"]:
        stub_lines = [f"## RAG Stubs ({len(inventory['stubs'])})", ""]
        stub_lines += [_entry(p) for p in inventory["stubs"]]
        sections.append("\n".join(stub_lines))
    else:
        sections.append("## RAG Stubs (0)\n\n<!-- No stubs yet — run auto_wiki_builder.py --stubs -->")

    if inventory["other"]:
        sections.append(_section(
            f"## Other ({len(inventory['other'])})",
            inventory["other"],
        ))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_stubs(llm: LLMClient, force: bool = False) -> int:
    """Generate RAG stub pages for all ingested sources in parallel."""
    print("\n=== GENERATING RAG STUBS ===\n")

    STUBS_DIR.mkdir(parents=True, exist_ok=True)

    sources = _discover_rag_sources()
    if not sources:
        print("  No RAG sources found. Run: python scripts/ingest.py --process-all")
        return 0

    all_chunks = _load_chunks()
    n = len(sources)

    est_in = n * 3500
    est_out = n * 500
    print(f"  Sources to stub: {n}")
    print(f"  Estimated tokens: ~{est_in:,} input + ~{est_out:,} output")
    print(f"  Existing stubs will be {'OVERWRITTEN' if force else 'skipped (use --force to overwrite)'}\n")

    written = skipped = failed = 0

    def _process_stub(source_info: dict):
        slug = source_info["slug"]
        stub_path = STUBS_DIR / f"stub-{slug}.md"
        legacy_path = WIKI_DIR / f"stub-{slug}.md"

        if (stub_path.exists() or legacy_path.exists()) and not force:
            return "skipped", slug, None

        src_normalized = source_info["source_path"].replace("\\", "/")
        src_stem = Path(source_info["source_path"]).stem.lower()

        source_chunks = [
            c for c in all_chunks
            if (src_normalized in c.get("source", "").replace("\\", "/"))
            or (src_stem in Path(c.get("source", "")).stem.lower())
        ][:5]

        if not source_chunks:
            return "failed", slug, "No matching chunks found in chunks.json"

        try:
            system, user = _build_stub_prompt(source_info, source_chunks)
            content = llm.call(system, user, MAX_TOKENS_STUB)
            content = _strip_code_fences(content)
            _write_page_atomic(stub_path, content + "\n", force)
            return "written", slug, str(stub_path.relative_to(VAULT))
        except Exception as exc:
            return "failed", slug, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=STUB_WORKERS) as executor:
        future_to_slug = {executor.submit(_process_stub, s): s["slug"] for s in sources}
        for future in concurrent.futures.as_completed(future_to_slug):
            status, slug, detail = future.result()
            if status == "written":
                print(f"  ✓ stub-{slug}")
                written += 1
            elif status == "skipped":
                print(f"  ─ stub-{slug} (exists)")
                skipped += 1
            else:
                print(f"  ✗ stub-{slug}: {detail}")
                placeholder = (
                    f"---\ntype: stub\naliases: []\nrelationships: []\n"
                    f"tags: [rag-stub, generation-failed]\n"
                    f"rag_source: {future_to_slug.get(future, 'unknown')}\n"
                    f"rag_chunks: 0\n---\n\n"
                    f"# {slug.replace('-', ' ').title()}\n\n"
                    f"**Type**: RAG stub — GENERATION FAILED\n\n"
                    f"<!-- Error: {str(detail)[:200]} -->\n"
                    f"<!-- Re-run: python scripts/auto_wiki_builder.py --stubs --force -->\n"
                )
                try:
                    _write_page_atomic(
                        STUBS_DIR / f"stub-{slug}.md", placeholder, force=True
                    )
                except OSError:
                    pass
                failed += 1

    print(f"\n  Stubs — written: {written}, skipped: {skipped}, failed: {failed}")
    return written


def cmd_index() -> bool:
    """Rebuild wiki/index.md programmatically — no LLM, never truncates."""
    print("\n=== REBUILDING INDEX.MD ===\n")

    inventory = _scan_wiki_inventory()
    total = sum(len(v) for v in inventory.values())

    print(f"  Scanned {total} wiki pages:")
    for cat, pages in inventory.items():
        if pages:
            print(f"    {cat}: {len(pages)}")

    try:
        content = _build_index(inventory)
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_FILE.write_text(content + "\n", encoding="utf-8")
        print(f"  ✓ index.md written ({len(content):,} chars, {total} pages catalogued)")
        return True
    except Exception as exc:
        print(f"  ✗ index.md generation failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(operation: str, description: str, metadata: Optional[dict] = None):
    """Write to wiki_logger if available, otherwise stdout only."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from wiki_logger import log_to_wiki_log
        log_to_wiki_log(operation, description, metadata)
    except ImportError:
        ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
        meta_str = " | " + json.dumps(metadata) if metadata else ""
        print(f"[Log] [{ts}] {operation} | {description}{meta_str}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate wiki pages using an LLM API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/auto_wiki_builder.py --stubs\n"
            "  python scripts/auto_wiki_builder.py --index\n"
            "  python scripts/auto_wiki_builder.py --all\n"
            "  python scripts/auto_wiki_builder.py --all --force\n"
            "  python scripts/auto_wiki_builder.py --all --model claude-sonnet-4-6\n"
            "  python scripts/auto_wiki_builder.py --all --provider openai --model gpt-4o\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stubs", action="store_true", help="Generate RAG stub pages")
    group.add_argument("--index", action="store_true", help="Rebuild wiki/index.md")
    group.add_argument("--all", action="store_true", help="Run stubs + index")

    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pages (default: skip)")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=["anthropic", "openai"],
                        help=f"LLM provider (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    # --index only: no LLM needed
    llm = None
    needs_llm = args.stubs or args.all
    if needs_llm:
        key_env = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(key_env, "")
        if not api_key:
            print(f"ERROR: {key_env} not set. Add it to .env or export it.")
            sys.exit(1)
        try:
            llm = LLMClient(provider=args.provider, model=args.model, api_key=api_key)
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR initializing LLM client: {exc}")
            sys.exit(1)
        print(f"Provider: {args.provider} | Model: {args.model}")

    if args.force:
        print("Mode: --force (existing pages WILL be overwritten)\n")

    start = time.time()
    stubs_written = 0
    index_ok = False

    if args.stubs or args.all:
        stubs_written = cmd_stubs(llm, force=args.force)

    if args.index or args.all:
        index_ok = cmd_index()

    # Rebuild knowledge graph after writing new pages
    if stubs_written > 0:
        print("\n=== UPDATING KNOWLEDGE GRAPH ===\n")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from graph import update_graph
            graph = update_graph()
            n_nodes = len(graph.get("nodes", {}))
            n_edges = len(graph.get("edges", []))
            print(f"  ✓ Graph updated: {n_nodes} nodes, {n_edges} edges")
        except ImportError:
            print("  [SKIP] graph.py not found — run: python scripts/graph.py --build")
        except Exception as exc:
            print(f"  ✗ Graph update failed: {exc}")

    elapsed = time.time() - start

    _log(
        "auto_wiki_builder",
        "Automated wiki page generation complete",
        {
            "model": args.model,
            "stubs_written": stubs_written,
            "index_rebuilt": index_ok,
            "elapsed_s": round(elapsed, 1),
        },
    )

    print(f"\n{'=' * 52}")
    print(f"  Completed in {elapsed:.1f}s")
    if args.stubs or args.all:
        print(f"  Stubs written:  {stubs_written}")
    if args.index or args.all:
        print(f"  Index rebuilt:  {'✓' if index_ok else '✗'}")
    print(f"\n  Next steps:")
    print(f"    python scripts/export_for_web.py")
    print(f"    python scripts/sync_wiki.py --seed")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    main()
