"""
auto_wiki_builder.py — Automates wiki page creation using an LLM API.

Replaces the "Claude Code" manual steps in the wiki setup workflow:
  --stubs   : Generate RAG stub pages for all ingested sources (Step 4)
  --persona : Generate tacit knowledge persona pages using 3-pass extraction (Step 3)
  --index   : Rebuild wiki/index.md as a complete catalog (Step 5)
  --all     : Run stubs → persona → index in sequence

Usage:
    python scripts/auto_wiki_builder.py --stubs
    python scripts/auto_wiki_builder.py --persona
    python scripts/auto_wiki_builder.py --index
    python scripts/auto_wiki_builder.py --all
    python scripts/auto_wiki_builder.py --all --force
    python scripts/auto_wiki_builder.py --all --model claude-sonnet-4-6
    python scripts/auto_wiki_builder.py --all --provider openai --model gpt-4o
    python scripts/auto_wiki_builder.py --persona --profile "raw/profile/myprofile.md"

Three-pass persona extraction (adapted from Cho et al. 2024):
  Pass 1 — Content selection: What they teach/emphasize, deliberate omissions, signature examples
  Pass 2 — Communication style: Rhetorical moves, scaffolding, cross-domain bridges
  Pass 3 — Page generation: One LLM call per wiki page using both analyses as context

Both Pass 1 and Pass 2 results are cached to data/ so individual pages can be re-run
without repeating the expensive analysis calls.
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
# Paths — resolve relative to this script's parent.parent (project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vault name: set WIKI_VAULT_NAME in .env to override (useful for new avatar clones)
# Default matches the renamed vault folder
_vault_name = os.environ.get("WIKI_VAULT_NAME", "Vault")
VAULT = PROJECT_ROOT / _vault_name
WIKI_DIR = VAULT / "wiki"
RAW_DIR = VAULT / "raw"
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_FILE = DATA_DIR / "chunks.json"
INGESTED_FILE = DATA_DIR / "ingested.json"
ANALYSIS_FILE = DATA_DIR / "persona_analysis.json"
COMM_ANALYSIS_FILE = DATA_DIR / "persona_comm_analysis.json"

PERSONA_DIR = WIKI_DIR / "persona"
STUBS_DIR = WIKI_DIR / "stubs"
INDEX_FILE = WIKI_DIR / "index.md"

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# LLM configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-4-6"          # correct Anthropic model string as of April 2026
MAX_TOKENS_ANALYSIS = 4096
MAX_TOKENS_PAGE = 3000
MAX_TOKENS_STUB = 1200
MAX_TOKENS_INDEX = 6000
STUB_WORKERS = 4                            # parallel threads for stub generation
PROFILE_CHUNKS_FOR_ANALYSIS = 40           # broad sweep for Pass 1
CHUNKS_PER_PAGE = 20                       # targeted chunks per persona page
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2                       # seconds (doubles each retry)


# ---------------------------------------------------------------------------
# LLM Client — model-agnostic abstraction
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Model-agnostic LLM client supporting Anthropic and OpenAI providers.

    Both providers receive identical (system, user) string pairs.
    Retry logic with exponential backoff is handled here so callers
    don't need to worry about transient API errors.
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
    slug = re.sub(r"[^\w\s-]", "", slug)     # strip non-word chars except hyphen
    slug = re.sub(r"[\s_]+", "-", slug)       # spaces/underscores → hyphens
    slug = re.sub(r"-{2,}", "-", slug)        # collapse multiple hyphens
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


def _bm25_select_chunks(
    all_chunks: list,
    query: str,
    source_filter: Optional[str],
    top_k: int,
) -> list:
    """
    Return top_k chunks ranked by BM25 score for query.
    Optionally pre-filters by source path substring.
    Falls back to sequential order if rank_bm25 is not installed.
    """
    filtered = all_chunks
    if source_filter:
        sf = source_filter.lower()
        filtered = [
            c for c in all_chunks
            if sf in c.get("source", "").lower().replace("\\", "/")
        ]

    if not filtered:
        return []

    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
    except ImportError:
        # Fallback: return first top_k without scoring
        return filtered[:top_k]

    tokenized = [c.get("content", "").lower().split() for c in filtered]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    top_idx = np.argsort(scores)[::-1][:top_k]
    # Only return chunks with non-zero score
    result = [filtered[i] for i in top_idx if scores[i] > 0]
    if not result:
        result = filtered[:top_k]   # fallback if all scores are 0
    return result


def _write_page_atomic(path: Path, content: str, force: bool) -> bool:
    """
    Atomically write content to path (temp file → os.replace).
    Returns True if written, False if skipped (exists and not force).
    Raises OSError on write failure.
    """
    if path.exists() and not force:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)   # atomic on POSIX
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


def _parse_json_response(text: str, label: str) -> Optional[dict]:
    """
    Robustly parse a JSON object from LLM output.
    Handles: code fences, leading preamble, trailing commentary,
    and the common LLM bug of embedding literal newlines inside JSON strings.
    """
    text = _strip_code_fences(text)

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 3: fix literal newlines inside JSON strings
    # Replace \n inside "quoted strings" with a space
    def _fix_string_newlines(m):
        return m.group(0).replace("\n", " ").replace("\r", "")

    fixed = re.sub(r'"(?:[^"\\]|\\.)*"', _fix_string_newlines, text, flags=re.DOTALL)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    print(f"  [WARN] Could not parse {label} JSON from LLM response "
          f"(first 300 chars: {text[:300]!r})")
    return None


def _get_profile_source_filter(profile_arg: Optional[str]) -> str:
    """
    Determine the BM25 source filter string for profile chunks.
    Priority order:
      1. Explicit --profile CLI argument (uses filename)
      2. First .md file found in raw/profile/
      3. Falls back to 'knowledge_base' as a heuristic
    """
    if profile_arg:
        return Path(profile_arg).name

    profile_dir = RAW_DIR / "profile"
    if profile_dir.exists():
        md_files = sorted(profile_dir.glob("*.md"))
        if md_files:
            return md_files[0].name

    return "knowledge_base"


def _discover_rag_sources(
    skip_dirs: tuple = ("research_papers_md", "profile")
) -> list:
    """
    Build list of RAG source dicts: {slug, name, source_path, chunks, words}.
    Primary source: data/ingested.json (structured tracking).
    Fallback: unique source paths in data/chunks.json.
    Always skips profile sources and derived/converted directories.
    """
    sources = []
    seen_slugs: set = set()

    def _should_skip(path_str: str) -> bool:
        normalized = path_str.replace("\\", "/")
        return any(f"/{skip}/" in normalized or normalized.endswith(f"/{skip}")
                   for skip in skip_dirs)

    # ── Primary: ingested.json ────────────────────────────────────────────
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
                name = stem.replace("_", " ").replace("-", " ").title()
                # Clean up common filename noise
                name = re.sub(r"\s+", " ", name).strip()
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

    # ── Fallback: chunks.json unique sources ──────────────────────────────
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
    Returns dict keyed by category with list of page dicts.
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

        # Extract H1 title and first substantive description line
        title = stem.replace("-", " ").replace("_", " ").title()
        description = ""
        in_frontmatter = False
        frontmatter_done = False
        dash_count = 0

        for line in content.splitlines():
            stripped = line.strip()

            # Handle YAML frontmatter
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
        parts = Path(rel_path).parts  # e.g. ('wiki', 'concepts', 'slug.md')

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
# Prompt builders — three-pass persona extraction + stubs + index
# ---------------------------------------------------------------------------

def _build_pass1_prompt(profile_chunks: list, person_name: str, domain: str) -> tuple:
    """
    Pass 1: Content Selection Analysis (adapted from Cho et al. 2024).

    Analyzes WHAT the expert chooses to emphasize, what they omit, what
    examples they return to repeatedly, and their cross-domain reach.
    This mirrors the 'observed expert behavior' methodology — we give the LLM
    the actual body of work and ask it to surface implicit selection reasoning.

    Returns (system_prompt, user_prompt).
    """
    chunks_text = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c.get('content', '')[:700]}"
        for i, c in enumerate(profile_chunks[:PROFILE_CHUNKS_FOR_ANALYSIS])
    )

    system = (
        f"You are extracting tacit knowledge from the public work of {person_name}, "
        f"an expert in {domain}.\n\n"
        "Following the Cho et al. (2024) methodology for tacit knowledge externalization, "
        "you analyze OBSERVED EXPERT BEHAVIOR to surface the implicit reasoning behind "
        "their intellectual choices — what they emphasize, what they omit, what they "
        "return to repeatedly, and how they bridge domains.\n\n"
        "Output ONLY valid JSON with no markdown fences, no preamble, no commentary.\n"
        "All string values must be on a SINGLE LINE (no literal newlines inside strings)."
    )

    user = (
        f"Analyze the following text excerpts from {person_name}'s public work "
        f"(articles, interviews, research summaries, lectures) and extract the tacit "
        f"knowledge embedded in their intellectual choices.\n\n"
        f"TEXT EXCERPTS:\n{chunks_text}\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "person_name": "Full Name",\n'
        '  "person_slug": "kebab-case-name",\n'
        '  "primary_domain": "their main field",\n'
        '  "primary_affiliation": "university or institution",\n'
        '  "career_phases": [\n'
        '    {"name": "Phase Name", "years": "YYYY-YYYY", "description": "one sentence"}\n'
        "  ],\n"
        '  "core_intellectual_themes": [\n'
        '    {\n'
        '      "theme": "Theme Name",\n'
        '      "why_it_matters_to_them": "one specific sentence citing text evidence",\n'
        '      "key_examples": ["specific example 1", "specific example 2"]\n'
        "    }\n"
        "  ],\n"
        '  "rhetorical_patterns": [\n'
        '    {"pattern": "Pattern Name", "example": "brief concrete example from their work"}\n'
        "  ],\n"
        '  "key_influences": [\n'
        '    {"name": "Person Name", "relationship": "advisor/collaborator/etc", "ideas_borrowed": "one sentence"}\n'
        "  ],\n"
        '  "signature_examples": ["specific case/company/example they reuse"],\n'
        '  "deliberate_omissions": ["topic they avoid despite mainstream usage"],\n'
        '  "domain_pages_needed": [\n'
        '    {"slug": "kebab-case", "title": "Title Case", "description": "one sentence"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Be SPECIFIC — use actual names, titles, and phrases from the text\n"
        "- Limit domain_pages_needed to 2-3 most important domains\n"
        "- Limit core_intellectual_themes to 4-6 most prominent\n"
        "- All values on single lines (no embedded newlines)"
    )

    return system, user


def _build_pass2_prompt(
    profile_chunks: list, person_name: str, pass1_analysis: dict
) -> tuple:
    """
    Pass 2: Communication and Rhetorical Analysis (adapted from Cho et al. 2024 Pass 2).

    Analyzes HOW the expert communicates — opening moves, scaffolding logic,
    cross-domain bridges, and what they never do rhetorically.
    Uses BM25 to select communication-relevant chunks.

    Returns (system_prompt, user_prompt).
    """
    # Target chunks about explanation and communication
    comm_chunks = _bm25_select_chunks(
        profile_chunks,
        "explain teach example analogy simple illustrate scaffold argue persuade",
        None,
        25,
    )
    if not comm_chunks:
        comm_chunks = profile_chunks[:25]

    chunks_text = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c.get('content', '')[:600]}"
        for i, c in enumerate(comm_chunks)
    )

    themes_summary = "; ".join(
        t["theme"]
        for t in pass1_analysis.get("core_intellectual_themes", [])
    )
    patterns_summary = "; ".join(
        r["pattern"]
        for r in pass1_analysis.get("rhetorical_patterns", [])
    )

    system = (
        f"You are extracting the communication and pedagogical tacit knowledge of {person_name}.\n\n"
        "You have already identified their content selection patterns (provided in context). "
        "Now focus on HOW they communicate — their rhetorical opening moves, how they scaffold "
        "complex ideas step-by-step, what cross-domain analogies they use, and what they "
        "consistently avoid.\n\n"
        "Output ONLY valid JSON. No markdown fences. No preamble. "
        "All string values on a single line."
    )

    user = (
        f"Context: {person_name}'s core themes are: {themes_summary}\n"
        f"Their known rhetorical patterns include: {patterns_summary}\n\n"
        f"Now analyze HOW they communicate from these excerpts:\n\n"
        f"{chunks_text}\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "opening_moves": "How they typically begin explanations — counterintuitive claim, concrete case, historical narrative, etc. With example.",\n'
        '  "scaffolding_approach": "How they build from simple to complex — describe the specific progression pattern with an example.",\n'
        '  "signature_rhetorical_moves": [\n'
        '    {"move": "move name", "example": "concrete example from their work"}\n'
        "  ],\n"
        '  "cross_domain_bridges": [\n'
        '    {"from_domain": "source field", "to_domain": "target field", "how_they_bridge": "one sentence"}\n'
        "  ],\n"
        '  "audience_adaptation": "How they adjust communication for different audiences, if evidence exists.",\n'
        '  "what_they_never_do": "Communication anti-patterns they consistently avoid."\n'
        "}\n\n"
        "Be SPECIFIC. Use actual examples from the excerpts."
    )

    return system, user


def _build_persona_page_prompt(
    page_type: str,
    page_title: str,
    page_slug: str,
    bm25_query: str,
    other_page_slugs: list,
    analysis: dict,
    comm_analysis: dict,
    profile_chunks: list,
    person_name: str,
    person_slug: str,
) -> tuple:
    """
    Pass 3: Generate one wiki persona page using both analysis passes as context.

    Selects relevant profile chunks via BM25, builds a rich system+user prompt,
    and returns (system_prompt, user_prompt).
    """
    # Select relevant chunks for this specific page
    relevant_chunks = _bm25_select_chunks(profile_chunks, bm25_query, None, CHUNKS_PER_PAGE)
    if not relevant_chunks:
        relevant_chunks = profile_chunks[:CHUNKS_PER_PAGE]

    chunks_text = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c.get('content', '')[:600]}"
        for i, c in enumerate(relevant_chunks)
    )

    # Compact analysis context (not sending the full JSON, just the key facts)
    career_phases_str = ", ".join(
        f"{p['name']} ({p['years']})"
        for p in analysis.get("career_phases", [])
    )
    themes_str = "; ".join(
        f"{t['theme']} (e.g. {', '.join(t.get('key_examples', [])[:2])})"
        for t in analysis.get("core_intellectual_themes", [])
    )
    influences_str = ", ".join(
        i["name"] for i in analysis.get("key_influences", [])
    )
    signature_str = ", ".join(analysis.get("signature_examples", [])[:5])
    rhetorical_str = "; ".join(
        r["pattern"] for r in analysis.get("rhetorical_patterns", [])
    )
    opening_str = comm_analysis.get("opening_moves", "")
    scaffolding_str = comm_analysis.get("scaffolding_approach", "")
    sig_moves_str = "; ".join(
        m["move"] for m in comm_analysis.get("signature_rhetorical_moves", [])
    )

    analysis_context = (
        f"ANALYSIS CONTEXT (do not reproduce verbatim — use as evidence for synthesis):\n"
        f"- Career phases: {career_phases_str}\n"
        f"- Core themes: {themes_str}\n"
        f"- Signature examples: {signature_str}\n"
        f"- Rhetorical patterns: {rhetorical_str}\n"
        f"- Key influences: {influences_str}\n"
        f"- Communication opening moves: {opening_str}\n"
        f"- Scaffolding approach: {scaffolding_str}\n"
        f"- Signature rhetorical moves: {sig_moves_str}\n"
        f"- Cross-domain bridges: "
        + "; ".join(
            f"{b['from_domain']}→{b['to_domain']}"
            for b in comm_analysis.get("cross_domain_bridges", [])
        )
    )

    # Build YAML relationships block
    rel_lines = []
    for slug in other_page_slugs[:4]:
        if slug != page_slug:
            rel_lines.append(f"  - target: {slug}\n    type: related_to")
    if page_type != "main_persona" and person_slug != page_slug:
        rel_lines.append(f"  - target: {person_slug}\n    type: part_of")
    relationships_yaml = "\n".join(rel_lines) if rel_lines else "  []"

    # Page-type-specific instructions
    type_instructions = {
        "main_persona": (
            "This is the MAIN PERSONA PAGE — the intellectual DNA summary.\n"
            "Include:\n"
            "  - Core intellectual identity (2-3 paragraphs, specific and non-generic)\n"
            "  - 4-6 'Signature Intellectual Moves' as a bullet list, each with a concrete example\n"
            "  - Brief career arc paragraph\n"
            "  - Cross-reference links to other persona pages\n"
            "Target length: 400-500 words of body content."
        ),
        "thinking_patterns": (
            "This page captures REUSABLE INTELLECTUAL TOOLS.\n"
            "Include 4-6 named frameworks/lenses they apply. For each:\n"
            "  - What it is (one sentence)\n"
            "  - When/why they apply it\n"
            "  - A concrete example from their actual work\n"
            "Target length: 500-600 words."
        ),
        "intellectual_evolution": (
            "This page captures the EVOLUTION OF THEIR THINKING across career phases.\n"
            "Structure by phases (use the career_phases from analysis).\n"
            "For each phase: what shifted, what triggered the shift, what they abandoned, "
            "what they added.\n"
            "Target length: 400-500 words."
        ),
        "rhetorical_style": (
            "This page captures HOW THEY COMMUNICATE — their pedagogical DNA.\n"
            "Include: opening moves, scaffolding patterns, signature rhetorical devices "
            "(counterintuitive inversions, cross-domain analogies, concrete arithmetic), "
            "multi-level treatment (how they adapt for different audiences).\n"
            "Target length: 400-500 words."
        ),
        "domain": (
            "This page covers ONE SPECIFIC DOMAIN of their expertise.\n"
            "Include:\n"
            "  - Their distinctive take (what makes their view non-obvious in this domain)\n"
            "  - Key frameworks they apply here\n"
            "  - Concrete cases/examples they use\n"
            "  - How this domain connects to their broader intellectual project\n"
            "Target length: 400-500 words."
        ),
        "mentor_network": (
            "This page captures the PEOPLE AND WORKS that shaped their thinking.\n"
            "Include: key influences with specific ideas borrowed, intellectual lineage, "
            "how they develop ideas dialogically (with whom, in what settings).\n"
            "Target length: 350-400 words."
        ),
    }

    page_instruction = type_instructions.get(page_type, type_instructions["domain"])

    system = (
        f"You are writing a wiki knowledge base page for {person_name}'s personal knowledge system.\n\n"
        "This page is part of a tacit knowledge externalization — capturing implicit patterns "
        "in an expert's thinking so they can be consulted at query time.\n\n"
        "Rules:\n"
        "- Write in THIRD PERSON about the person, not as them\n"
        "- Be SPECIFIC: reference actual paper titles, examples, cases, and quotes from the source material\n"
        "- Avoid generic phrases ('they emphasize practical application') without concrete backing\n"
        "- Output ONLY raw markdown with YAML frontmatter — no preamble, no code fences, no commentary"
    )

    user = (
        f"Write a wiki page titled \"{page_title}\" for {person_name}.\n\n"
        f"{analysis_context}\n\n"
        f"INSTRUCTIONS FOR THIS PAGE:\n{page_instruction}\n\n"
        f"RELEVANT SOURCE EXCERPTS:\n{chunks_text}\n\n"
        f"Output exactly this format:\n\n"
        f"---\n"
        f"type: persona\n"
        f"aliases: [{page_title}]\n"
        f"relationships:\n"
        f"{relationships_yaml}\n"
        f"tags: [persona, {_sanitize_slug(analysis.get('primary_domain', 'knowledge'))}]\n"
        f"---\n\n"
        f"# {page_title}\n\n"
        f"[Body content — write here, following the instructions above]\n\n"
        f"---\n"
        f"*Part of the [[{person_slug}|{person_name}]] persona knowledge base*"
    )

    return system, user


def _build_stub_prompt(source_info: dict, first_chunks: list) -> tuple:
    """
    Build prompt for a single RAG stub page.
    Returns (system_prompt, user_prompt).
    """
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
        f"> \"Explain [main topic] from {name}\"\n"
        f"> \"What does {name} say about [concept]?\"\n\n"
        f"---\n"
        f"*RAG stub — {chunk_count} chunks indexed. Source: `{source_path}`*\n\n"
        f"Rules:\n"
        f"- Ground ALL claims in the provided excerpts\n"
        f"- Do not invent content not supported by the text\n"
        f"- Replace every placeholder line above with real content"
    )

    return system, user


def _build_index_prompt(inventory: dict, existing_index: str) -> tuple:
    """
    Build prompt to regenerate wiki/index.md from page inventory.
    Returns (system_prompt, user_prompt).
    """

    def _fmt_entries(pages: list) -> str:
        if not pages:
            return "<!-- None yet -->"
        lines = []
        for p in pages:
            # Build clean wikilink path (strip wiki/ prefix and .md)
            rel = p["path"].replace("\\", "/")
            if rel.startswith("wiki/"):
                rel = rel[5:]
            if rel.endswith(".md"):
                rel = rel[:-3]
            desc = f" — {p['description']}" if p.get("description") else ""
            lines.append(f"- [[{rel}|{p['title']}]]{desc}")
        return "\n".join(lines)

    inventory_text = (
        f"CURRENT WIKI PAGES:\n\n"
        f"Persona ({len(inventory['persona'])} pages):\n{_fmt_entries(inventory['persona'])}\n\n"
        f"Concepts ({len(inventory['concepts'])} pages):\n{_fmt_entries(inventory['concepts'])}\n\n"
        f"Entities ({len(inventory['entities'])} pages):\n{_fmt_entries(inventory['entities'])}\n\n"
        f"Stubs ({len(inventory['stubs'])} pages):\n{_fmt_entries(inventory['stubs'])}\n\n"
        f"Synthesized ({len(inventory['synthesized'])} pages):\n{_fmt_entries(inventory['synthesized'])}\n\n"
        f"Other ({len(inventory['other'])} pages):\n{_fmt_entries(inventory['other'])}"
    )

    system = (
        "You are rebuilding wiki/index.md — the master catalog of a personal knowledge base.\n"
        "The index is read before every query to find relevant pages.\n"
        "Output ONLY the raw markdown for index.md. No preamble. No commentary."
    )

    user = (
        f"Rebuild wiki/index.md using this exact page inventory.\n\n"
        f"{inventory_text}\n\n"
        f"EXISTING INDEX (preserve structure, notes, and any manually added content):\n"
        f"{existing_index[:2000] if existing_index else '[Does not exist yet]'}\n\n"
        f"Output a complete index.md with these sections in order:\n"
        f"1. # Wiki Index header + one-line description\n"
        f"2. ## Persona Page — the main persona page entry\n"
        f"3. ## Tacit Knowledge Pages (persona/) — thinking-patterns, evolution, rhetorical-style\n"
        f"4. ## Domain Pages (persona/) — domain-specific pages\n"
        f"5. ## Network and Philosophy Pages (persona/) — mentor network etc.\n"
        f"6. ## Extracted Concepts (N pages) — all concept pages\n"
        f"7. ## Extracted Entities (N pages) — all entity pages\n"
        f"8. ## Query-Synthesized Pages — synthesized/ pages (HTML comment if empty)\n"
        f"9. ## RAG Stubs — stub pages organized into ### Books, ### Research Papers, ### Profile\n\n"
        f"Format each entry as: - [[path/slug|Title]] — one-line description\n"
        f"Use accurate page counts in section headers.\n"
        f"Keep descriptions concise (under 15 words each).\n"
        f"If a section has no pages, include it with a <!-- comment --> placeholder."
    )

    return system, user


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

    # Cost estimate
    est_in = n * 3500
    est_out = n * 500
    print(f"  Sources to stub: {n}")
    print(f"  Estimated tokens: ~{est_in:,} input + ~{est_out:,} output")
    print(f"  Existing stubs will be {'OVERWRITTEN' if force else 'skipped (use --force to overwrite)'}\n")

    written = skipped = failed = 0

    def _process_stub(source_info: dict):
        slug = source_info["slug"]
        stub_path = STUBS_DIR / f"stub-{slug}.md"
        legacy_path = WIKI_DIR / f"stub-{slug}.md"   # old flat location

        if (stub_path.exists() or legacy_path.exists()) and not force:
            return "skipped", slug, None

        # Find chunks for this source (match by full path or stem)
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

    # Run in parallel (STUB_WORKERS threads)
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
                # Write a recoverable placeholder so the slot is tracked
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


def cmd_persona(
    llm: LLMClient,
    profile_arg: Optional[str] = None,
    force: bool = False,
) -> int:
    """
    Generate persona wiki pages using the three-pass tacit knowledge extraction.

    Pass 1 and Pass 2 analysis results are cached to data/ so individual pages
    can be regenerated with --force without repeating expensive analysis calls.
    """
    print("\n=== GENERATING PERSONA PAGES ===\n")

    PERSONA_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = _load_chunks()
    if not all_chunks:
        print("  ERROR: chunks.json is empty. Run: python scripts/ingest.py --process-all")
        return 0

    # Find profile chunks
    profile_filter = _get_profile_source_filter(profile_arg)
    profile_chunks = [
        c for c in all_chunks
        if profile_filter.lower() in c.get("source", "").lower().replace("\\", "/")
    ]

    if not profile_chunks:
        print(f"  ERROR: No chunks found for profile filter '{profile_filter}'")
        print(f"  Hint: Use --profile 'raw/profile/yourfile.md'")
        print(f"  Available sources: {sorted({c.get('source','') for c in all_chunks[:5]})}")
        return 0

    print(f"  Profile source: '{profile_filter}' ({len(profile_chunks)} chunks)")

    # ── PASS 1: Content Selection Analysis ───────────────────────────────────
    analysis: Optional[dict] = None

    if ANALYSIS_FILE.exists() and not force:
        try:
            analysis = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
            print(f"  [Pass 1] Using cached analysis → {ANALYSIS_FILE.name}")
        except (json.JSONDecodeError, OSError):
            analysis = None

    if analysis is None:
        print(f"  [Pass 1] Content selection analysis ({len(profile_chunks[:PROFILE_CHUNKS_FOR_ANALYSIS])} chunks)...")
        # Infer names for the prompt (will be corrected by LLM output)
        raw_name = profile_filter.replace("_knowledge_base", "").replace(".md", "")
        raw_name = raw_name.replace("_", " ").replace("-", " ").title()
        domain = "their field"

        system, user = _build_pass1_prompt(profile_chunks, raw_name, domain)
        raw_response = llm.call(system, user, MAX_TOKENS_ANALYSIS)
        analysis = _parse_json_response(raw_response, "Pass 1 analysis")

        if not analysis:
            print("  ERROR: Pass 1 JSON parsing failed. Aborting persona generation.")
            return 0

        ANALYSIS_FILE.write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"    ✓ Cached to {ANALYSIS_FILE.name}")

    person_name = analysis.get("person_name", "Unknown Person")
    person_slug = analysis.get("person_slug", "persona")
    domain_pages = analysis.get("domain_pages_needed", [])

    print(f"  Person: {person_name}")
    print(f"  Career phases: {len(analysis.get('career_phases', []))}")
    print(f"  Core themes: {len(analysis.get('core_intellectual_themes', []))}")
    print(f"  Domain pages to generate: {len(domain_pages)}\n")

    # ── PASS 2: Communication/Rhetorical Analysis ─────────────────────────────
    comm_analysis: dict = {}

    if COMM_ANALYSIS_FILE.exists() and not force:
        try:
            comm_analysis = json.loads(COMM_ANALYSIS_FILE.read_text(encoding="utf-8"))
            print(f"  [Pass 2] Using cached communication analysis → {COMM_ANALYSIS_FILE.name}")
        except (json.JSONDecodeError, OSError):
            comm_analysis = {}

    if not comm_analysis:
        print(f"  [Pass 2] Communication style analysis...")
        system, user = _build_pass2_prompt(profile_chunks, person_name, analysis)
        raw_comm = llm.call(system, user, MAX_TOKENS_ANALYSIS)
        comm_analysis = _parse_json_response(raw_comm, "Pass 2 communication analysis") or {}
        COMM_ANALYSIS_FILE.write_text(
            json.dumps(comm_analysis, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"    ✓ Cached to {COMM_ANALYSIS_FILE.name}")

    # ── PASS 3: Page Generation ───────────────────────────────────────────────
    print(f"\n  [Pass 3] Generating wiki pages...")

    # Determine all other slugs for cross-reference relationships
    domain_slugs = [dp["slug"] for dp in domain_pages[:3]]
    standard_slugs = [
        "thinking-patterns",
        "intellectual-evolution",
        "rhetorical-style-and-pedagogy",
        "mentor-network-and-influences",
    ] + domain_slugs

    # Define every page to generate
    pages: list = [
        {
            "slug": person_slug,
            "title": person_name,
            "type": "main_persona",
            "bm25_query": f"{person_name} biography career overview intellectual identity work",
            "other_slugs": standard_slugs,
        },
        {
            "slug": "thinking-patterns",
            "title": "Thinking Patterns and Frameworks",
            "type": "thinking_patterns",
            "bm25_query": "framework method approach lens think analyze argue mechanism principle",
            "other_slugs": [person_slug] + [s for s in standard_slugs if s != "thinking-patterns"],
        },
        {
            "slug": "intellectual-evolution",
            "title": "Intellectual Evolution",
            "type": "intellectual_evolution",
            "bm25_query": "early career phase transition changed focus evolution theory applied history",
            "other_slugs": [person_slug] + [s for s in standard_slugs if s != "intellectual-evolution"],
        },
        {
            "slug": "rhetorical-style-and-pedagogy",
            "title": "Rhetorical Style and Pedagogy",
            "type": "rhetorical_style",
            "bm25_query": "explain teach example analogy counterintuitive inversion illustrate students pedagogy",
            "other_slugs": [person_slug] + [s for s in standard_slugs if s != "rhetorical-style-and-pedagogy"],
        },
        {
            "slug": "mentor-network-and-influences",
            "title": "Mentor Network and Influences",
            "type": "mentor_network",
            "bm25_query": "influenced advisor colleague mentor inspired dialogic idea development collaboration",
            "other_slugs": [person_slug] + [s for s in standard_slugs if s != "mentor-network-and-influences"],
        },
    ]

    # Append domain pages from Pass 1 analysis
    for dp in domain_pages[:3]:
        pages.append({
            "slug": dp["slug"],
            "title": dp["title"],
            "type": "domain",
            "bm25_query": dp["title"] + " " + dp.get("description", ""),
            "other_slugs": [person_slug] + [
                s for s in standard_slugs if s != dp["slug"]
            ],
        })

    written = skipped = failed = 0

    for page_spec in pages:
        slug = page_spec["slug"]
        page_path = PERSONA_DIR / f"{slug}.md"
        title = page_spec["title"]

        if page_path.exists() and not force:
            print(f"  ─ {title} (exists)")
            skipped += 1
            continue

        print(f"  Generating: {title}...")
        try:
            system, user = _build_persona_page_prompt(
                page_type=page_spec["type"],
                page_title=title,
                page_slug=slug,
                bm25_query=page_spec["bm25_query"],
                other_page_slugs=page_spec["other_slugs"],
                analysis=analysis,
                comm_analysis=comm_analysis,
                profile_chunks=profile_chunks,
                person_name=person_name,
                person_slug=person_slug,
            )
            content = llm.call(system, user, MAX_TOKENS_PAGE)
            content = _strip_code_fences(content)
            _write_page_atomic(page_path, content + "\n", force)
            print(f"    ✓ {page_path.relative_to(VAULT)}")
            written += 1
            time.sleep(0.5)   # small pause between calls

        except Exception as exc:
            print(f"    ✗ Failed: {exc}")
            placeholder = (
                f"---\ntype: persona\naliases: [{title}]\nrelationships: []\n"
                f"tags: [persona, generation-failed]\n---\n\n"
                f"# {title}\n\n"
                f"<!-- GENERATION FAILED: {str(exc)[:200]} -->\n"
                f"<!-- Re-run: python scripts/auto_wiki_builder.py --persona --force -->\n"
            )
            try:
                _write_page_atomic(page_path, placeholder, force=True)
            except OSError:
                pass
            failed += 1

    print(f"\n  Persona — written: {written}, skipped: {skipped}, failed: {failed}")
    return written


def cmd_index(llm: LLMClient) -> bool:
    """Rebuild wiki/index.md as a complete, well-organized catalog."""
    print("\n=== REBUILDING INDEX.MD ===\n")

    inventory = _scan_wiki_inventory()
    total = sum(len(v) for v in inventory.values())

    print(f"  Scanned {total} wiki pages:")
    for cat, pages in inventory.items():
        if pages:
            print(f"    {cat}: {len(pages)}")

    existing_index = ""
    if INDEX_FILE.exists():
        try:
            existing_index = INDEX_FILE.read_text(encoding="utf-8")
        except OSError:
            pass

    print(f"\n  Calling LLM to generate index ({total} pages)...")
    system, user = _build_index_prompt(inventory, existing_index)

    try:
        content = llm.call(system, user, MAX_TOKENS_INDEX)
        content = _strip_code_fences(content)
        _write_page_atomic(INDEX_FILE, content + "\n", force=True)   # always overwrite
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
            "  python scripts/auto_wiki_builder.py --persona\n"
            "  python scripts/auto_wiki_builder.py --index\n"
            "  python scripts/auto_wiki_builder.py --all\n"
            "  python scripts/auto_wiki_builder.py --all --force\n"
            "  python scripts/auto_wiki_builder.py --all --model claude-sonnet-4-6\n"
            "  python scripts/auto_wiki_builder.py --all --provider openai --model gpt-4o\n"
            "  python scripts/auto_wiki_builder.py --persona --profile raw/profile/myfile.md\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stubs", action="store_true", help="Generate RAG stub pages")
    group.add_argument("--persona", action="store_true", help="Generate persona wiki pages (3-pass)")
    group.add_argument("--index", action="store_true", help="Rebuild wiki/index.md")
    group.add_argument("--all", action="store_true", help="Run stubs + persona + index")

    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pages (default: skip)")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=["anthropic", "openai"],
                        help=f"LLM provider (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--profile", type=str, default=None,
                        help="Profile source path/filename for persona extraction")
    args = parser.parse_args()

    # ── Load API key ──────────────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass   # dotenv optional; key must be in environment

    key_env = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_env, "")
    if not api_key:
        print(f"ERROR: {key_env} not set. Add it to .env or export it.")
        sys.exit(1)

    # ── Initialize LLM client ─────────────────────────────────────────────────
    try:
        llm = LLMClient(provider=args.provider, model=args.model, api_key=api_key)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR initializing LLM client: {exc}")
        sys.exit(1)

    print(f"Provider: {args.provider} | Model: {args.model}")
    if args.force:
        print("Mode: --force (existing pages WILL be overwritten)\n")

    start = time.time()
    stubs_written = persona_written = 0
    index_ok = False

    # ── Run requested commands ────────────────────────────────────────────────
    if args.stubs or args.all:
        stubs_written = cmd_stubs(llm, force=args.force)

    if args.persona or args.all:
        persona_written = cmd_persona(llm, profile_arg=args.profile, force=args.force)

    if args.index or args.all:
        index_ok = cmd_index(llm)

    # ── Rebuild knowledge graph ───────────────────────────────────────────────
    if stubs_written + persona_written > 0:
        print("\n=== REBUILDING KNOWLEDGE GRAPH ===\n")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from graph import save_graph
            graph = save_graph()
            n_nodes = len(graph.get("nodes", {}))
            n_edges = len(graph.get("edges", []))
            print(f"  ✓ Graph rebuilt: {n_nodes} nodes, {n_edges} edges")
        except ImportError:
            print("  [SKIP] graph.py not found — run: python scripts/graph.py --build")
        except Exception as exc:
            print(f"  ✗ Graph rebuild failed: {exc}")

    # ── Log and summarize ─────────────────────────────────────────────────────
    elapsed = time.time() - start

    _log(
        "auto_wiki_builder",
        "Automated wiki page generation complete",
        {
            "model": args.model,
            "stubs_written": stubs_written,
            "persona_written": persona_written,
            "index_rebuilt": index_ok,
            "elapsed_s": round(elapsed, 1),
        },
    )

    print(f"\n{'=' * 52}")
    print(f"  Completed in {elapsed:.1f}s")
    if args.stubs or args.all:
        print(f"  Stubs written:         {stubs_written}")
    if args.persona or args.all:
        print(f"  Persona pages written: {persona_written}")
    if args.index or args.all:
        print(f"  Index rebuilt:         {'✓' if index_ok else '✗'}")
    print(f"\n  Next steps:")
    print(f"    python scripts/export_for_web.py")
    print(f"    python scripts/sync_wiki.py --seed")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    main()
