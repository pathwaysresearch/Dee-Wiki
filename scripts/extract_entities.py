"""
Extract concepts and entities from RAG chunks → wiki/concepts/ and wiki/entities/.

Uses Gemini 2.5 Pro to analyze existing RAG chunks in batches and identify
key concepts (theories, models, methods) and entities (people, institutions,
instruments). Generates wiki pages with YAML frontmatter and typed relationships.

Tracks processed chunks in data/extracted.json (SHA-256 of content) so
re-runs only process new chunks — safe to run after adding new sources.

Usage:
    python scripts/extract_entities.py --source "Valuation-Damodaran"
    python scripts/extract_entities.py --all
    python scripts/extract_entities.py --list-sources

Requires GEMINI_API_KEY in .env
"""

import hashlib
import os
import sys
import json
import time
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
VAULT = PROJECT_ROOT / os.environ.get("WIKI_VAULT_NAME", "webapp/Vault")
WIKI_DIR = VAULT / "wiki"
CONCEPTS_DIR = WIKI_DIR / "concepts"
ENTITIES_DIR = WIKI_DIR / "entities"
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EXTRACTED_FILE = DATA_DIR / "extracted.json"   # tracks processed chunk hashes

sys.path.insert(0, str(Path(__file__).parent))


GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent?key={key}"
)

RELATIONSHIP_TYPE_MAP = {
    "is central to": "related_to",
    "has approach": "uses",
    "is a study area in": "part_of",
    "is studied in relation to": "related_to",
    "is a field where": "contains",
    "is a component of": "part_of",
    "is used in": "used_by",
    "is based on": "depends_on",
    "is related to": "related_to",
    "is an example of": "instance_of",
    "is a type of": "instance_of",
    "developed": "created",
    "proposed": "created",
    "introduced": "created",
    "is a prerequisite for": "depends_on",
    "has subtype": "contains",
}


# ---------------------------------------------------------------------------
# Chunk hash tracking
# ---------------------------------------------------------------------------

def chunk_hash(chunk: dict) -> str:
    """SHA-256 of chunk content — stable identity across runs."""
    return hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()


def load_extracted_hashes() -> set:
    """Load set of already-processed chunk hashes from data/extracted.json."""
    if not EXTRACTED_FILE.exists():
        return set()
    try:
        return set(json.loads(EXTRACTED_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def save_extracted_hashes(hashes: set):
    """Persist the full set of processed chunk hashes."""
    EXTRACTED_FILE.write_text(
        json.dumps(sorted(hashes), indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Relationship normalisation
# ---------------------------------------------------------------------------

def _normalize_rel_type(rel_type: str) -> str:
    rel_type = rel_type.strip().lower()
    if rel_type in RELATIONSHIP_TYPE_MAP:
        return RELATIONSHIP_TYPE_MAP[rel_type]
    canonical = {
        "uses", "depends_on", "contrasts_with", "extends", "sourced_from",
        "contradicts", "supersedes", "instance_of", "created_by", "created",
        "related_to", "part_of", "contains", "used_by", "references",
    }
    cleaned = rel_type.replace(" ", "_")
    if cleaned in canonical:
        return cleaned
    return re.sub(r"\s+", "_", rel_type)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert knowledge graph builder for a knowledge base.

Analyze the following text excerpts from "{source_name}" and extract key concepts and entities.

RULES:
- Extract ONLY concepts and entities that are substantively discussed (not just mentioned in passing)
- A **concept** is an idea, theory, model, method, principle, or technique (e.g., "Discounted Cash Flow", "CAPM", "Heteroscedasticity", "Hedging")
- An **entity** is a person, institution, instrument, regulation, or named thing (e.g., "William Sharpe", "NYSE", "Basel III", "Black-Scholes Model")
- For each, provide a clear 1-2 sentence description on a SINGLE LINE (no newlines inside strings). Based ONLY on what the text says
- Identify relationships between extracted items AND to previously known items
- Use kebab-case for slugs (e.g., "discounted-cash-flow", "william-sharpe")
- Keep ALL string values on a single line — no line breaks inside JSON strings

Respond with ONLY valid JSON in this exact format (no markdown, no commentary):
{{
  "concepts": [
    {{
      "slug": "discounted-cash-flow",
      "name": "Discounted Cash Flow",
      "description": "A valuation method that estimates...",
      "relationships": [
        {{"target": "capm", "type": "uses"}},
        {{"target": "wacc", "type": "depends_on"}}
      ],
      "tags": ["valuation", "corporate-finance"]
    }}
  ],
  "entities": [
    {{
      "slug": "william-sharpe",
      "name": "William Sharpe",
      "description": "Nobel laureate who developed...",
      "relationships": [
        {{"target": "capm", "type": "created"}}
      ],
      "tags": ["economist", "nobel-laureate"]
    }}
  ]
}}

TEXT EXCERPTS:
{chunks_text}
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chunks_by_source(source_filter=None) -> list:
    """Load RAG chunks, optionally filtered by source path substring."""
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found. Run ingest.py first.")
        sys.exit(1)

    all_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if source_filter:
        return [
            c for c in all_chunks
            if isinstance(c, dict) and source_filter.lower() in c.get("source", "").lower()
        ]

    return [c for c in all_chunks if isinstance(c, dict) and "content" in c]


def list_sources():
    """List unique sources in chunks.json with extracted status."""
    chunks = load_chunks_by_source()
    extracted_hashes = load_extracted_hashes()

    sources: dict[str, dict] = {}
    for c in chunks:
        src = c.get("source", "unknown")
        name = Path(src).stem if src else "unknown"
        if name not in sources:
            sources[name] = {"total": 0, "done": 0}
        sources[name]["total"] += 1
        if chunk_hash(c) in extracted_hashes:
            sources[name]["done"] += 1

    print(f"\n{'Source':<55} {'Total':>7} {'Done':>6} {'New':>6}")
    print("-" * 78)
    for name, counts in sorted(sources.items()):
        new = counts["total"] - counts["done"]
        print(f"{name:<55} {counts['total']:>7} {counts['done']:>6} {new:>6}")
    print(f"\nTotal: {len(sources)} sources, {sum(s['total'] for s in sources.values())} chunks")


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

def _fix_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(
        r'(?<=": ")(.*?)(?="[,\}\]])',
        lambda m: m.group(0).replace("\n", " ").replace("\r", ""),
        text,
        flags=re.DOTALL,
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, api_key: str) -> dict:
    import requests

    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL, key=api_key)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _fix_json(text)
            if parsed is not None:
                return parsed
            raise ValueError(f"Could not parse JSON ({len(text)} chars)")
        except Exception as exc:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/3 in {wait}s: {exc}")
                time.sleep(wait)
            else:
                print(f"  ERROR: Gemini failed after 3 attempts: {exc}")
                return {"concepts": [], "entities": []}


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_item(existing: dict, new: dict):
    """Merge a newly extracted item into an existing one in-place."""
    if new["description"] not in existing["description"]:
        existing["description"] += " " + new["description"]

    existing_rels = {(r["target"], r["type"]) for r in existing.get("relationships", [])}
    for rel in new.get("relationships", []):
        if (rel["target"], rel["type"]) not in existing_rels:
            existing.setdefault("relationships", []).append(rel)

    existing_tags = set(existing.get("tags", []))
    for tag in new.get("tags", []):
        if tag not in existing_tags:
            existing.setdefault("tags", []).append(tag)


def _normalise_item(item: dict):
    """Normalise slug and relationship fields in-place. Returns slug or None."""
    slug = item.get("slug", "").strip().replace("_", "-").lower()
    item["slug"] = slug
    for rel in item.get("relationships", []):
        target = rel.get("target", rel.get("target_slug", "")).strip().replace("_", "-").lower()
        rel["target"] = target
        rel.pop("target_slug", None)
        rel["type"] = _normalize_rel_type(rel.get("type", "related_to"))
    if not slug or not item.get("name") or not item.get("description"):
        return None
    return slug


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_from_source(
    source_name: str,
    chunks: list,
    api_key: str,
    batch_size: int = 8,
) -> tuple[dict, dict]:
    """Extract concepts and entities from a source's NEW chunks only."""

    extracted_hashes = load_extracted_hashes()
    new_chunks = [c for c in chunks if chunk_hash(c) not in extracted_hashes]

    if not new_chunks:
        print(f"  All {len(chunks)} chunks already extracted. Skipping.")
        return {}, {}

    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  Skipping {skipped} already-processed chunks.")
    print(f"  Processing {len(new_chunks)} new chunks in "
          f"{(len(new_chunks) + batch_size - 1) // batch_size} batches...")

    all_concepts: dict = {}
    all_entities: dict = {}
    processed_hashes: set = set()

    total_batches = (len(new_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        chunks_text = "\n\n---\n\n".join(
            f"[Chunk {c.get('chunk_index', '?')}]\n{c['content'][:1500]}"
            for c in batch
        )
        prompt = EXTRACTION_PROMPT.format(
            source_name=source_name, chunks_text=chunks_text
        )

        print(f"  Batch {batch_num}/{total_batches}...", end=" ", flush=True)
        result = call_gemini(prompt, api_key)

        n_c = len(result.get("concepts", []))
        n_e = len(result.get("entities", []))
        print(f"{n_c} concepts, {n_e} entities")

        for item in result.get("concepts", []):
            slug = _normalise_item(item)
            if not slug:
                continue
            if slug in all_concepts:
                _merge_item(all_concepts[slug], item)
            else:
                all_concepts[slug] = item

        for item in result.get("entities", []):
            slug = _normalise_item(item)
            if not slug:
                continue
            if slug in all_entities:
                _merge_item(all_entities[slug], item)
            else:
                all_entities[slug] = item

        # Mark this batch's chunks as processed
        processed_hashes.update(chunk_hash(c) for c in batch)

        if batch_num < total_batches:
            time.sleep(2)

    # Persist hashes only after all batches succeed
    save_extracted_hashes(extracted_hashes | processed_hashes)
    print(f"  Marked {len(processed_hashes)} chunks as extracted → {EXTRACTED_FILE.name}")

    return all_concepts, all_entities


# ---------------------------------------------------------------------------
# Wiki page generation
# ---------------------------------------------------------------------------

def generate_wiki_page(item: dict, item_type: str, source_name: str) -> str:
    slug = item["slug"]
    name = item["name"]
    desc = item["description"]
    relationships = item.get("relationships", [])
    tags = item.get("tags", [])

    lines = ["---", f"type: {item_type}", f"aliases: [{name}]"]

    if relationships:
        lines.append("relationships:")
        for rel in relationships:
            lines.append(f"  - target: {rel['target']}")
            lines.append(f"    type: {rel['type']}")

    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")

    lines += [f"sourced_from: {source_name}", "---", "", f"# {name}", "", desc, ""]

    if relationships:
        lines.append("## Relationships")
        lines.append("")
        for rel in relationships:
            target_name = rel["target"].replace("-", " ").title()
            lines.append(f"- **{rel['type']}**: [[{rel['target']}|{target_name}]]")
        lines.append("")

    lines.append(f"---\n*Extracted from: {source_name}*")
    return "\n".join(lines)


def _merge_into_existing_page(path: Path, new_item: dict, source_name: str):
    """
    Merge new relationships and tags into an existing wiki page on disk.
    Replaces the old behaviour of just appending a reference line.
    """
    content = path.read_text(encoding="utf-8")

    # Parse existing frontmatter relationships so we don't duplicate
    existing_rels = set(re.findall(r"target:\s*(\S+)", content))

    additions = []
    for rel in new_item.get("relationships", []):
        if rel["target"] not in existing_rels:
            additions.append(f"  - target: {rel['target']}\n    type: {rel['type']}")

    if additions or source_name not in content:
        # Insert new relationships before the closing ---
        if additions:
            # Find relationships block or insert one before closing ---
            if "relationships:" in content:
                insert_after = content.index("relationships:") + len("relationships:")
                block = "\n" + "\n".join(additions)
                content = content[:insert_after] + block + content[insert_after:]
            else:
                # No relationships block yet — add one before closing ---
                close = content.index("\n---\n", content.index("---") + 3)
                block = "\nrelationships:\n" + "\n".join(additions)
                content = content[:close] + block + content[close:]

        if source_name not in content:
            content += f"\n\n---\n*Also referenced in: {source_name}*"

        path.write_text(content, encoding="utf-8")
        return True
    return False


def write_pages(concepts: dict, entities: dict, source_name: str) -> dict:
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    written = {"concepts": [], "entities": []}

    for slug, item in concepts.items():
        path = CONCEPTS_DIR / f"{slug}.md"
        if path.exists():
            if _merge_into_existing_page(path, item, source_name):
                written["concepts"].append(f"{slug} (updated)")
        else:
            path.write_text(generate_wiki_page(item, "concept", source_name), encoding="utf-8")
            written["concepts"].append(slug)

    for slug, item in entities.items():
        path = ENTITIES_DIR / f"{slug}.md"
        if path.exists():
            if _merge_into_existing_page(path, item, source_name):
                written["entities"].append(f"{slug} (updated)")
        else:
            path.write_text(generate_wiki_page(item, "entity", source_name), encoding="utf-8")
            written["entities"].append(slug)

    return written


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def extract_source(source_filter: str, api_key: str):
    chunks = load_chunks_by_source(source_filter)
    if not chunks:
        print(f"No chunks found matching '{source_filter}'")
        return

    src_path = chunks[0].get("source", source_filter)
    source_name = Path(src_path).stem.replace("_", " ").replace("-", " ").title()

    print(f"\n=== Extracting from: {source_name} ===")
    print(f"  Chunks total: {len(chunks)}")

    concepts, entities = extract_from_source(source_name, chunks, api_key)

    if not concepts and not entities:
        print("  Nothing new extracted.")
        return

    print(f"\n  Extracted: {len(concepts)} concepts, {len(entities)} entities")

    written = write_pages(concepts, entities, source_name)

    print(f"\n  Written:")
    print(f"    Concepts ({len(written['concepts'])}) → wiki/concepts/")
    for c in written["concepts"]:
        print(f"      - {c}")
    print(f"    Entities ({len(written['entities'])}) → wiki/entities/")
    for e in written["entities"]:
        print(f"      - {e}")

    from graph import update_graph
    graph = update_graph()
    print(f"\n  Graph updated: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Extract concepts and entities from RAG chunks"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str,
                       help="Source name filter (substring match on chunk source paths)")
    group.add_argument("--all", action="store_true",
                       help="Extract from all sources (skips already-processed chunks)")
    group.add_argument("--list-sources", action="store_true",
                       help="List sources with chunk counts and extraction status")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        sys.exit(0)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY must be set in .env")
        sys.exit(1)

    if args.source:
        extract_source(args.source, api_key)

    elif args.all:
        chunks = load_chunks_by_source()
        source_stems: set = set()
        for c in chunks:
            src = c.get("source", "")
            if src:
                source_stems.add(Path(src).stem)

        print(f"Extracting from {len(source_stems)} sources...")
        for stem in sorted(source_stems):
            try:
                extract_source(stem, api_key)
            except Exception as exc:
                print(f"  ERROR extracting {stem}: {exc}")
                continue