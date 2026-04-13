"""
Extract concepts and entities from RAG chunks → wiki/concepts/ and wiki/entities/.

Uses Gemini 2.5 Flash to analyze existing RAG chunks in batches and identify
key concepts (theories, models, methods) and entities (people, institutions,
instruments). Generates wiki pages with YAML frontmatter and typed relationships.

Usage:
    python scripts/extract_entities.py --source "Valuation-Damodaran"
    python scripts/extract_entities.py --all
    python scripts/extract_entities.py --list-sources

Requires GEMINI_API_KEY in .env
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
VAULT = PROJECT_ROOT / "Vault"
WIKI_DIR = VAULT / "wiki"
CONCEPTS_DIR = WIKI_DIR / "concepts"
ENTITIES_DIR = WIKI_DIR / "entities"
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"

GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent?key={key}"
)

# Map verbose LLM relationship types to canonical types
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


def _normalize_rel_type(rel_type):
    """Normalize verbose relationship types to canonical graph types."""
    rel_type = rel_type.strip().lower()
    # Check direct map
    if rel_type in RELATIONSHIP_TYPE_MAP:
        return RELATIONSHIP_TYPE_MAP[rel_type]
    # Check if it's already a canonical type
    canonical = {"uses", "depends_on", "contrasts_with", "extends", "sourced_from",
                 "contradicts", "supersedes", "instance_of", "created_by", "created",
                 "related_to", "part_of", "contains", "used_by", "references"}
    cleaned = rel_type.replace(" ", "_")
    if cleaned in canonical:
        return cleaned
    # Default: use as-is but snake_case it
    return re.sub(r'\s+', '_', rel_type)


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


def load_chunks_by_source(source_filter=None):
    """Load RAG chunks, optionally filtered by source path substring."""
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found. Run ingest.py first.")
        sys.exit(1)

    all_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if source_filter:
        filtered = [c for c in all_chunks
                    if isinstance(c, dict) and source_filter.lower() in c.get("source", "").lower()]
        return filtered

    return [c for c in all_chunks if isinstance(c, dict) and "content" in c]


def list_sources():
    """List unique sources in chunks.json."""
    chunks = load_chunks_by_source()
    sources = {}
    for c in chunks:
        src = c.get("source", "unknown")
        name = Path(src).stem if src else "unknown"
        sources[name] = sources.get(name, 0) + 1

    print(f"\n{'Source':<60} {'Chunks':>8}")
    print("-" * 70)
    for name, count in sorted(sources.items()):
        print(f"{name:<60} {count:>8}")
    print(f"\nTotal: {len(sources)} sources, {sum(sources.values())} chunks")


def _fix_json(text):
    """Try to fix common JSON issues from LLM output."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Replace newlines inside JSON strings (common Gemini issue)
    # This is a heuristic: replace literal newlines between quotes
    fixed = re.sub(r'(?<=": ")(.*?)(?="[,\}\]])',
                   lambda m: m.group(0).replace('\n', ' ').replace('\r', ''),
                   text, flags=re.DOTALL)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract the JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def call_gemini(prompt, api_key):
    """Call Gemini API for extraction."""
    import requests
    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL, key=api_key)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            result = resp.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _fix_json(text)
            if parsed is not None:
                return parsed
            raise ValueError(f"Could not parse JSON from response ({len(text)} chars)")
        except Exception as exc:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/3 in {wait}s: {exc}")
                time.sleep(wait)
            else:
                print(f"  ERROR: Gemini call failed after 3 attempts: {exc}")
                return {"concepts": [], "entities": []}


def extract_from_source(source_name, chunks, api_key, batch_size=8):
    """Extract concepts and entities from a source's chunks."""
    all_concepts = {}  # slug → concept dict
    all_entities = {}  # slug → entity dict

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"  Processing {len(chunks)} chunks in {total_batches} batches...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        # Combine chunk texts
        chunks_text = "\n\n---\n\n".join(
            f"[Chunk {c.get('chunk_index', '?')}]\n{c['content'][:1500]}"
            for c in batch
        )

        prompt = EXTRACTION_PROMPT.format(
            source_name=source_name,
            chunks_text=chunks_text,
        )

        print(f"  Batch {batch_num}/{total_batches}...", end=" ", flush=True)
        result = call_gemini(prompt, api_key)

        n_concepts = len(result.get("concepts", []))
        n_entities = len(result.get("entities", []))
        print(f"{n_concepts} concepts, {n_entities} entities")

        # Merge concepts
        for c in result.get("concepts", []):
            slug = c.get("slug", "").strip().replace("_", "-").lower()
            c["slug"] = slug
            # Normalize relationship targets and types
            for rel in c.get("relationships", []):
                target = rel.get("target", rel.get("target_slug", "")).strip().replace("_", "-").lower()
                rel["target"] = target
                rel.pop("target_slug", None)
                rel["type"] = _normalize_rel_type(rel.get("type", "related_to"))
            if not slug or not c.get("name") or not c.get("description"):
                continue
            if slug in all_concepts:
                # Merge: append description if different, merge relationships
                existing = all_concepts[slug]
                if c["description"] not in existing["description"]:
                    existing["description"] += " " + c["description"]
                existing_targets = {(r["target"], r["type"])
                                    for r in existing.get("relationships", [])}
                for rel in c.get("relationships", []):
                    if (rel["target"], rel["type"]) not in existing_targets:
                        existing.setdefault("relationships", []).append(rel)
                for tag in c.get("tags", []):
                    if tag not in existing.get("tags", []):
                        existing.setdefault("tags", []).append(tag)
            else:
                all_concepts[slug] = c

        # Merge entities
        for e in result.get("entities", []):
            slug = e.get("slug", "").strip().replace("_", "-").lower()
            e["slug"] = slug
            for rel in e.get("relationships", []):
                target = rel.get("target", rel.get("target_slug", "")).strip().replace("_", "-").lower()
                rel["target"] = target
                rel.pop("target_slug", None)
                rel["type"] = _normalize_rel_type(rel.get("type", "related_to"))
            if not slug or not e.get("name") or not e.get("description"):
                continue
            if slug in all_entities:
                existing = all_entities[slug]
                if e["description"] not in existing["description"]:
                    existing["description"] += " " + e["description"]
                existing_targets = {(r["target"], r["type"])
                                    for r in existing.get("relationships", [])}
                for rel in e.get("relationships", []):
                    if (rel["target"], rel["type"]) not in existing_targets:
                        existing.setdefault("relationships", []).append(rel)
                for tag in e.get("tags", []):
                    if tag not in existing.get("tags", []):
                        existing.setdefault("tags", []).append(tag)
            else:
                all_entities[slug] = e

        # Rate limit — Gemini 2.5 Flash has generous limits but be polite
        if batch_num < total_batches:
            time.sleep(2)

    return all_concepts, all_entities


def generate_wiki_page(item, item_type, source_name):
    """Generate a wiki markdown page with YAML frontmatter."""
    slug = item["slug"]
    name = item["name"]
    desc = item["description"]
    relationships = item.get("relationships", [])
    tags = item.get("tags", [])

    # Build YAML frontmatter
    lines = [
        "---",
        f"type: {item_type}",
        f"aliases: [{name}]",
    ]

    if relationships:
        lines.append("relationships:")
        for rel in relationships:
            lines.append(f"  - target: {rel['target']}")
            lines.append(f"    type: {rel['type']}")

    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")

    lines.append(f"sourced_from: {source_name}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append(desc)
    lines.append("")

    # Add cross-references section
    if relationships:
        lines.append("## Relationships")
        lines.append("")
        for rel in relationships:
            target_name = rel["target"].replace("-", " ").title()
            lines.append(f"- **{rel['type']}**: [[{rel['target']}|{target_name}]]")
        lines.append("")

    lines.append(f"---\n*Extracted from: {source_name}*")

    return "\n".join(lines)


def write_pages(concepts, entities, source_name):
    """Write concept and entity pages to wiki folders."""
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    written = {"concepts": [], "entities": []}

    for slug, item in concepts.items():
        path = CONCEPTS_DIR / f"{slug}.md"
        if path.exists():
            # Don't overwrite — append source info if from a new source
            existing = path.read_text(encoding="utf-8")
            if source_name not in existing:
                existing += f"\n\n---\n*Also referenced in: {source_name}*"
                path.write_text(existing, encoding="utf-8")
                written["concepts"].append(f"{slug} (updated)")
        else:
            content = generate_wiki_page(item, "concept", source_name)
            path.write_text(content, encoding="utf-8")
            written["concepts"].append(slug)

    for slug, item in entities.items():
        path = ENTITIES_DIR / f"{slug}.md"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if source_name not in existing:
                existing += f"\n\n---\n*Also referenced in: {source_name}*"
                path.write_text(existing, encoding="utf-8")
                written["entities"].append(f"{slug} (updated)")
        else:
            content = generate_wiki_page(item, "entity", source_name)
            path.write_text(content, encoding="utf-8")
            written["entities"].append(slug)

    return written


def extract_source(source_filter, api_key):
    """Full extraction pipeline for one source."""
    chunks = load_chunks_by_source(source_filter)
    if not chunks:
        print(f"No chunks found matching '{source_filter}'")
        return

    # Derive source name from the chunks
    src_path = chunks[0].get("source", source_filter)
    source_name = Path(src_path).stem.replace("_", " ").replace("-", " ").title()

    print(f"\n=== Extracting from: {source_name} ===")
    print(f"  Chunks: {len(chunks)}")

    concepts, entities = extract_from_source(source_name, chunks, api_key)

    print(f"\n  Extracted: {len(concepts)} concepts, {len(entities)} entities")

    if not concepts and not entities:
        print("  Nothing extracted.")
        return

    # Write wiki pages
    written = write_pages(concepts, entities, source_name)

    print(f"\n  Written:")
    print(f"    Concepts: {len(written['concepts'])} → wiki/concepts/")
    for c in written["concepts"]:
        print(f"      - {c}")
    print(f"    Entities: {len(written['entities'])} → wiki/entities/")
    for e in written["entities"]:
        print(f"      - {e}")

    # Rebuild graph
    sys.path.insert(0, str(Path(__file__).parent))
    from graph import save_graph
    graph = save_graph()
    print(f"\n  Graph rebuilt: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    # Log ingest (single entry per source)
    pages_created = written["concepts"] + written["entities"]
    log_to_wiki_log(
        "ingest",
        source_name,
        {
            "route": "RAG+stub",
            "pages_created": pages_created,
            "chunks": len(chunks)
        }
    )

    return written


from wiki_logger import log_to_wiki_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Extract concepts and entities from RAG chunks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str,
                       help="Source name filter (substring match on chunk source paths)")
    group.add_argument("--all", action="store_true",
                       help="Extract from all sources")
    group.add_argument("--list-sources", action="store_true",
                       help="List available sources and chunk counts")
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
        # Get unique source stems
        chunks = load_chunks_by_source()
        source_stems = set()
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
