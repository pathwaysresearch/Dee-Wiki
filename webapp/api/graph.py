"""
Knowledge graph builder and traversal for the wiki.

Parses YAML frontmatter from wiki pages, builds a typed graph index
(_graph.json), and provides traversal functions for the two-stage
LLM pipeline.

Usage:
    python scripts/graph.py --build          # Build _graph.json from wiki pages
    python scripts/graph.py --traverse NODE  # BFS from a node (default 2 hops)
    python scripts/graph.py --stats          # Show graph statistics

The graph doesn't replace the wiki pages — it augments them.
Pages are for reading. The graph is for navigation and discovery.
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import deque

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).parent.parent.parent  # webapp/api/ → webapp/ → project root
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
VAULT = PROJECT_ROOT / os.environ.get("WIKI_VAULT_NAME", "webapp/Vault")
WIKI_DIR = VAULT / "wiki"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH_PATH = DATA_DIR / "_graph.json"

# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(content):
    """Extract YAML frontmatter from a markdown file. Returns dict or {}."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return {}
    raw = m.group(1)
    if yaml:
        try:
            return yaml.safe_load(raw) or {}
        except yaml.YAMLError:
            return {}
    # Fallback: simple key-value parser for when PyYAML isn't installed
    return _simple_yaml_parse(raw)


def _simple_yaml_parse(raw):
    """Minimal YAML-like parser for frontmatter (no PyYAML dependency)."""
    result = {}
    current_key = None
    current_list = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key: value
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                # Inline list: [a, b, c]
                items = [x.strip().strip("'\"") for x in val[1:-1].split(",")]
                result[key] = [x for x in items if x]
            elif val:
                result[key] = val
            else:
                result[key] = []
                current_key = key
                current_list = result[key]
            continue

        # List item under current key
        if stripped.startswith("- ") and current_key is not None:
            item = stripped[2:].strip()
            if ":" in item and not item.startswith('"'):
                # Nested dict in list: - target: foo
                d = {}
                # Parse this and following indented lines as a dict
                k, _, v = item.partition(":")
                d[k.strip()] = v.strip()
                current_list.append(d)
            else:
                current_list.append(item)
            continue

        # Indented key under list item dict
        if stripped.startswith("type:") or stripped.startswith("target:"):
            if current_list and isinstance(current_list[-1], dict):
                k, _, v = stripped.partition(":")
                current_list[-1][k.strip()] = v.strip()

    return result


def strip_frontmatter(content):
    """Return content without YAML frontmatter."""
    m = FRONTMATTER_RE.match(content)
    if m:
        return content[m.end():]
    return content


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


def build_graph():
    """Scan all wiki pages, extract frontmatter, build knowledge graph."""
    nodes = {}
    edges = []

    if not WIKI_DIR.exists():
        return {"nodes": nodes, "edges": edges}

    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        # Skip special files
        if md_file.name.startswith("_"):
            continue

        stem = md_file.stem.lower()
        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)

        # Extract title
        title = stem.replace("-", " ").replace("_", " ").title()
        body = strip_frontmatter(content)
        for line in body.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        # Build node
        node_type = fm.get("type", "unknown")
        aliases = fm.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        nodes[stem] = {
            "type": node_type,
            "title": title,
            "aliases": aliases,
            "tags": tags,
            "path": str(md_file.relative_to(VAULT)),
        }

        # Build edges from relationships
        relationships = fm.get("relationships", [])
        if isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, dict):
                    target = rel.get("target", "").strip().lower()
                    edge_type = rel.get("type", "related_to")
                    if target:
                        edges.append({
                            "from": stem,
                            "to": target,
                            "type": edge_type,
                        })

        # Also extract [[wiki-links]] as implicit edges
        wikilinks = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", body)
        for link in wikilinks:
            link_stem = link.strip().lower()
            if link_stem != stem:  # no self-links
                # Only add if not already in explicit relationships
                existing = {(e["from"], e["to"]) for e in edges}
                if (stem, link_stem) not in existing:
                    edges.append({
                        "from": stem,
                        "to": link_stem,
                        "type": "references",
                    })

    graph = {"nodes": nodes, "edges": edges}
    return graph


def save_graph(graph=None):
    """Build (if needed) and save graph to _graph.json."""
    if graph is None:
        graph = build_graph()
    GRAPH_PATH.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return graph


def load_graph():
    """Load graph from _graph.json, or build if missing."""
    if GRAPH_PATH.exists():
        return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return save_graph()


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


def get_neighbors(graph, node, edge_types=None, direction="both"):
    """Get neighbors of a node, optionally filtered by edge type and direction."""
    neighbors = []
    for edge in graph["edges"]:
        if direction in ("both", "outgoing") and edge["from"] == node:
            if edge_types is None or edge["type"] in edge_types:
                neighbors.append({
                    "node": edge["to"],
                    "edge_type": edge["type"],
                    "direction": "outgoing",
                })
        if direction in ("both", "incoming") and edge["to"] == node:
            if edge_types is None or edge["type"] in edge_types:
                neighbors.append({
                    "node": edge["from"],
                    "edge_type": edge["type"],
                    "direction": "incoming",
                })
    return neighbors


def traverse(graph, start_node, hops=2, edge_types=None):
    """BFS traversal from start_node, up to N hops.
    Returns list of (node, hop_distance, edge_path) tuples."""
    start = start_node.lower()
    if start not in graph["nodes"]:
        return []

    visited = {start}
    queue = deque([(start, 0, [])])
    results = []

    while queue:
        node, depth, path = queue.popleft()
        if depth > 0:
            results.append({
                "node": node,
                "depth": depth,
                "path": path,
                "type": graph["nodes"].get(node, {}).get("type", "unknown"),
                "title": graph["nodes"].get(node, {}).get("title", node),
            })

        if depth >= hops:
            continue

        for neighbor in get_neighbors(graph, node, edge_types):
            n = neighbor["node"]
            if n not in visited and n in graph["nodes"]:
                visited.add(n)
                new_path = path + [{
                    "from": node, "to": n, "type": neighbor["edge_type"]
                }]
                queue.append((n, depth + 1, new_path))

    return results


def get_node_context(graph, node_stem):
    """Get a node's full context: its info + direct neighbors."""
    node = graph["nodes"].get(node_stem.lower())
    if not node:
        return None
    neighbors = get_neighbors(graph, node_stem.lower())
    return {
        "node": node,
        "neighbors": neighbors,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def graph_stats(graph):
    """Print graph statistics."""
    nodes = graph["nodes"]
    edges = graph["edges"]

    type_counts = {}
    for n in nodes.values():
        t = n.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    edge_type_counts = {}
    for e in edges:
        t = e.get("type", "unknown")
        edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

    # Find disconnected nodes
    connected = set()
    for e in edges:
        connected.add(e["from"])
        connected.add(e["to"])
    disconnected = [s for s in nodes if s not in connected]

    print(f"\n=== Knowledge Graph Stats ===")
    print(f"Nodes: {len(nodes)}")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"\nEdges: {len(edges)}")
    for t, c in sorted(edge_type_counts.items()):
        print(f"  {t}: {c}")
    if disconnected:
        print(f"\nDisconnected nodes ({len(disconnected)}):")
        for s in sorted(disconnected):
            print(f"  - {s}")
    else:
        print(f"\nAll nodes are connected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge graph tools")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true",
                       help="Build _graph.json from wiki pages")
    group.add_argument("--traverse", type=str, metavar="NODE",
                       help="BFS from a node")
    group.add_argument("--stats", action="store_true",
                       help="Show graph statistics")
    parser.add_argument("--hops", type=int, default=2,
                        help="Max hops for traversal (default: 2)")
    args = parser.parse_args()

    if args.build:
        graph = save_graph()
        print(f"Built graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        print(f"Saved to: {GRAPH_PATH}")
        graph_stats(graph)

    elif args.traverse:
        graph = load_graph()
        results = traverse(graph, args.traverse, hops=args.hops)
        if not results:
            print(f"Node '{args.traverse}' not found or has no connections.")
        else:
            print(f"\nTraversal from '{args.traverse}' ({args.hops} hops):\n")
            for r in results:
                indent = "  " * r["depth"]
                path_str = " → ".join(
                    f"{p['type']}" for p in r["path"]
                )
                print(f"{indent}[{r['type']}] {r['title']} (via {path_str})")

    elif args.stats:
        graph = load_graph()
        graph_stats(graph)
