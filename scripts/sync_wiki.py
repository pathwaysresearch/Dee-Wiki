"""
Sync wiki pages between local Obsidian vault and Upstash Redis.

Usage:
    python scripts/sync_wiki.py --push         # Upload local wiki → Redis
    python scripts/sync_wiki.py --pull         # Download new + updated pages from Redis
    python scripts/sync_wiki.py --status       # Show diff between local and remote
    python scripts/sync_wiki.py --seed         # Initial seed: push local wiki + embeddings
    python scripts/sync_wiki.py --lint         # Run wiki health checks
    python scripts/sync_wiki.py --sync-and-pr  # Pull from Redis + lint + create GitHub PR
    python scripts/sync_wiki.py --prune-remote # Remove junk/duplicate pages from Redis

Requires KV_REST_API_URL and KV_REST_API_TOKEN in .env
"""

import os
import re
import sys
import json
import hashlib
import argparse
import subprocess
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
from chunker import get_embeddings_batch

IST = timezone(timedelta(hours=5, minutes=30))

PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
VAULT = PROJECT_ROOT / os.environ.get("WIKI_VAULT_NAME", "webapp/Vault")
WIKI_DIR = VAULT / "wiki"
INDEX_FILE = WIKI_DIR / "index.md"
LOG_FILE = WIKI_DIR / "log.md"
DATA_DIR = PROJECT_ROOT / "data"
WEBAPP_DATA = PROJECT_ROOT / "webapp" / "data"

REDIS_KEY = "wiki_pages"
REDIS_LOG_KEY = "wiki_log_entries"


# ---------------------------------------------------------------------------
# Title / path normalization
# ---------------------------------------------------------------------------

def _normalize_title(title):
    """Normalize a title for comparison: lowercase, collapse hyphens/spaces,
    strip special characters and trailing punctuation."""
    t = title.lower().strip()
    # Replace hyphens with spaces then collapse whitespace
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    # Strip characters that shouldn't be in a title (JSON braces, quotes, etc.)
    t = re.sub(r'[{}"\[\]`]', "", t)
    # Strip trailing punctuation
    t = t.strip(" ?!.,;:'\"")
    return t


def _is_junk_title(title):
    """Detect titles that are clearly junk (leaked JSON, garbage, etc.)."""
    t = title.strip()
    # Starts with JSON characters
    if t.startswith("{") or t.startswith("["):
        return True
    # Contains JSON-like patterns
    if '"answer"' in t.lower() or '"content"' in t.lower():
        return True
    # Too short after cleanup
    cleaned = _normalize_title(t)
    if len(cleaned) < 3:
        return True
    return False


def _sanitize_slug(text):
    """Create a safe filesystem slug from a title string."""
    slug = text.lower().strip()
    # Replace spaces, underscores with hyphens
    slug = slug.replace(" ", "-").replace("_", "-")
    # Remove all characters that aren't alphanumeric or hyphens
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    # Truncate to reasonable length
    if len(slug) > 80:
        slug = slug[:80].rstrip("-")
    return slug


def _normalize_path(path_str):
    """Normalize a path string: always use forward slashes, ensure it's
    under wiki/ and has a .md extension."""
    # Replace backslashes with forward slashes
    p = path_str.replace("\\", "/")
    # Ensure it starts with wiki/
    if not p.startswith("wiki/"):
        # Extract just the filename part
        parts = PurePosixPath(p).parts
        filename = parts[-1] if parts else p
        p = f"wiki/{filename}"
    # Ensure .md extension
    if not p.endswith(".md"):
        p += ".md"
    # Sanitize the filename part
    parent = str(PurePosixPath(p).parent)
    stem = PurePosixPath(p).stem
    safe_stem = _sanitize_slug(stem)
    if not safe_stem:
        safe_stem = "untitled"
    return f"{parent}/{safe_stem}.md"


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def get_redis_config():
    url = os.environ.get("KV_REST_API_URL", "")
    token = os.environ.get("KV_REST_API_TOKEN", "")
    if not url or not token:
        print("ERROR: KV_REST_API_URL and KV_REST_API_TOKEN must be set in .env")
        sys.exit(1)
    return url.rstrip("/"), token


def redis_get(url, token, key):
    import requests
    resp = requests.get(
        f"{url}/get/{key}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json().get("result")
    if result is None:
        return []
    # Handle potentially double-encoded JSON
    parsed = result
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
    return [p for p in parsed if isinstance(p, dict) and "content" in p]


def redis_set(url, token, key, value):
    import requests
    resp = requests.post(
        f"{url}/set/{key}",
        headers={"Authorization": f"Bearer {token}"},
        json=value,
        timeout=10,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Local wiki helpers
# ---------------------------------------------------------------------------

def load_local_wiki():
    """Load all wiki pages from local Obsidian vault."""
    pages = []
    if not WIKI_DIR.exists():
        return pages
    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if len(content) < 50:
            continue
        title = md_file.stem.replace("-", " ").replace("_", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        pages.append({
            "title": title,
            "path": str(md_file.relative_to(VAULT)),
            "content": content,
            "type": "wiki",
        })
    return pages


def save_page_locally(page):
    """Write a wiki page to the local Obsidian vault."""
    path = page.get("path", "")
    if path:
        # Normalize the path from Redis (fix backslashes, special chars)
        path = _normalize_path(path)
    else:
        slug = _sanitize_slug(page["title"])
        path = f"wiki/{slug}.md"

    full_path = VAULT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(page["content"], encoding="utf-8")
    return full_path


def _content_hash(content):
    """Hash content for comparison (ignoring trailing whitespace)."""
    return hashlib.md5(content.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status():
    url, token = get_redis_config()
    local_pages = load_local_wiki()
    remote_pages = redis_get(url, token, REDIS_KEY)

    # Filter out junk pages from remote for display purposes
    junk_pages = [p for p in remote_pages if _is_junk_title(p.get("title", ""))]
    clean_remote = [p for p in remote_pages if not _is_junk_title(p.get("title", ""))]

    local_titles = {_normalize_title(p["title"]) for p in local_pages}
    remote_titles = {_normalize_title(p["title"]) for p in clean_remote}

    only_local = local_titles - remote_titles
    only_remote = remote_titles - local_titles
    shared = local_titles & remote_titles

    # Check for content differences in shared pages
    local_by_title = {_normalize_title(p["title"]): p for p in local_pages}
    remote_by_title = {_normalize_title(p["title"]): p for p in clean_remote}
    content_diff = []
    for t in shared:
        lh = _content_hash(local_by_title[t]["content"])
        rh = _content_hash(remote_by_title[t]["content"])
        if lh != rh:
            content_diff.append(t)

    # Detect duplicate titles on remote (after normalization)
    seen_normalized = {}
    duplicate_titles = []
    for p in remote_pages:
        norm = _normalize_title(p["title"])
        if norm in seen_normalized:
            duplicate_titles.append((p["title"], seen_normalized[norm]))
        else:
            seen_normalized[norm] = p["title"]

    print(f"\nLocal:  {len(local_pages)} pages")
    print(f"Remote: {len(remote_pages)} pages ({len(clean_remote)} clean, {len(junk_pages)} junk)")
    print(f"Shared: {len(shared)} pages")

    if only_local:
        print(f"\nOnly local ({len(only_local)}):")
        for t in sorted(only_local):
            print(f"  + {t}")

    if only_remote:
        print(f"\nOnly remote ({len(only_remote)}) — created by web conversations:")
        for t in sorted(only_remote):
            print(f"  * {t}")

    if content_diff:
        print(f"\nContent differs ({len(content_diff)}):")
        for t in sorted(content_diff):
            meta = remote_by_title[t]
            updated = meta.get("updated_at", "unknown")
            query = meta.get("source_query", "")
            extra = f" (updated: {updated})" if updated != "unknown" else ""
            if query:
                extra += f" query: \"{query[:60]}...\""
            print(f"  ~ {t}{extra}")

    if junk_pages:
        print(f"\nJunk pages on remote ({len(junk_pages)}) — leaked JSON or garbage:")
        for p in junk_pages:
            print(f"  ✗ {p['title'][:80]}")
        print("  → Run --prune-remote to clean these up")

    if duplicate_titles:
        print(f"\nDuplicate titles on remote ({len(duplicate_titles)}) — differ only by hyphens/punctuation:")
        for dup, original in duplicate_titles:
            print(f"  ≈ \"{dup}\" duplicates \"{original}\"")
        print("  → Run --prune-remote to deduplicate")

    if not only_local and not only_remote and not content_diff and not junk_pages and not duplicate_titles:
        print("\nLocal and remote are in sync.")


def cmd_push():
    url, token = get_redis_config()
    local_pages = load_local_wiki()
    remote_pages = redis_get(url, token, REDIS_KEY)

    # Merge: local pages override remote, but keep remote-only pages
    remote_by_title = {_normalize_title(p["title"]): p for p in remote_pages}
    local_by_title = {_normalize_title(p["title"]): p for p in local_pages}

    merged = list(local_pages)
    for norm_title, p in remote_by_title.items():
        if norm_title not in local_by_title:
            # Only keep non-junk remote-only pages
            if not _is_junk_title(p.get("title", "")):
                merged.append(p)

    redis_set(url, token, REDIS_KEY, merged)
    print(f"Pushed {len(local_pages)} local pages to Redis (total: {len(merged)})")


# ---------------------------------------------------------------------------
# Index.md and log helpers
# ---------------------------------------------------------------------------


def update_index_md(new_pages):
    """Append new query-synthesized pages to the ## Query-Synthesized Pages
    section of index.md.

    Each entry in new_pages should have 'title' and 'path' keys.
    """
    if not new_pages or not INDEX_FILE.exists():
        return

    content = INDEX_FILE.read_text(encoding="utf-8")
    section_header = "## Query-Synthesized Pages"

    if section_header not in content:
        print("  WARN: '## Query-Synthesized Pages' section not found in index.md")
        return

    # Remove the placeholder comment if still present
    placeholder = "<!-- Pages will be added as /api/chat-v2 generates new insights warranting wiki updates -->"
    content = content.replace(placeholder, "")

    # Find insertion point — right after the section description paragraph
    lines = content.split("\n")
    insert_idx = None
    in_section = False
    for i, line in enumerate(lines):
        if line.strip() == section_header:
            in_section = True
            continue
        if in_section:
            # Skip blank lines and description text after the header
            if line.strip() == "" or (line.strip() and not line.startswith("- ") and not line.startswith("##")):
                insert_idx = i + 1
                continue
            # We've hit either a list item or the next section
            if line.startswith("## "):
                insert_idx = i
                break
            if line.startswith("- "):
                # Find end of existing list
                insert_idx = i + 1
                continue
            insert_idx = i
            break

    if insert_idx is None:
        # Append at end of file
        insert_idx = len(lines)

    # Build entries, skip duplicates
    existing_content = content.lower()
    entries_to_add = []
    for page in new_pages:
        title = page["title"]
        path = page.get("path", "")
        # Derive slug from path or title
        if path:
            slug = Path(path).stem
        else:
            slug = _sanitize_slug(title)

        # Skip if already in index
        if f"[[{slug}]]" in existing_content or f"[[{slug.lower()}]]" in existing_content:
            continue

        # Read file to get first-line description
        desc = title
        full_path = VAULT / path if path else WIKI_DIR / f"{slug}.md"
        if full_path.exists():
            file_content = full_path.read_text(encoding="utf-8").strip()
            # Get first non-heading, non-empty line as description
            for fline in file_content.splitlines():
                stripped = fline.strip()
                if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
                    desc = stripped[:120]
                    if len(stripped) > 120:
                        desc += "…"
                    break

        entries_to_add.append(f"- [[{slug}]] — {desc}")

    if not entries_to_add:
        print("  index.md: no new entries to add.")
        return

    # Insert the new entries
    for entry in reversed(entries_to_add):
        lines.insert(insert_idx, entry)

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  index.md: added {len(entries_to_add)} query-synthesized page(s).")


def cmd_pull_log():
    """Pull log entries from Redis and append to local log.md, then clear Redis."""
    url, token = get_redis_config()

    # Read all entries from the Redis list
    import requests
    try:
        resp = requests.get(
            f"{url}/lrange/{REDIS_LOG_KEY}/0/-1",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        raw_entries = resp.json().get("result", [])
    except Exception as exc:
        print(f"ERROR: Failed to read Redis log: {exc}")
        return

    if not raw_entries:
        print("No log entries in Redis.")
        return

    # Parse entries (they may be JSON strings or dicts)
    entries = []
    for raw in raw_entries:
        if isinstance(raw, str):
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw, dict):
            entry = raw
        else:
            continue
        if "operation" in entry and "description" in entry:
            entries.append(entry)

    if not entries:
        print("No valid log entries in Redis.")
        return

    # Redis LPUSH adds newest first, so reverse to get chronological order
    entries.reverse()

    # Format and append to local log.md
    log_text = ""
    for entry in entries:
        ts = entry.get("timestamp_str", entry.get("timestamp_iso", "unknown"))
        op = entry.get("operation", "unknown")
        desc = entry.get("description", "")
        meta = entry.get("metadata", {})

        log_text += f"\n## [{ts}] {op} | {desc}\n"
        if meta:
            for key, val in meta.items():
                if isinstance(val, (list, dict)):
                    log_text += f"- {key}: {json.dumps(val)}\n"
                else:
                    log_text += f"- {key}: {val}\n"

    if LOG_FILE.exists():
        current = LOG_FILE.read_text(encoding="utf-8")
        updated = current + log_text
    else:
        updated = (
            "# Wiki Log\n\n"
            "Append-only chronological record of ingests, queries, and wiki updates.\n\n"
            f"---{log_text}"
        )

    LOG_FILE.write_text(updated, encoding="utf-8")
    print(f"Appended {len(entries)} log entries from Redis to local log.md.")

    # Clear the Redis list
    try:
        resp = requests.post(
            f"{url}/del/{REDIS_LOG_KEY}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        resp.raise_for_status()
        print("Cleared Redis log entries.")
    except Exception as exc:
        print(f"WARN: Failed to clear Redis log: {exc}")


# ---------------------------------------------------------------------------
# Pull command
# ---------------------------------------------------------------------------


def cmd_pull():
    """Pull new AND updated pages from Redis to local vault.
    Also updates index.md, rebuilds _graph.json, and pulls log entries."""
    url, token = get_redis_config()
    remote_pages = redis_get(url, token, REDIS_KEY)
    local_pages = load_local_wiki()
    local_by_title = {_normalize_title(p["title"]): p for p in local_pages}

    new_count = 0
    updated_count = 0
    skipped_junk = 0
    changes = []

    for page in remote_pages:
        # Skip junk pages
        if _is_junk_title(page.get("title", "")):
            skipped_junk += 1
            continue

        norm_title = _normalize_title(page["title"])
        if norm_title not in local_by_title:
            # New page
            path = save_page_locally(page)
            print(f"  New: {page['title']} → {path}")
            new_count += 1
            changes.append({"title": page["title"], "action": "new",
                            "path": str(path.relative_to(VAULT)) if path.is_relative_to(VAULT) else str(path)})
        else:
            # Check content difference
            local_hash = _content_hash(local_by_title[norm_title]["content"])
            remote_hash = _content_hash(page["content"])
            if local_hash != remote_hash:
                path = save_page_locally(page)
                print(f"  Updated: {page['title']} → {path}")
                updated_count += 1
                changes.append({"title": page["title"], "action": "updated",
                                "path": str(path.relative_to(VAULT)) if path.is_relative_to(VAULT) else str(path)})

    total = new_count + updated_count
    if total:
        print(f"\nPulled {new_count} new + {updated_count} updated page(s).")
    else:
        print("No changes to pull.")

    if skipped_junk:
        print(f"Skipped {skipped_junk} junk page(s). Run --prune-remote to clean Redis.")

    # BUG 2 fix: Update index.md with new query-synthesized pages
    new_pages = [c for c in changes if c["action"] == "new"]
    if new_pages:
        print("\n--- Updating index.md ---")
        update_index_md(new_pages)

    # BUG 3 fix: Rebuild knowledge graph
    if total:
        print("\n--- Rebuilding knowledge graph ---")
        from graph import save_graph
        graph = save_graph()
        print(f"  Graph rebuilt: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    # BUG 1 fix: Pull log entries from Redis
    print("\n--- Pulling log entries from Redis ---")
    cmd_pull_log()

    return changes


def cmd_seed():
    """Initial seed: push local wiki pages WITH embeddings to Redis."""
    url, token = get_redis_config()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    wiki_json_path = WEBAPP_DATA / "wiki_pages.json"
    if wiki_json_path.exists():
        pages = json.loads(wiki_json_path.read_text(encoding="utf-8"))
        print(f"Seeding from exported data: {len(pages)} pages")
    else:
        pages = load_local_wiki()
        print(f"Seeding from local wiki: {len(pages)} pages")
        if gemini_key and pages:
            print("Generating embeddings...")
            texts = [p["content"] for p in pages]
            embs = get_embeddings_batch(texts, gemini_key, batch_pause=0.05)
            for page, emb in zip(pages, embs):
                page["embedding"] = emb

    redis_set(url, token, REDIS_KEY, pages)
    print(f"Seeded {len(pages)} pages to Redis.")


def cmd_prune_remote():
    """Remove junk and deduplicate pages on Redis."""
    url, token = get_redis_config()
    remote_pages = redis_get(url, token, REDIS_KEY)
    original_count = len(remote_pages)

    # Step 1: Remove junk pages
    junk_removed = []
    clean_pages = []
    for p in remote_pages:
        if _is_junk_title(p.get("title", "")):
            junk_removed.append(p["title"])
        else:
            clean_pages.append(p)

    # Step 2: Deduplicate by normalized title (keep the one with more content
    # or the one updated more recently)
    seen = {}
    deduped = []
    duplicates_removed = []
    for p in clean_pages:
        norm = _normalize_title(p["title"])
        if norm in seen:
            existing = seen[norm]
            # Keep the one with more content, or newer updated_at
            existing_len = len(existing.get("content", ""))
            current_len = len(p.get("content", ""))
            if current_len > existing_len:
                duplicates_removed.append(existing["title"])
                # Replace in deduped list
                deduped = [x for x in deduped if _normalize_title(x["title"]) != norm]
                deduped.append(p)
                seen[norm] = p
            else:
                duplicates_removed.append(p["title"])
        else:
            seen[norm] = p
            deduped.append(p)

    # Step 3: Normalize paths in remaining pages
    for p in deduped:
        if "path" in p:
            p["path"] = _normalize_path(p["path"])

    if junk_removed:
        print(f"Removing {len(junk_removed)} junk page(s):")
        for t in junk_removed:
            print(f"  ✗ {t[:80]}")

    if duplicates_removed:
        print(f"\nRemoving {len(duplicates_removed)} duplicate(s):")
        for t in duplicates_removed:
            print(f"  ≈ {t}")

    if not junk_removed and not duplicates_removed:
        print("No junk or duplicates found on remote. Redis is clean!")
        return

    # Confirm before writing
    print(f"\nTotal: {original_count} → {len(deduped)} pages")
    response = input("Write cleaned data to Redis? [y/N] ").strip().lower()
    if response == "y":
        redis_set(url, token, REDIS_KEY, deduped)
        print(f"Done. Redis now has {len(deduped)} clean pages.")
    else:
        print("Aborted. No changes written.")


# ---------------------------------------------------------------------------
# Lint checks
# ---------------------------------------------------------------------------

def cmd_lint():
    """Run wiki health checks per CLAUDE.md periodic lint rules."""
    pages = load_local_wiki()
    if not pages:
        print("No wiki pages found.")
        return []

    # Build link graph
    all_stems = set()
    for md_file in WIKI_DIR.rglob("*.md"):
        all_stems.add(md_file.stem.lower())

    # Parse [[wiki-links]] from all pages
    link_pattern = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
    inbound_count = {}  # stem → count of pages linking TO it
    broken_links = []   # (source_title, broken_target)
    all_links = {}      # title → set of targets

    for page in pages:
        links = link_pattern.findall(page["content"])
        targets = set()
        for link in links:
            target_stem = link.strip().lower()
            targets.add(target_stem)
            # Check if target exists
            if target_stem not in all_stems:
                broken_links.append((page["title"], link.strip()))
            else:
                inbound_count[target_stem] = inbound_count.get(target_stem, 0) + 1
        all_links[page["title"]] = targets

    # Exclude index.md and log.md from orphan check
    skip_stems = {"index", "log"}

    results = []

    # Check 1: Orphan pages (fewer than 2 inbound links)
    orphans = []
    for page in pages:
        stem = Path(page["path"]).stem.lower()
        if stem in skip_stems:
            continue
        count = inbound_count.get(stem, 0)
        if count < 2:
            orphans.append((page["title"], count))

    if orphans:
        results.append("### Orphan Pages (fewer than 2 inbound links)")
        for title, count in sorted(orphans):
            results.append(f"- **{title}** ({count} inbound links)")

    # Check 2: Broken links
    if broken_links:
        results.append("\n### Broken Links (target page does not exist)")
        for source, target in sorted(set(broken_links)):
            results.append(f"- `[[{target}]]` in **{source}**")

    # Check 3: Stub promotion candidates (stubs with 3+ inbound links)
    stubs = []
    for page in pages:
        if "**Type**: RAG stub" in page["content"] or \
           page["path"].startswith("wiki/stub-"):
            stem = Path(page["path"]).stem.lower()
            count = inbound_count.get(stem, 0)
            if count >= 3:
                stubs.append((page["title"], count))

    if stubs:
        results.append("\n### Stub Promotion Candidates (3+ inbound links)")
        for title, count in sorted(stubs):
            results.append(
                f"- **{title}** ({count} links) — consider full wiki ingest")

    # Print results
    if results:
        print("\n=== Wiki Lint Report ===\n")
        for line in results:
            print(line)
    else:
        print("\nWiki lint: all checks passed.")

    return results


# ---------------------------------------------------------------------------
# Sync and PR
# ---------------------------------------------------------------------------

def cmd_sync_and_pr():
    """Pull from Redis, run lint, create a GitHub PR for review.

    Complete workflow: pull pages + log, update index, rebuild graph,
    run lint, commit, push, create PR.
    """
    # Step 1: Pull changes (includes index.md update, graph rebuild, log pull)
    print("=== Step 1: Pull changes from Redis ===")
    changes = cmd_pull()

    if not changes:
        print("\nNo changes from Redis. Checking lint anyway...")

    # Step 2: Run lint
    print("\n=== Step 2: Wiki lint checks ===")
    lint_results = cmd_lint()

    # Step 3: Check if there are any git changes to commit
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "Vault/wiki/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if not result.stdout.strip():
        print("\nNo wiki file changes to commit.")
        return

    # Step 4: Create branch
    timestamp = datetime.now(IST).strftime("%Y-%m-%d-%H%M%S")
    branch = f"wiki-sync/{timestamp}"
    print(f"\n=== Step 3: Creating branch {branch} ===")

    subprocess.run(["git", "checkout", "-b", branch], cwd=PROJECT_ROOT, check=True)

    # Step 5: Stage and commit
    subprocess.run(
        ["git", "add", "Vault/wiki/"],
        cwd=PROJECT_ROOT, check=True,
    )

    # Build commit message
    commit_lines = ["Wiki sync from Redis\n"]
    for ch in changes:
        commit_lines.append(f"- {ch['action'].capitalize()}: {ch['title']}")

    subprocess.run(
        ["git", "commit", "-m", "\n".join(commit_lines)],
        cwd=PROJECT_ROOT, check=True,
    )

    # Step 6: Push and create PR
    print(f"\n=== Step 4: Push and create PR ===")
    subprocess.run(
        ["git", "push", "-u", "origin", branch],
        cwd=PROJECT_ROOT, check=True,
    )

    # Build PR body
    body_lines = ["## Wiki Sync from Redis\n"]
    body_lines.append(
        "Pages updated by the live chatbot, pulled from Upstash Redis.\n")

    if changes:
        body_lines.append("### Changes\n")
        for ch in changes:
            meta = ""
            if ch.get("updated_at"):
                meta += f" (updated: {ch['updated_at']})"
            if ch.get("source_query"):
                meta += f" query: \"{ch['source_query'][:60]}\""
            body_lines.append(f"- **{ch['action'].upper()}**: {ch['title']}{meta}")

    if lint_results:
        body_lines.append("\n---\n")
        body_lines.append("## Lint Report\n")
        body_lines.extend(lint_results)

    body_lines.append("\n---\n")
    body_lines.append("**Reminder**: After merging, run `python scripts/export_for_web.py` "
                      "and `python scripts/sync_wiki.py --seed` to update the deployment.")

    pr_body = "\n".join(body_lines)

    subprocess.run(
        ["gh", "pr", "create",
         "--title", f"Wiki sync {timestamp}",
         "--body", pr_body,
         "--base", "main"],
        cwd=PROJECT_ROOT, check=True,
    )

    # Switch back to main
    subprocess.run(["git", "checkout", "main"], cwd=PROJECT_ROOT, check=True)

    print("\nDone! PR created. Review it on GitHub.")
    print("After merging, run:")
    print("  python scripts/export_for_web.py")
    print("  python scripts/sync_wiki.py --seed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Sync wiki with Upstash Redis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--push", action="store_true",
                       help="Push local wiki to Redis")
    group.add_argument("--pull", action="store_true",
                       help="Pull new + updated pages from Redis")
    group.add_argument("--pull-log", action="store_true",
                       help="Pull log entries from Redis to local log.md")
    group.add_argument("--status", action="store_true",
                       help="Show diff between local and remote")
    group.add_argument("--seed", action="store_true",
                       help="Initial seed with embeddings")
    group.add_argument("--lint", action="store_true",
                       help="Run wiki health checks")
    group.add_argument("--sync-and-pr", action="store_true",
                       help="Pull from Redis + lint + create GitHub PR")
    group.add_argument("--prune-remote", action="store_true",
                       help="Remove junk and duplicate pages from Redis")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.push:
        cmd_push()
    elif args.pull:
        cmd_pull()
    elif args.pull_log:
        cmd_pull_log()
    elif args.seed:
        cmd_seed()
    elif args.lint:
        cmd_lint()
    elif args.sync_and_pr:
        cmd_sync_and_pr()
    elif args.prune_remote:
        cmd_prune_remote()
