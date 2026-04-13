"""
Shared wiki logging utility for the Two-Tier LLM Wiki.

Single canonical implementation of log_to_wiki_log used by:
  - scripts/extract_entities.py
  - scripts/ingest.py
  - (index.py keeps its own inline copy for Vercel compatibility)

All timestamps in IST (UTC+5:30).
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

PROJECT_ROOT = Path(__file__).parent.parent
VAULT = PROJECT_ROOT / "Vault"
LOG_FILE = VAULT / "wiki" / "log.md"


def log_to_wiki_log(operation, description, metadata=None):
    """Append entry to wiki/log.md and optionally to Redis.

    Tries Redis first (if env vars set), then local filesystem.
    Always prints to stdout as fallback.

    Format: ## [YYYY-MM-DD HH:MM IST] operation | description
    """
    timestamp = datetime.now(IST)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M IST")
    log_entry_text = f"[{timestamp_str}] {operation} | {description}"

    if metadata:
        for key, val in metadata.items():
            if isinstance(val, (list, dict)):
                log_entry_text += f" | {key}: {json.dumps(val)}"
            else:
                log_entry_text += f" | {key}: {val}"

    # PRIMARY: Try Redis if env vars are set
    kv_url = os.environ.get("KV_REST_API_URL", "")
    kv_token = os.environ.get("KV_REST_API_TOKEN", "")

    if kv_url and kv_token:
        try:
            import requests
            log_json = {
                "timestamp_iso": timestamp.isoformat(),
                "timestamp_str": timestamp_str,
                "operation": operation,
                "description": description,
                "metadata": metadata or {}
            }
            resp = requests.post(
                f"{kv_url.rstrip('/')}/lpush/wiki_log_entries",
                headers={"Authorization": f"Bearer {kv_token}"},
                json=log_json,
                timeout=5
            )
            resp.raise_for_status()
            print(f"[Log] Redis stored: {operation}")
        except Exception as exc:
            print(f"[Log] Redis write failed: {exc}")

    # SECONDARY: Write to local log.md
    try:
        entry = f"\n## [{timestamp_str}] {operation} | {description}\n"
        if metadata:
            for key, val in metadata.items():
                if isinstance(val, (list, dict)):
                    entry += f"- {key}: {json.dumps(val)}\n"
                else:
                    entry += f"- {key}: {val}\n"

        if LOG_FILE.exists():
            current = LOG_FILE.read_text(encoding="utf-8")
            updated = current + entry
        else:
            updated = (
                "# Wiki Log\n\n"
                "Append-only chronological record of ingests, queries, and wiki updates.\n\n"
                f"---{entry}"
            )

        LOG_FILE.write_text(updated, encoding="utf-8")
        print(f"[Log] Local file written: {operation}")
    except Exception as exc:
        print(f"[Log] Local write failed: {exc}")

    # TERTIARY: stdout (always)
    print(f"[Log] {log_entry_text}")
