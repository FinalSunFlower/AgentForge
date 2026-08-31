"""Print measured eval numbers and optionally refresh data/evals_snapshot.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.core_api.app.routers.evals import SNAPSHOT_PATH, evals_summary_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print AgentForge eval numbers (no live LLM).")
    parser.add_argument(
        "--write-snapshot", action="store_true", help="Refresh data/evals_snapshot.json"
    )
    args = parser.parse_args()
    payload = {
        **evals_summary_payload(),
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "snapshot",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.write_snapshot:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {Path(SNAPSHOT_PATH).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
