from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SENSITIVE = {
    "password": re.compile(r"\bpassword\b", re.IGNORECASE),
    "access_token": re.compile(r"\baccess[_-]?token\b", re.IGNORECASE),
    "raw_push_token": re.compile(r"\b(push[_-]?token|device[_-]?token)\b", re.IGNORECASE),
    "precise_location": re.compile(r"\bprecise[_-]?location\b", re.IGNORECASE),
    "microphone": re.compile(r"\bmicrophone\b", re.IGNORECASE),
    "camera": re.compile(r"\bcamera\b", re.IGNORECASE),
}


def scan(path: Path, allowed: set[str] | None = None) -> list[str]:
    allowed = allowed or set()
    findings: list[str] = []
    files = (
        [path]
        if path.is_file()
        else [
            item
            for item in path.rglob("*")
            if item.is_file() and "node_modules" not in item.parts and ".next" not in item.parts
        ]
    )
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in SENSITIVE.items():
            if name not in allowed and pattern.search(text):
                findings.append(f"{file}:{name}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed mobile privacy gate")
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow", action="append", default=[])
    args = parser.parse_args()
    findings = scan(args.path, set(args.allow))
    if findings:
        print("Privacy gate failed:")
        print("\n".join(findings))
        return 1
    print("Privacy gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
