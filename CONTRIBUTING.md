# Contributing

This repository is meant to stay an honest, closed-loop prototype.

## Local checks

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy services packages --ignore-missing-imports
Set-Location apps/web
npm ci
npm run build
```

Refresh the public eval table only when the harness changes:

```powershell
.\.venv\Scripts\python.exe scripts\print_evals.py --write-snapshot
```

Academic tests: `pip install -e academic` then `pytest academic/tests`.

## Product contract

- Catalog agents stay these four slugs: `academic-writer`, `supervisor`,
  `code-data-specialist`, `retrieval-specialist`.
- Product foresight is a tool-outcome simulator. ToolWorld-v1 is a separate
  CPU research package.
- Do not add live-LLM routing scores unless they are actually measured.
- Hard eval suites are allowed to score lower than the 24-task core.
- Do not fine-tune or claim a locally trained reranker.

## Pull requests

Keep the change small enough to review. Update README only when behavior or
measured numbers change. Do not commit `.env`, `*.db`, cache directories, or
private notes that are not part of this prototype.
