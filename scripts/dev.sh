#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv first: python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Copied .env.example to .env"
fi

export PYTHONPATH="$PWD"
echo "Core API  http://localhost:8100"
echo "Runtime   http://localhost:8101"
echo "Console   cd apps/web && npm run dev  (http://localhost:3000)"
echo "Evals/tools/architecture need no LLM key. Playground runs do."

.venv/bin/python -m uvicorn services.core_api.app.main:app --port 8100 &
trap 'kill %1' EXIT
.venv/bin/python -m uvicorn services.agent_runtime.app.main:app --port 8101
