# AgentForge

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00)](https://www.sqlalchemy.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Auditable **tool-augmented agent runtime** in Python, with a Next.js console.
One closed loop you can clone, start, and inspect: identity → ReAct → policy
gate → hybrid retrieval or a deterministic tool → grounding → SQL events →
SSE replay → usage. Measured evals are checked in. Method boundaries are
written down.

This is a prototype you can run. It is not a trained MoE, not a production
SaaS, and not a claim that a public URL is live.

A separate CPU-only research package lives in [`academic/`](academic/README.md).
On GitHub, the `academic` branch uses that write-up as the landing page.
Those numbers are not evidence that this runtime has a learned world model.

## Start here (no vendor key)

| You want to see | Open |
|---|---|
| Measured retrieval + routing + hard suites | [`/evals`](http://localhost:3000/evals) ← `GET /v1/evals/summary` |
| Approved tool registry | [`/tools`](http://localhost:3000/tools) |
| Process diagram + what is *not* implemented | [`/architecture`](http://localhost:3000/architecture) |
| Playground run (needs `LLM_API_KEY`) | [`/`](http://localhost:3000) after sign-in |

Without an LLM key the Runtime **fails closed**. Evals, tools, and architecture
still work. A playground run never fabricates an answer.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\scripts\dev.ps1
```

Second terminal: `cd apps/web && npm install && npm run dev`.
Open [http://localhost:3000/evals](http://localhost:3000/evals).

## Measured snapshot

Checked in at [`data/evals_snapshot.json`](data/evals_snapshot.json).
Regenerate with `python scripts/print_evals.py --write-snapshot`.
These are **keyword / MiniLM baselines on the checked-in corpus**, not
live-LLM routing accuracy. Hard suites are supposed to score lower.

| Suite | Metric | Score |
|---|---|---:|
| Retrieval, 19 queries, MiniLM MaxSim (production) | recall@k / MRR | 1.00 / 1.00 |
| Same 19 queries, MS MARCO MiniLM-L-6 ONNX CE (eval column) | recall@k / MRR | 1.00 / 0.96 |
| Keyword / BM25, 19 queries | recall@k | 0.84 |
| Zero-overlap MiniLM | recall@3 | 1.00 (keyword = 0.00) |
| Hard retrieval (paraphrase + distractors) | keyword / CE / MaxSim recall | 0.00 / 0.83 / 1.00 |
| Tool routing, 24-task core | keyword / MiniLM success | 1.00 / 1.00 |
| Hard routing (traps, ordered multi-step) | keyword / MiniLM success | 0.33 / 0.50 |
| Needle + structured memory | extractive facts | pass |

```text
JWT login → thread/run → streamed function calling → policy gate
  → hybrid retrieval or deterministic tool → grounding check
  → durable SQL events → SSE replay → usage / audit
```

The required business slice is a **novel reader**: published chapters,
bookshelves, and monotonic offline reading-progress sync. Feed, test billing,
simulated notifications, and scientific tools are small supporting demos.

## Tech stack

Versions are the constraints in [`pyproject.toml`](pyproject.toml) and
[`apps/web/package.json`](apps/web/package.json).

| Layer | Choice | Role |
|---|---|---|
| Language | Python 3.12+ | All backend processes |
| HTTP | FastAPI 0.115, Uvicorn, Pydantic v2 | Core API + Agent Runtime |
| Persistence | SQLAlchemy 2.0 async | Same models on SQLite (`aiosqlite`) or PostgreSQL (`asyncpg`) |
| Cache / streams | Redis 5 (optional) | Event-stream mirror, Demo Mode IP window |
| Embeddings | fastembed + all-MiniLM-L6-v2 ONNX; MS MARCO MiniLM-L-6 cross-encoder ONNX (CPU) | Semantic retrieval and pairwise rerank (not a generative LLM) |
| Identity | PyJWT, Argon2id (`argon2-cffi`) | Access tokens; login-time PBKDF2 migration |
| Observability | OpenTelemetry SDK + FastAPI/httpx instrumentation | Request / trace correlation headers |
| Console | Next.js 16 (App Router), React 19, TypeScript 5.8 | Playground, tools, evals, architecture |
| Quality | pytest, ruff, mypy, pytest-asyncio, pgserver | Local + CI |
| CI services | PostgreSQL 16, Redis 7 | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

There is no Go service, no pnpm/Turborepo workspace, and no generated OpenAPI
client. The console talks to Core API with `fetch` in
[`apps/web/app/lib/api.ts`](apps/web/app/lib/api.ts). Shared event types live in
[`packages/contracts`](packages/contracts).

## Features

| Capability | What is implemented | What it is not |
|---|---|---|
| Model adapter | OpenAI-compatible stream and non-stream turns | Vendor-specific SDK lock-in |
| ReAct loop | Incremental text, reasoning, tool-call parse; 8-step cap | Unlimited agent swarms |
| Hybrid RAG | BM25 + MiniLM + RRF + MiniLM MaxSim in production; feature rerank and MS MARCO MiniLM-L-6 ONNX cross-encoder are measured eval columns; `passage_id` citations | Locally trained reranker / `bge-reranker` fine-tune |
| Tool routing | Live `tool.routing`: keyword vs MiniLM, catalog reorder; `intent_router` MiniLM + keyword fallback | Learned policy / live-LLM tool selection |
| Scheduler | Batches independent reads; isolates writes and `tool_call_id` argument deps | Distributed workflow engine |
| Grounding | Lexical overlap on cited snippets | Abstractive summarization |
| Memory | Extractive compression of structured facts | Implicit-prose recall |
| Multi-agent | Four catalog agents; one-way sticky supervisor → specialist | Bidirectional handoff |
| Tool governance | Versioned registry, JSON Schema, risk, timeout, approval, audit. `POST /v1/admin/mcp/sync` discovers MCP tools into **quarantined** rows | Auto-approve or skip schema hash |
| Events | SQL event store, transactional outbox, optional Redis Stream; `tool.routing` and `tool.foresight` on live runs | ClickHouse analytics |
| Reconnect | SSE heartbeat + `Last-Event-ID`; SQL is authoritative | Exactly-once push |
| Identity | JWT, Argon2id, refresh/API-key paths, plan limits | MFA / SSO |
| Evals | 24-task keyword **and** MiniLM routers; hard trap/multi-step suite (scores may drop); 19-query retrieval including MaxSim and MS MARCO CE; harder paraphrase+distractor retrieval; needle; memory | Live-LLM routing accuracy |
| Foresight | Deterministic tool-outcome simulator (calculator AST, MiniLM retrieval preview, SQL allowlist, sonar closed form) | Academic RAP / tabular world models (see `academic/`) |
| Observability | OpenTelemetry SDK; console/OTLP opt-in | Hosted APM product |
| Demo Mode | Global `UsageDaily` token budget, per-IP window, high-risk tools stripped from the **model** catalog | Production rate-limit product |

Deterministic calculator / retrieval / science tools run without a GPU or a
vendor SDK. Without `LLM_API_KEY` + `LLM_MODEL` the Runtime **fails closed**;
it never fabricates an answer.

## Catalog agents

`GET /v1/agents` returns only these four slugs. Local pytest fixtures that share
`agentforge.local.db` (for example `usage-{uuid}`) are **not** product agents
and are filtered out.

| Console label | Slug | Role |
|---|---|---|
| Assistant | `default-assistant` | ReAct + built-in tools |
| Supervisor | `supervisor` | One-way handoff to a specialist |
| Science | `science-specialist` | Calculator, passive sonar, wind tunnel |
| Retrieval | `retrieval-specialist` | Hybrid search with citations |

## Built-in tools

| Name | Risk | What it does | Foresight preview |
|---|---|---|---|
| `calculator` | low | AST-allowlisted arithmetic | exact AST value |
| `retrieval` | medium | BM25 + MiniLM + RRF + MaxSim; `passage_id` snippets | MiniLM vector preview |
| `passive_sonar` | medium | 2D bearing least squares | closed-form source |
| `wind_tunnel` | medium | 2D inviscid cylinder potential flow | grid-size estimate |
| `readonly_sql` | high | Allowlisted `SELECT` / `WITH` | schema validate only |
| `intent_router` | low | Keyword counts + MiniLM intent prototypes | uninformative prior |
| `handoff` | low | One-way sticky transfer to a specialist | uninformative prior |

MCP tools appear only after `POST /v1/admin/mcp/sync` (quarantined), admin
approve, and attach to an agent. They are never auto-approved.

## Runtime events and probes

Durable events (`packages/contracts/events.py`) include `run.*`,
`message.delta`, `tool.requested` / `started` / `result`, `tool.routing`
(`kind: not_live_llm`), `tool.foresight` (`kind: tool_outcome_simulator`),
`agent.handoff`, `context.compressed`, and `usage.final`. Token deltas stay in
the transcript; Trace lists the durable stream.

| Probe | Meaning |
|---|---|
| `GET /healthz` | Process is up |
| `GET /ready` | Core API can read the database and reach Runtime; **503** if Runtime is down |
| `GET /v1/status` | Public booleans: `runtime`, `llm_configured`, `demo_mode`. No secrets |
| `GET /v1/evals/summary` | Checked-in snapshot by default; `?live=1` recomputes |

Fresh databases seed **Harbor Field Notes** from the 19-query eval corpus so
hybrid retrieval is not empty on first clone.

## Worker (optional)

`python -m services.worker.app.main` loops: outbox relay (needs Redis), retry
stuck `created` runs, usage rollup, pending-order expiry, simulated push
dispatch, thread summaries. The first local playground run does not need it.

Supporting HTTP slices (not the agent contract): novels/chapters/bookshelf/
monotonic reading progress; public feed + likes; idempotent checkout + webhook
dedup; notification facts and simulated push. These exist so the runtime sits
on a real product surface, not a toy chat stub.

## Architecture

![AgentForge runtime](figures/architecture.jpg)

```text
Next.js console  (apps/web, :3000)
        |
        v
 Core API  FastAPI :8100
 identity, threads, runs, novels, feed, checkout, evals summary
        |
        |  X-Runtime-Token  (RUNTIME_INTERNAL_TOKEN)
        v
 Agent Runtime  FastAPI :8101
 provider, ReAct, MiniLM RAG, MaxSim rerank, ONNX CE eval column, MCP catalog, tool-outcome simulator, DAG scheduler
        |
        +----------------------+
        |                      |
        v                      v
 SQLite or PostgreSQL      Redis (optional)
 durable source            stream mirror / IP window
        ^
        |
     Worker
 outbox relay, usage rollup, retry dispatch, simulated delivery
```

The Core API POSTs a new run to the Runtime immediately. The Worker retries
stuck `created` runs and is **not** required for the first local demo.

Local profile: async SQLite, no Docker (`scripts/dev.ps1`). Optional Postgres +
Redis: `docker compose up`. Optional API containers: `docker compose --profile stack up --build`.
Production profile: the same SQLAlchemy models on PostgreSQL + Redis
(`DATABASE_URL`, `REDIS_URL`).

Supporting demos (not the main product contract): public feed, test checkout
with webhook dedup, in-app notification facts, scientific tools, MCP JSON-RPC
client as an adapter boundary.

## Repository layout

```text
pyproject.toml                     Python package + ruff/pytest
.env.example                       keys the Settings classes actually read
docker-compose.yml                 optional Postgres 16 + Redis 7 (same as CI)
services/core_api/                 identity, novels, events, outbox, Demo Mode
services/agent_runtime/app/
  executor.py                      ReAct loop, routing emit, simulator, grounding, usage
  hybrid_retrieval.py              BM25 + MiniLM + RRF + late-interaction MaxSim
  cross_encoder.py                 off-the-shelf MS MARCO MiniLM-L-6 ONNX (eval column)
  embedding_router.py              MiniLM cosine tool/specialist/intent routing
  science_tools.py                 sonar, wind tunnel, SQL allowlist, intent_router
  mcp.py                           MCP JSON-RPC client + catalog wrappers
  foresight.py                     tool-outcome simulator: AST / vector preview / SQL validate (not RAP)
  grounding.py                     citation overlap check
  memory.py                        extractive long-context compression
  handoff.py                       supervisor specialist specs
  scheduler.py                     dependency-aware tool batches
  policy.py                        schema, depth, high-risk approval gate
  eval_harness.py                  keyword vs MiniLM routers, hard suites, retrieval, needle
  tools.py / provider.py
services/worker/app/               outbox, usage, dispatch retry, simulated delivery
packages/contracts/                EventEnvelope and event types
apps/web/                          Next.js console: /, /tools, /evals, /architecture
apps/mobile_contract/              fail-closed privacy scan CLI (tests + `privacy_gate.py`)
academic/                          research package — see academic/README.md
data/evals_snapshot.json           checked-in eval numbers served by /evals
figures/architecture.jpg           runtime architecture diagram
scripts/dev.ps1, scripts/dev.sh    local Core API + Runtime
scripts/print_evals.py             refresh the eval snapshot
scripts/smoke_test_deployed.py     black-box check against SMOKE_BASE_URL
tests/                             product tests
CONTRIBUTING.md / SECURITY.md      review and secret-handling notes
```

## Quick start

Requires **Python 3.12+**. **Node.js 20.9+** is needed only for the console.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

The first live retrieval or `?live=1` eval run downloads ~80 MB all-MiniLM-L6-v2
ONNX plus the quantized MS MARCO MiniLM-L-6 CE into `~/.cache/fastembed`.
The public `/evals` page serves the checked-in snapshot and does not wait on
that download.

From the repository root, start Core API and Runtime:

```powershell
.\scripts\dev.ps1
```

```bash
chmod +x scripts/dev.sh && ./scripts/dev.sh
```

Or two terminals:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn services.core_api.app.main:app --port 8100
.\.venv\Scripts\python.exe -m uvicorn services.agent_runtime.app.main:app --port 8101
```

```bash
export PYTHONPATH=.
uvicorn services.core_api.app.main:app --port 8100
uvicorn services.agent_runtime.app.main:app --port 8101
```

Optional Worker (outbox, usage, retry dispatch):

```powershell
.\.venv\Scripts\python.exe -m services.worker.app.main
```

For a real model turn set `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` in
`.env`. OpenAPI: [http://localhost:8100/docs](http://localhost:8100/docs).

Console:

```powershell
Set-Location apps/web
npm install
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:8100"
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

| Route | Auth | Contents |
|---|---|---|
| `/` | Sign-in only to start a run | Playground: four suggestion cards, agent dropdown, composer, optional Trace drawer |
| `/tools` | Public | Approved registry rows. High-risk tools stay listed here; Demo Mode hides them from the **model-visible** catalog |
| `/evals` | Public | `GET /v1/evals/summary` — MiniLM, MaxSim, MS MARCO CE, hard suites, needle, memory |
| `/architecture` | Public | Process diagram and known boundaries |

Presets: calculator, hybrid retrieval, supervisor → science, extractive memory.
Signed-in users can also fire the idempotent checkout demo. Token deltas stay
in the transcript; durable events go to Trace.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy services packages --ignore-missing-imports
Set-Location apps/web
npx tsc --noEmit
npm run build
```

Lint is ruff (`E4/E7/E9/F/I/UP/B`, FastAPI `B008` ignored) plus `ruff format`.

Product suite: **89 passed, 0 skipped**. CI also runs academic tests
(`pytest academic/tests`) after `pip install -e academic`. Those tests do not
download language-model weights.

Deployed smoke (`scripts/smoke_test_deployed.py`) is a separate workflow.
The job is **skipped (grey)** until `SMOKE_BASE_URL` is set as a variable or
secret — that is not a green success.

PostgreSQL outbox-claim and first-write progress tests start real Postgres via
`pgserver` when `DATABASE_URL` is not already `postgresql+asyncpg`. They force
two open transactions to overlap with `asyncio.Barrier` after both `SELECT`s.
`docker-compose.yml` matches the CI services.

Print eval numbers without a live model (same file the console snapshot uses):

```powershell
.\.venv\Scripts\python.exe scripts\print_evals.py
```

### Known implementation boundaries

- Provider cost uses configured per-million-token rates. Zero rates are for
  free/local providers, not monetary production measurements.
- Reading progress uses SQLite writer serialization or PostgreSQL `FOR UPDATE`.
- Extractive compression keeps structured facts. It does not guarantee recall
  of implicit prose such as “I don't like meetings on Fridays”.
- The 24-task and supervisor scores include a deterministic keyword-router
  baseline **and** a MiniLM-embedding router on the same tasks. Live runs emit
  both as `tool.routing` (`kind: not_live_llm`). Live-LLM routing has no
  separate measured number.
- A separate hard routing suite (wrong-tool traps, ordered multi-step,
  paraphrase) is allowed to score lower than the 24-task core.
- Production retrieval's neural step is MiniLM sentence-level MaxSim
  (ColBERT-style late interaction). `hybrid_cross_encoder` is a measured
  column using the off-the-shelf MS MARCO MiniLM-L-6 quantized ONNX model.
  That is a checkpoint swap, not local training, and not `bge-reranker`.
- Hard retrieval adds zero-overlap paraphrases plus distractor passages.
- MCP discovery writes quarantined registry rows. Execution requires admin
  approval and an explicit attach to an agent.
- Runtime `tool.foresight` is a reproducible tool-outcome simulator:
  calculator AST, MiniLM retrieval preview, SQL allowlist validation, sonar
  closed form. It is not RAP and not the academic tabular world-model package.
- `intent_router` is MiniLM cosine over frozen intent prototypes with a
  keyword fallback (`method: minilm_embedding_with_keyword_fallback`), not a
  live-LLM classifier.
- The DAG scheduler batches independent reads and isolates writes / argument
  dependencies. It is not a distributed workflow engine.
- `apps/mobile_contract` is a fail-closed source scan for secrets and sensor
  keywords. There is no iOS/Android app.
- Handoff is one-way and sticky. After transfer, the thread stays on that
  specialist.
- Demo Mode reuses `UsageDaily` + `budget_exceeded`. The daily token budget is
  an eventually consistent soft limit: concurrent runs can spend a little past
  `DAILY_TOKEN_BUDGET`.
- Intentionally deferred: Go services, Kubernetes/Helm, S3, ClickHouse,
  Capacitor/RN, real payment settlement, APNs/FCM, file analysis, UUID-pivot
  scale recall, and a locally trained reranker.

## Deploy (optional)

This is wiring, not a claim that a public service is live. Railway Hobby is a
**paid** starting tier. Do not describe this as zero-cost hosting.

| Component | Suggested host |
|---|---|
| `apps/web` | Vercel; set project root to `apps/web` |
| `core_api`, `agent_runtime`, `worker` | Railway Hobby, three services |
| PostgreSQL | Railway plugin or Neon |
| Redis | Upstash (`rediss://…`) |

Start commands (`PYTHONPATH` = repository root):

```text
core_api:        uvicorn services.core_api.app.main:app --host 0.0.0.0 --port $PORT
agent_runtime:   uvicorn services.agent_runtime.app.main:app --host 0.0.0.0 --port $PORT
worker:          python -m services.worker.app.main
```

Production names the Settings classes already read:

```text
APP_ENV=production
DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST/DB
REDIS_URL=rediss://default:TOKEN@REGION.upstash.io:6379
AGENT_RUNTIME_URL=https://<runtime-service>.up.railway.app
RUNTIME_INTERNAL_TOKEN=<shared-secret>
JWT_SECRET=<random-32-byte-secret>
LLM_API_KEY=<your key>
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
DEMO_MODE=true
DAILY_TOKEN_BUDGET=200000
DEMO_RUNS_PER_IP_PER_HOUR=5
CORS_ORIGINS=https://<your-app>.vercel.app
NEXT_PUBLIC_API_BASE_URL=https://<core-api>.up.railway.app
```

Preview `*.vercel.app` origins are allowed by a CORS regex; pin production in
`CORS_ORIGINS`. After a public Core API exists, set repository variable
`SMOKE_BASE_URL` so `.github/workflows/smoke.yml` runs.

## Academic research

See **[academic/README.md](academic/README.md)** for the routed world-model
study (ToolWorld-v1, hypotheses, figures). Do not read those numbers as
ALFWorld, LoRA, or a product-runtime world model.

```powershell
Set-Location academic
..\.venv\Scripts\python.exe -m pip install -e ".[research]"
..\.venv\Scripts\python.exe -m pytest tests -q
```

## Security model

The Runtime treats model output, retrieval results, and tool results as
untrusted. Every tool call is schema-validated, risk-checked, bounded by
timeout/depth, and recorded with actor, version, schema hash, status, duration,
and an output summary. Calculator expressions use an AST allowlist. Read-only
SQL rejects writes, comments, multi-statements, and non-allowlisted tables.

TLS termination, secret-manager integration, MFA, SSRF/IP validation, file
sandboxing, real payment signatures, and real push providers are deployment
extensions, not completed features.

OpenTelemetry is wired (FastAPI/httpx instrumentation). Console spans and OTLP
export are opt-in via `OTEL_CONSOLE_EXPORTER` / `OTEL_EXPORTER_OTLP_ENDPOINT`.

## License

Original source is [MIT](LICENSE). Pretrained checkpoints, papers, and
pip/npm dependencies keep their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
Also [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
