# Security

AgentForge is an auditable runtime prototype. Treat a public deploy as a demo,
not a production tenancy.

## Secrets

- Copy `.env.example` to `.env`. Never commit `.env`.
- Rotate `JWT_SECRET`, `RUNTIME_INTERNAL_TOKEN`, and `WEBHOOK_SIGNING_SECRET`
  before any host that is reachable from the internet.
- The Runtime rejects runs when `LLM_API_KEY` or `LLM_MODEL` is missing. It
  does not invent answers.
- `GET /v1/status` reports only booleans (`runtime`, `llm_configured`). It
  never returns keys or model names.

## Trust boundary

Model output, retrieval snippets, and tool results are untrusted. Tool calls
are schema-validated, risk-checked, depth/timeout bounded, and written to the
SQL event log with schema hash and an output summary.

Calculator expressions use an AST allowlist. Read-only SQL rejects writes,
comments, multi-statements, and tables outside `{papers, paper_sections, posts,
usage_daily}`.

## Demo Mode

`DEMO_MODE=true` applies a global daily token budget and a per-IP run window.
High-risk tools stay in the public catalog page but are stripped from the
model-visible catalog. The budget is an eventually consistent soft limit.

## Reporting

Open a GitHub issue with the `security` label. Do not file a public issue for
an active secret leak — rotate the secret first, then describe the class of
bug without the value.
