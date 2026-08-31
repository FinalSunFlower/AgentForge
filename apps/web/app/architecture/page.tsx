"use client";

import { useEffect } from "react";

import { AppChrome } from "../components/AppChrome";

const DIAGRAM = `flowchart TB
  Web["Next.js console"]
  API["Core API"]
  RT["Agent Runtime"]
  DB[("SQLite or PostgreSQL")]
  Redis[("Redis optional")]
  WK["Worker"]
  Web --> API
  API -->|"internal token"| RT
  API --> DB
  RT --> DB
  RT --> Redis
  WK --> DB
  API --> Redis`;

const BOUNDARIES = [
  {
    title: "Cost math",
    body: "Provider cost uses configured per-million-token rates. Zero rates are for free/local providers, not monetary production measurements.",
  },
  {
    title: "Concurrency gates",
    body: "Progress and outbox tests run against real Postgres (CI Docker, or pip pgserver). Overlap is forced with a barrier after both SELECTs.",
  },
  {
    title: "Extractive memory",
    body: "Compression keeps structured facts. It does not guarantee recall of implicit prose such as “I don't like meetings on Fridays”.",
  },
  {
    title: "Routing scores",
    body: "24-task and supervisor numbers are keyword-router and MiniLM-embedding-router baselines, now also emitted on live runs as tool.routing. A separate hard suite (traps, paraphrase, ordered multi-tool) is allowed to score lower. Live-LLM routing has no measured number.",
  },
  {
    title: "Reranker",
    body: "Production retrieval uses MiniLM late-interaction MaxSim, with a measured off-the-shelf MS MARCO MiniLM-L-6 ONNX cross-encoder column. Feature rerank remains a baseline. No local training.",
  },
  {
    title: "MCP tools",
    body: "Admin POST /v1/admin/mcp/sync lists remote tools into quarantined registry rows. They stay unused until approve + attach to an agent. Auto-approve is not implemented.",
  },
  {
    title: "Product foresight",
    body: "tool.foresight is a deterministic tool-outcome simulator: calculator AST, retrieval MiniLM preview, SQL allowlist check, sonar closed form. Not RAP and not the academic world-model package.",
  },
  {
    title: "Handoff",
    body: "One-way and sticky. After a thread moves to a specialist it stays there. No automatic return or switch.",
  },
  {
    title: "Hosting",
    body: "A public demo is not a production deployment. Railway Hobby is paid. Neon free can cold-start. Not zero-cost hosting.",
  },
  {
    title: "Closed-loop demo",
    body: "Fresh clones seed Harbor Field Notes from the eval corpus. /evals serves data/evals_snapshot.json. Playground runs still require an LLM key and fail closed without one.",
  },
];

export default function ArchitecturePage() {
  useEffect(() => {
    const existing = document.querySelector<HTMLScriptElement>("script[data-mermaid]");
    const run = () => {
      const mermaid = (window as unknown as { mermaid?: { initialize: (c: object) => void; run: () => void } }).mermaid;
      if (!mermaid) return;
      const dark = document.documentElement.dataset.theme !== "light";
      mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral", securityLevel: "strict" });
      mermaid.run();
    };
    if (existing) {
      run();
      return;
    }
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
    script.async = true;
    script.dataset.mermaid = "true";
    script.onload = run;
    document.body.appendChild(script);
  }, []);

  return (
    <AppChrome status="Scope contract">
      <div className="page-head">
        <div>
          <p className="kicker">System</p>
          <h1>Architecture</h1>
          <p>
            Local profile is SQLite. Production points the same SQLAlchemy models at Postgres and optional Redis. That
            is environment wiring, not a second implementation.
          </p>
        </div>
      </div>
      <section className="table-card diagram-card">
        <h2>Process topology</h2>
        <pre className="mermaid architecture-diagram">{DIAGRAM}</pre>
      </section>
      <section>
        <h2>Known implementation boundaries</h2>
        <div className="bound-grid">
          {BOUNDARIES.map((item) => (
            <article className="bound-card" key={item.title}>
              <strong>{item.title}</strong>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </section>
    </AppChrome>
  );
}
