"use client";

import { useEffect, useState } from "react";

import { AppChrome } from "../components/AppChrome";
import { API } from "../lib/api";
import type { EvalsSummary, ModeRow, RouterRow } from "../lib/types";

function formatScore(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(2);
}

function modeRows(rows: Record<string, ModeRow | number | undefined>): [string, ModeRow][] {
  return Object.entries(rows).filter((entry): entry is [string, ModeRow] => {
    const value = entry[1];
    return Boolean(value && typeof value === "object" && "recall_at_k" in value);
  });
}

function ScoreTable({ rows }: { rows: Record<string, ModeRow | number | undefined> }) {
  return (
    <div className="table-card">
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Recall@k</th>
            <th>MRR</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {modeRows(rows).map(([name, row]) => (
            <tr key={name}>
              <td>
                {name === "vector"
                  ? "vector (MiniLM)"
                  : name === "hybrid_late_interaction"
                    ? "hybrid + MiniLM MaxSim"
                    : name === "hybrid_cross_encoder"
                      ? "hybrid + MS MARCO MiniLM CE"
                      : name}
              </td>
              <td className="mono">{formatScore(row.recall_at_k)}</td>
              <td className="mono">{formatScore(row.mrr)}</td>
              <td>
                <span className="bar">
                  <i style={{ width: `${Math.max(2, row.recall_at_k * 100)}%` }} />
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EvalsPage() {
  const [data, setData] = useState<EvalsSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/v1/evals/summary`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`GET /v1/evals/summary failed (${response.status})`);
        setData((await response.json()) as EvalsSummary);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const memoryPassed = data ? Object.values(data.memory.survived).filter(Boolean).length : 0;
  const memoryTotal = data ? Object.keys(data.memory.survived).length : 0;

  return (
    <AppChrome status={data ? (data.source === "live" ? "Recomputed" : "Snapshot") : "Loading"}>
      <div className="page-head">
        <div>
          <p className="kicker">Evidence</p>
          <h1>Eval snapshot</h1>
          <p>
            Checked-in snapshot of <code>GET /v1/evals/summary</code>. No live model. Routing scores are keyword and
            MiniLM-embedding baselines, not live-LLM routing. <code>hybrid_cross_encoder</code> is the off-the-shelf MS
            MARCO MiniLM-L-6 ONNX model. Hard suites are trap / paraphrase / multi-step tasks and are allowed to score
            lower.
          </p>
        </div>
      </div>
      {error && <p className="banner error">{error}</p>}
      {!data && !error && <p className="muted">Loading the checked-in eval snapshot…</p>}
      {data && (
        <>
          <section className="stat-grid">
            <article className="stat">
              <small>Zero-overlap MiniLM</small>
              <strong>{formatScore(data.zero_overlap.minilm_recall_at_3)}</strong>
              <span>recall@3 · keyword/BM25 are 0.00</span>
            </article>
            <article className="stat">
              <small>Needle</small>
              <strong>{data.needle.survives_compression ? "Pass" : "Fail"}</strong>
              <span>{data.needle.scope}</span>
            </article>
            <article className="stat">
              <small>Structured memory</small>
              <strong>
                {memoryPassed}/{memoryTotal}
              </strong>
              <span>explicit facts recalled</span>
            </article>
          </section>
          <div className="split-docs">
            <section>
              <h2>Retrieval modes</h2>
              <ScoreTable rows={data.retrieval} />
            </section>
            <section>
              <h2>Zero n-gram overlap</h2>
              <ScoreTable rows={data.zero_overlap} />
            </section>
          </div>
          {data.hard_retrieval && (
            <section>
              <h2>Hard retrieval (paraphrase + distractors)</h2>
              <ScoreTable rows={data.hard_retrieval} />
            </section>
          )}
          {data.routing && (
            <section className="table-card">
              <h2>Tool routing</h2>
              <table>
                <thead>
                  <tr>
                    <th>Router</th>
                    <th>Kind</th>
                    <th>Selection</th>
                    <th>Success</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ["keyword 24-task", data.routing.keyword],
                      ["MiniLM 24-task", data.routing.embedding],
                      ["keyword supervisor", data.routing.supervisor_keyword],
                      ["MiniLM supervisor", data.routing.supervisor_embedding],
                      ...(data.routing.hard_keyword ? [["hard keyword", data.routing.hard_keyword] as [string, RouterRow]] : []),
                      ...(data.routing.hard_embedding
                        ? [["hard MiniLM", data.routing.hard_embedding] as [string, RouterRow]]
                        : []),
                    ] as [string, RouterRow][]
                  ).map(([label, row]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      <td className="mono">{row.eval_kind}</td>
                      <td className="mono">{formatScore(row.tool_selection_accuracy)}</td>
                      <td className="mono">{formatScore(row.task_success_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="fineprint">{data.routing.embedding.claim}</p>
              {data.routing.hard_keyword?.claim && (
                <p className="fineprint">{data.routing.hard_keyword.claim}</p>
              )}
            </section>
          )}
          <section className="table-card">
            <h2>Memory cases</h2>
            <table>
              <tbody>
                {Object.entries(data.memory.survived).map(([name, ok]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td>{ok ? "recalled" : "missed"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="fineprint">{data.memory.claim}</p>
          </section>
          <p className="fineprint">{data.disclaimer}</p>
        </>
      )}
    </AppChrome>
  );
}
