"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, type FormEvent } from "react";

import { AppChrome } from "./components/AppChrome";
import { AuthMenu } from "./components/AuthMenu";
import { TraceList } from "./components/TraceList";
import { API, DEMO_BUDGET_COPY, TOKEN_KEY, request, streamEvents } from "./lib/api";
import { AGENT_BLURB, AGENT_LABEL, CHECKOUT_PRESET, PRESETS, insightsFromEvents } from "./lib/events";
import type { Agent, EventRecord, Preset, PublicStatus, RunRead, ToolRow, User } from "./lib/types";

export default function PlaygroundPage() {
  const [token, setToken] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [agentId, setAgentId] = useState("");
  const [threadId, setThreadId] = useState("");
  const [runId, setRunId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [answer, setAnswer] = useState("");
  const [userTurn, setUserTurn] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [apiLive, setApiLive] = useState<boolean | null>(null);
  const [stack, setStack] = useState<PublicStatus | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);

  useEffect(() => {
    request<Agent[]>("/v1/agents")
      .then((items) => {
        setAgents(items);
        const preferred = items.find((item) => item.slug === "default-assistant") ?? items[0];
        if (preferred) setAgentId(preferred.id);
        setApiLive(true);
      })
      .catch((reason: Error) => {
        setError(reason.message);
        setApiLive(false);
      });
    request<ToolRow[]>("/v1/tools/catalog")
      .then(setTools)
      .catch(() => setTools([]));
    fetch(`${API}/v1/status`)
      .then(async (response) => {
        if (!response.ok) {
          setApiLive(false);
          return;
        }
        const body = (await response.json()) as PublicStatus;
        setStack(body);
        setApiLive(true);
      })
      .catch(() => setApiLive(false));
  }, []);

  useEffect(() => {
    const stored = window.localStorage.getItem(TOKEN_KEY);
    if (!stored) return;
    request<User>("/v1/auth/me", { headers: { Authorization: `Bearer ${stored}` } })
      .then((me) => {
        setToken(stored);
        setUser(me);
      })
      .catch(() => {
        window.localStorage.removeItem(TOKEN_KEY);
      });
  }, []);

  const selected = agents.find((agent) => agent.id === agentId);
  const insights = useMemo(() => insightsFromEvents(events), [events]);
  const hasChat = Boolean(userTurn || answer);

  function persistSession(nextToken: string, me: User) {
    setToken(nextToken);
    setUser(me);
    window.localStorage.setItem(TOKEN_KEY, nextToken);
  }

  function logout() {
    setToken("");
    setUser(null);
    window.localStorage.removeItem(TOKEN_KEY);
  }

  async function authenticate(email: string, password: string, mode: "login" | "register") {
    setBusy(true);
    setError("");
    try {
      if (mode === "register") {
        await request("/v1/auth/register", {
          method: "POST",
          body: JSON.stringify({ email, display_name: "Demo User", password }),
        });
      }
      const result = await request<{ access_token: string }>("/v1/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const me = await request<User>("/v1/auth/me", {
        headers: { Authorization: `Bearer ${result.access_token}` },
      });
      persistSession(result.access_token, me);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  function applyPreset(preset: Preset) {
    if (preset.kind === "checkout") {
      void runIdempotencyDemo();
      return;
    }
    if (preset.agentSlug) {
      const match = agents.find((agent) => agent.slug === preset.agentSlug);
      if (match) {
        if (match.id !== agentId) setThreadId("");
        setAgentId(match.id);
      }
    }
    if (preset.prompt) setPrompt(preset.prompt);
    setNote(preset.note ?? "");
  }

  async function runIdempotencyDemo() {
    if (!token) {
      setError("Sign in to submit the same checkout twice.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const key = `demo-order-${Date.now()}`;
      const payload = { type: "credits", product_ref: "credits-100", amount: 500, currency: "usd" };
      const headers = { Authorization: `Bearer ${token}`, "Idempotency-Key": key };
      const first = await request<{ id: string }>("/v1/checkout", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      const second = await request<{ id: string }>("/v1/checkout", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      setNote(
        first.id === second.id
          ? `Both checkouts returned order ${first.id.slice(0, 8)}…`
          : `Unexpected: ${first.id} vs ${second.id}`,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Checkout demo failed");
    } finally {
      setBusy(false);
    }
  }

  async function startRun(event: FormEvent) {
    event.preventDefault();
    if (!token || !agentId || !user) return;
    setBusy(true);
    setError("");
    setEvents([]);
    setAnswer("");
    setUserTurn(prompt);
    try {
      let nextThread = threadId;
      if (!nextThread) {
        const thread = await request<{ id: string }>("/v1/threads", {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: JSON.stringify({ user_id: user.id, agent_id: agentId, title: "Playground" }),
        });
        nextThread = thread.id;
        setThreadId(nextThread);
      }
      const run = await request<RunRead>(`/v1/threads/${nextThread}/runs`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify({ user_id: user.id, content: prompt }),
      });
      setRunId(run.id);
      setTraceOpen(true);
      if (run.status === "budget_exceeded" && run.terminal_reason === "demo_daily_budget") {
        setError(DEMO_BUDGET_COPY);
      }
      await streamEvents(run.id, token, (parsed) => {
        setEvents((current) => [...current, parsed]);
        if (parsed.type === "message.delta") setAnswer((current) => current + String(parsed.payload.delta ?? ""));
        if (parsed.type === "run.failed") {
          const reason = String(parsed.payload.reason ?? parsed.payload.message ?? "Run failed");
          setError(
            reason.includes("LLM_API_KEY")
              ? "Runtime is up, but LLM_API_KEY and LLM_MODEL are not set. Evals still work without a vendor key."
              : reason,
          );
        }
        if (parsed.type === "run.budget_exceeded" && parsed.payload.reason === "demo_daily_budget") {
          setError(DEMO_BUDGET_COPY);
        }
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Run failed");
    } finally {
      setBusy(false);
    }
  }

  async function cancelRun() {
    if (!token || !runId) return;
    try {
      await request(`/v1/runs/${runId}/cancel`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Cancel failed");
    }
  }

  function newThread() {
    setThreadId("");
    setRunId("");
    setEvents([]);
    setAnswer("");
    setUserTurn("");
    setNote("");
    setError("");
    setPrompt("");
  }

  return (
    <AppChrome
      wide
      apiLive={apiLive}
      status={busy ? "Running" : undefined}
      right={<AuthMenu user={user} busy={busy} onLogin={authenticate} onLogout={logout} />}
    >
      <div className={`stage ${traceOpen ? "with-trace" : ""}`}>
        <section className="chat">
          {!hasChat ? (
            <div className="hero">
              <p className="hero-kicker">Playground</p>
              <h1>What should the runtime do?</h1>
              <p>
                Four catalog agents. <Link href="/evals">Evals</Link>, <Link href="/tools">tools</Link>, and{" "}
                <Link href="/architecture">architecture</Link> are public and need no vendor key. A playground run
                needs sign-in plus <code>LLM_API_KEY</code>.
              </p>
              {stack && !stack.runtime && (
                <p className="note">Core API is up. Start the Runtime on :8101 to execute runs.</p>
              )}
              {stack?.runtime && !stack.llm_configured && (
                <p className="note">Runtime is up without an LLM key. Runs fail closed; they never invent an answer.</p>
              )}
              <div className="suggest">
                {PRESETS.map((preset) => (
                  <button type="button" key={preset.id} className="suggest-card" disabled={busy} onClick={() => applyPreset(preset)}>
                    <strong>{preset.label}</strong>
                    <span>{preset.hint}</span>
                  </button>
                ))}
              </div>
              {user && (
                <button type="button" className="hero-link" disabled={busy} onClick={() => applyPreset(CHECKOUT_PRESET)}>
                  {CHECKOUT_PRESET.label} · {CHECKOUT_PRESET.hint}
                </button>
              )}
              {note && <p className="note">{note}</p>}
            </div>
          ) : (
            <div className="messages">
              {note && <p className="note">{note}</p>}
              <article className="msg you">
                <b>You</b>
                <p>{userTurn}</p>
              </article>
              <article className="msg bot">
                <b>{insights.specialist ? AGENT_LABEL[insights.specialist] ?? insights.specialist : selected ? AGENT_LABEL[selected.slug] ?? selected.slug : "Assistant"}</b>
                {(insights.tools.length > 0 || insights.passageIds.length > 0 || insights.factsKept !== null) && (
                  <div className="chips">
                    {insights.tools.map((name) => (
                      <em key={name}>{name}</em>
                    ))}
                    {insights.passageIds.map((id) => (
                      <em key={id}>{id}</em>
                    ))}
                    {insights.routingKind && <em>{insights.routingKind}</em>}
                    {insights.foresightKind && <em>{insights.foresightKind}</em>}
                    {insights.factsKept !== null && (
                      <em title="Extractive compression keeps structured facts only">
                        compressed · {insights.factsKept} facts
                      </em>
                    )}
                  </div>
                )}
                {answer ? <p>{answer}</p> : <p className="muted">{busy ? "Thinking…" : insights.failReason || ""}</p>}
              </article>
            </div>
          )}

          <form className="dock" onSubmit={startRun}>
            <div className="dock-meta">
              <label className="picker">
                <select
                  value={agentId}
                  onChange={(event) => {
                    setAgentId(event.target.value);
                    setThreadId("");
                  }}
                >
                  {agents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {AGENT_LABEL[agent.slug] ?? agent.slug}
                    </option>
                  ))}
                </select>
              </label>
              <span className="picker-hint">{selected ? AGENT_BLURB[selected.slug] : ""}</span>
              <Link className="dock-link" href="/tools">
                {tools.length} tools
              </Link>
              <button type="button" className="ghost-btn" onClick={newThread} disabled={busy || !hasChat}>
                New
              </button>
              <button type="button" className={`ghost-btn ${traceOpen ? "on" : ""}`} onClick={() => setTraceOpen((current) => !current)}>
                Trace{events.length ? ` ${events.length}` : ""}
              </button>
              {busy && (
                <button type="button" className="ghost-btn danger" onClick={() => void cancelRun()}>
                  Stop
                </button>
              )}
            </div>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={2}
              placeholder={user ? "Message the agent…" : "Sign in to start a run"}
            />
            <div className="dock-bar">
              <small>8 steps max · high-risk tools hidden in Demo Mode</small>
              <button type="submit" className="primary-btn" disabled={busy || !user || !agentId || !prompt.trim()}>
                {busy ? "Running" : "Send"}
              </button>
            </div>
          </form>
          {error && <p className="banner error dock-error">{error}</p>}
        </section>

        {traceOpen && (
          <aside className="inspector">
            <header>
              <div>
                <h2>Trace</h2>
                <p>SQL events via SSE</p>
              </div>
              <button type="button" className="ghost-btn" onClick={() => setTraceOpen(false)}>
                Close
              </button>
            </header>
            <TraceList events={events} />
          </aside>
        )}
      </div>
    </AppChrome>
  );
}
