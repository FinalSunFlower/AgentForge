import type { EventRecord, Preset } from "./types";

export type RunInsight = {
  specialist: string | null;
  factsKept: number | null;
  droppedTurns: number | null;
  passageIds: string[];
  tools: string[];
  tokens: number | null;
  status: string | null;
  failReason: string | null;
  routingKind: string | null;
  foresightKind: string | null;
};

const TRACE_SKIP = new Set(["message.delta", "reasoning.delta"]);

export function insightsFromEvents(events: EventRecord[]): RunInsight {
  let specialist: string | null = null;
  let factsKept: number | null = null;
  let droppedTurns: number | null = null;
  let tokens: number | null = null;
  let status: string | null = null;
  let failReason: string | null = null;
  let routingKind: string | null = null;
  let foresightKind: string | null = null;
  const passageIds: string[] = [];
  const tools: string[] = [];
  const seenPassages = new Set<string>();
  const seenTools = new Set<string>();
  for (const event of events) {
    if (event.type === "tool.routing") {
      routingKind = String(event.payload.kind ?? "not_live_llm");
    }
    if (event.type === "tool.foresight") {
      foresightKind = String(event.payload.kind ?? "tool_outcome_simulator");
    }
    if (event.type === "run.failed") {
      failReason = String(event.payload.reason ?? event.payload.message ?? "run_failed");
    }
    if (event.type === "agent.handoff") {
      const slug = event.payload.agent_slug ?? event.payload.specialist;
      specialist = slug ? String(slug) : null;
    }
    if (event.type === "context.compressed") {
      factsKept = Number(event.payload.facts_kept ?? 0);
      droppedTurns = Number(event.payload.dropped_turns ?? 0);
    }
    if (event.type === "usage.final") {
      tokens = Number(event.payload.input_tokens ?? 0) + Number(event.payload.output_tokens ?? 0);
    }
    if (event.type.startsWith("run.")) status = event.type.replace("run.", "");
    const toolName = event.payload.tool_name;
    if (typeof toolName === "string" && !seenTools.has(toolName)) {
      seenTools.add(toolName);
      tools.push(toolName);
    }
    if (event.type !== "tool.result") continue;
    const result = event.payload.result;
    if (!result || typeof result !== "object" || !("results" in result)) continue;
    const hits = (result as { results?: Array<{ passage_id?: string }> }).results ?? [];
    for (const hit of hits) {
      if (hit.passage_id && !seenPassages.has(hit.passage_id)) {
        seenPassages.add(hit.passage_id);
        passageIds.push(hit.passage_id);
      }
    }
  }
  return {
    specialist,
    factsKept,
    droppedTurns,
    passageIds,
    tools,
    tokens,
    status,
    failReason,
    routingKind,
    foresightKind,
  };
}

export function visibleTrace(events: EventRecord[]): EventRecord[] {
  return events.filter((event) => !TRACE_SKIP.has(event.type));
}

export function eventLabel(type: string): string {
  const labels: Record<string, string> = {
    "run.started": "Run started",
    "tool.requested": "Tool requested",
    "tool.started": "Tool started",
    "tool.progress": "Tool progress",
    "tool.foresight": "Tool foresight",
    "tool.routing": "Tool routing",
    "tool.result": "Tool result",
    "agent.handoff": "Handoff",
    "context.compressed": "Context compressed",
    "usage.final": "Usage",
    "run.completed": "Completed",
    "run.failed": "Failed",
    "run.canceled": "Canceled",
    "run.budget_exceeded": "Budget exceeded",
    "run.approval_required": "Approval required",
  };
  return labels[type] ?? type;
}

export function eventSummary(event: EventRecord): string {
  const payload = event.payload;
  if (event.type === "tool.requested" || event.type === "tool.started" || event.type === "tool.result") {
    const name = payload.tool_name ? String(payload.tool_name) : "tool";
    const status = payload.status ? String(payload.status) : "";
    return status ? `${name} · ${status}` : name;
  }
  if (event.type === "tool.foresight") {
    return payload.kind ? String(payload.kind) : "preview";
  }
  if (event.type === "tool.routing") {
    return payload.agreement ? "keyword = MiniLM" : "keyword ≠ MiniLM";
  }
  if (event.type === "agent.handoff") {
    return String(payload.agent_slug ?? payload.specialist ?? "specialist");
  }
  if (event.type === "context.compressed") {
    return `kept ${payload.facts_kept ?? 0} facts`;
  }
  if (event.type === "usage.final") {
    const inTok = Number(payload.input_tokens ?? 0);
    const outTok = Number(payload.output_tokens ?? 0);
    return `${inTok + outTok} tokens`;
  }
  if (typeof payload.reason === "string") return payload.reason;
  if (typeof payload.message === "string") return payload.message;
  return "";
}

export const AGENT_LABEL: Record<string, string> = {
  "academic-writer": "Writer",
  supervisor: "Supervisor",
  "code-data-specialist": "Code & data",
  "retrieval-specialist": "Retrieval",
};

export const AGENT_BLURB: Record<string, string> = {
  "academic-writer": "Citation-grounded drafting",
  supervisor: "One-way handoff to a specialist",
  "code-data-specialist": "Calculator, plots, read-only SQL",
  "retrieval-specialist": "Hybrid search with citations",
};

export const PRESETS: Preset[] = [
  {
    id: "calc",
    label: "Calculate 12 × (3 + 4)",
    hint: "Writer · AST-checked calculator is on the code-data specialist",
    prompt: "Calculate 12*(3+4)",
    agentSlug: "code-data-specialist",
  },
  {
    id: "retrieval",
    label: "Search the literature notes",
    hint: "Writer · hybrid retrieval with citations",
    prompt: "Search the literature notes for hierarchical task networks. Cite every passage_id you use.",
    agentSlug: "academic-writer",
  },
  {
    id: "arxiv",
    label: "Closed arXiv catalog",
    hint: "Supervisor hands off to code-data or retrieval",
    prompt: "Search the arXiv literature catalog for ReAct tool traces. Do not invent a paper.",
    agentSlug: "supervisor",
  },
  {
    id: "memory",
    label: "Remember structured facts",
    hint: "Writer · extractive memory",
    prompt:
      "Help me remember that my name is Alex and I work on retrieval evaluation. If I later ask who I am, recall those structured facts.",
    agentSlug: "academic-writer",
    note: "This starts an explicit-fact turn. The measured 70-turn needle is on Evals — one prompt does not compress by itself.",
  },
];

export const QUOTA_PRESET: Preset = {
  id: "idempotency",
  label: "Idempotent quota reservation",
  hint: "Same reservation key twice",
  kind: "quota",
};
