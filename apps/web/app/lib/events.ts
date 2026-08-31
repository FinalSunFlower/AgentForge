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
  "default-assistant": "Assistant",
  supervisor: "Supervisor",
  "science-specialist": "Science",
  "retrieval-specialist": "Retrieval",
};

export const AGENT_BLURB: Record<string, string> = {
  "default-assistant": "ReAct + built-in tools",
  supervisor: "One-way handoff to a specialist",
  "science-specialist": "Calculator, sonar, wind tunnel",
  "retrieval-specialist": "Hybrid search with citations",
};

export const PRESETS: Preset[] = [
  {
    id: "calc",
    label: "Calculate 12 × (3 + 4)",
    hint: "Assistant · AST-checked calculator",
    prompt: "Calculate 12*(3+4)",
    agentSlug: "default-assistant",
  },
  {
    id: "retrieval",
    label: "Search the novel corpus",
    hint: "Assistant · hybrid retrieval with citations",
    prompt: "Search the notes for the monsoon delay of the caravan. Cite every passage_id you use.",
    agentSlug: "default-assistant",
  },
  {
    id: "sonar",
    label: "Passive sonar ranging",
    hint: "Supervisor hands off to Science",
    prompt: "Triangulate the passive sonar source with the science tools. Do not invent a bearing.",
    agentSlug: "supervisor",
  },
  {
    id: "memory",
    label: "Remember structured facts",
    hint: "Assistant · extractive memory",
    prompt:
      "Help me remember that my name is Alex and I like science-fiction novels. If I later ask who I am, recall those structured facts.",
    agentSlug: "default-assistant",
    note: "This starts an explicit-fact turn. The measured 70-turn needle is on Evals — one prompt does not compress by itself.",
  },
];

export const CHECKOUT_PRESET: Preset = {
  id: "idempotency",
  label: "Idempotent checkout",
  hint: "Same order key twice",
  kind: "checkout",
};
