import type { EventRecord } from "./types";

export const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8100";

export const DEMO_BUDGET_COPY =
  "Today's public demo token budget is exhausted. Run locally or try again tomorrow.";

type ApiErrorBody = {
  error?: { code?: string; message?: string };
  detail?: unknown;
};

export function formatApiError(body: ApiErrorBody | null, status: number): Error {
  const code = body?.error?.code ?? "";
  if (code === "demo_ip_rate_limited") {
    return new Error("This IP has reached the public demo run limit. Try again later, or run locally.");
  }
  if (code === "demo_daily_budget" || body?.error?.message?.includes("demo token budget")) {
    return new Error(`${DEMO_BUDGET_COPY} See Quick start in the README.`);
  }
  const message =
    body?.error?.message ?? (typeof body?.detail === "string" ? body.detail : `Request failed (${status})`);
  return new Error(message);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    throw formatApiError((await response.json().catch(() => null)) as ApiErrorBody | null, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const TOKEN_KEY = "agentforge-token";
export const THEME_KEY = "agentforge-theme";

export async function streamEvents(
  runId: string,
  token: string,
  onEvent: (event: EventRecord) => void,
): Promise<void> {
  let lastId = "0";
  let done = false;
  while (!done) {
    const response = await fetch(`${API}/v1/runs/${runId}/events`, {
      headers: { Authorization: `Bearer ${token}`, "Last-Event-ID": lastId },
    });
    if (!response.ok || !response.body) throw new Error("Unable to open event stream");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const id = frame.match(/^id: (.+)$/m)?.[1];
        const data = frame.match(/^data: (.+)$/m)?.[1];
        if (!data) continue;
        if (id) lastId = id;
        const parsed = JSON.parse(data) as EventRecord;
        onEvent(parsed);
        if (
          ["run.completed", "run.failed", "run.canceled", "run.budget_exceeded", "run.approval_required"].includes(
            parsed.type,
          )
        ) {
          done = true;
        }
      }
    }
    if (!done) await new Promise((resolve) => setTimeout(resolve, 250));
  }
}
