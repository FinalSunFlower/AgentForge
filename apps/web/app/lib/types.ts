export type Agent = {
  id: string;
  slug: string;
  version: string;
  model_ref: string;
  status: string;
};

export type ToolRow = {
  id: string;
  name: string;
  version: string;
  source: string;
  description: string;
  risk_level: string;
  status: string;
  schema_hash: string;
};

export type User = {
  id: string;
  email: string;
  display_name: string;
  plan: string;
};

export type EventRecord = {
  type: string;
  sequence: number;
  payload: Record<string, unknown>;
};

export type RunRead = {
  id: string;
  thread_id: string;
  status: string;
  terminal_reason?: string | null;
};

export type ModeRow = { recall_at_k: number; mrr: number; queries: number };

export type RouterRow = {
  tasks: number;
  tool_selection_accuracy: number;
  param_accuracy: number;
  task_success_rate: number;
  avg_steps: number;
  eval_kind: string;
  claim: string;
};

export type EvalsSummary = {
  retrieval: Record<string, ModeRow>;
  zero_overlap: Record<string, ModeRow | number | undefined> & { minilm_recall_at_3?: number };
  needle: { survives_compression: boolean; scope: string };
  memory: { survived: Record<string, boolean>; scope: string; claim: string };
  routing?: {
    keyword: RouterRow;
    embedding: RouterRow;
    supervisor_keyword: RouterRow;
    supervisor_embedding: RouterRow;
    hard_keyword?: RouterRow & { suite?: string };
    hard_embedding?: RouterRow & { suite?: string };
  };
  hard_retrieval?: Record<string, ModeRow>;
  disclaimer: string;
  source?: "snapshot" | "live";
};

export type PublicStatus = {
  api: string;
  runtime: boolean;
  llm_configured: boolean;
  demo_mode: boolean;
  evals_source?: string;
};

export type Preset = {
  id: string;
  label: string;
  hint: string;
  prompt?: string;
  agentSlug?: string;
  kind?: "run" | "checkout";
  note?: string;
};
