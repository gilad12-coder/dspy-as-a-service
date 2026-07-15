export type AgentStatus = "idle" | "streaming" | "done" | "error";

export type AgentToolStatus = "running" | "done" | "error";

export interface AgentToolCall {
  id: string;
  tool: string;
  reason: string;
  status: AgentToolStatus;
  startedAt: number;
  endedAt: number | null;
  payload?: Record<string, unknown>;
}

export interface AgentMessage {
  role: "assistant" | "user";
  content: string;
  toolCalls?: AgentToolCall[];
  model?: string | null;
}

export interface AgentThinking {
  reasoning: string;
  startedAt: number | null;
  endedAt: number | null;
  streaming: boolean;
}

/**
 * One pickable answer offered for a closed interview question — the Claude
 * Code / Codex-style multiple-choice option. The picker always adds its own
 * free-text path, so a `QuestionChoice` never represents "other".
 */
export interface QuestionChoice {
  label: string;
  /** One-line elaboration of what picking this answer means; may be empty. */
  description: string;
}
