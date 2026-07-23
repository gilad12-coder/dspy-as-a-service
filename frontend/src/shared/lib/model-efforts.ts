import { msg } from "@/shared/lib/messages";

// Effort vocabularies are per-API, not universal, and providers silently
// clamp or reject levels outside their documented ladder — so featured ids
// carry their exact documented ladder and unknown ids fall back to a
// per-family one. An empty ladder disables effort selection entirely
// (MiniMax M3's thinking is an on/off toggle, not a ladder). "max" is
// Sol-only among OpenAI models and "ultra" is a separate mode, not an
// effort level, so it is deliberately absent.
export const DEFAULT_EFFORTS = ["low", "medium", "high"] as const;
const OPENAI_EFFORTS = ["none", "low", "medium", "high", "xhigh"] as const;
const ANTHROPIC_EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;
const MODEL_EFFORTS: Record<string, readonly string[]> = {
  "openrouter/openai/gpt-5.6-sol": ["none", "low", "medium", "high", "xhigh", "max"],
  "openrouter/google/gemini-3.1-pro-preview": ["low", "medium", "high"],
  "openrouter/google/gemini-3.6-flash": ["minimal", "low", "medium", "high"],
  "openrouter/x-ai/grok-4.5": ["low", "medium", "high"],
  "openrouter/meta/muse-spark-1.1": ["minimal", "low", "medium", "high", "xhigh"],
  // DeepSeek accepts the full set but only none/high/max are distinct.
  "openrouter/deepseek/deepseek-v4-pro": ["none", "high", "max"],
  "openrouter/moonshotai/kimi-k3": ["low", "high", "max"],
  "openrouter/minimax/minimax-m3": [],
};

/** The reasoning-effort ladder a model actually supports. */
export function effortsFor(model: string | null): readonly string[] {
  if (!model) return DEFAULT_EFFORTS;
  const exact = MODEL_EFFORTS[model];
  if (exact) return exact;
  if (model.includes("anthropic/claude")) return ANTHROPIC_EFFORTS;
  // Matches both direct ids and OpenRouter-prefixed ones ("openrouter/openai/…").
  if (model.includes("openai/")) return OPENAI_EFFORTS;
  return DEFAULT_EFFORTS;
}

/** Localized display label for a reasoning-effort level. */
export function effortLabel(level: string): string {
  switch (level) {
    case "none":
      return msg("agent.model_menu.effort_none");
    case "minimal":
      return msg("agent.model_menu.effort_minimal");
    case "low":
      return msg("agent.model_menu.effort_low");
    case "medium":
      return msg("agent.model_menu.effort_medium");
    case "high":
      return msg("agent.model_menu.effort_high");
    case "xhigh":
      return msg("agent.model_menu.effort_xhigh");
    case "max":
      return msg("agent.model_menu.effort_max");
    default:
      return level;
  }
}
