import type { ModelConfig, SplitFractions } from "@/shared/types/api";
import { TERMS } from "@/shared/lib/terms";
import { msg } from "@/shared/lib/messages";
import { sentenceCase } from "@/shared/lib/formatters";

export const emptyModelConfig = (): ModelConfig => ({
  name: "",
  token_source: "managed",
  temperature: 0.7,
  max_tokens: 1024,
});

export const defaultSplit: SplitFractions = { train: 0.7, val: 0.15, test: 0.15 };

// A dataset column's role. React is now a generic GEPA module, so every run —
// react included — maps columns to signature I/O exactly the same way.
export type ColumnRole = "input" | "output" | "ignore";

// UI-side model of the react (ReAct-agent) tool-source configuration. React is
// generic: scoring is owned by the standard authored metric_code, so no reward
// knobs live here — only the live tool roster. Tools always come from a live
// MCP server, unfiltered (the wizard no longer offers dataset snapshots or
// tool filters; the backend keeps supporting both for old runs).
// `use-submit-wizard` reshapes this into the backend's ToolSource wire model
// at submit time.
export interface ReactConfig {
  mcpUrl: string;
  mcpAuthHeader: string;
}

export const defaultReactConfig = (): ReactConfig => ({
  mcpUrl: "",
  mcpAuthHeader: "",
});

// Labels are thunks, not pre-resolved strings: `msg()` reads the active locale's
// catalog, which is delivered out of band and is empty at module-eval time on the
// server (frozen process-wide to the raw key) but populated in the browser —
// resolving eagerly here would hydrate-mismatch. Resolving per render keeps both
// sides inside the request, where the catalog is pinned.
export const STEPS = [
  { id: "basics", label: () => msg("auto.features.submit.constants.literal.1") },
  // Glossary terms are lowercase for mid-sentence use; the stepper sits them
  // beside capitalized labels ("Basic details", "Parameters"), so they get
  // sentence case here.
  { id: "data", label: () => sentenceCase(TERMS.dataset) },
  { id: "params", label: () => msg("auto.features.submit.constants.literal.2") },
  { id: "code", label: () => msg("auto.features.submit.constants.literal.3") },
  { id: "model", label: () => sentenceCase(TERMS.model) },
  { id: "review", label: () => msg("auto.features.submit.constants.literal.4") },
] as const;

export const RECENT_KEY = "skynet:recent-model-configs";
export const MAX_RECENT = 5;

/** RTL: forward = slide from left, backward = slide from right. */
export const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? -80 : 80,
    opacity: 0,
    scale: 0.97,
  }),
  center: {
    x: 0,
    opacity: 1,
    scale: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? 80 : -80,
    opacity: 0,
    scale: 0.97,
  }),
};
