/**
 * Model-access policy: which models a given token mode + balance can run.
 *
 * The free/paid line is drawn by tier. In managed mode an account with no
 * purchased balance can run mini models on its monthly grant, but the frontier
 * task/reflection models are locked until it buys credits. BYOK mode bills the
 * user's own provider, so nothing is locked there.
 *
 * Tier is inferred from the model id by family hint until the backend tags each
 * catalog model with a real tier — unknown ids stay accessible (we never lock
 * what we can't confidently classify). Framework-agnostic, no React imports.
 */

import type { TokenSourceMode } from "./credit";

/** Coarse capability/price tier of a model. */
export type ModelTier = "mini" | "frontier";

// Small/cheap families — always runnable on the free grant. Checked first so a
// "gpt-5.5-mini" resolves to mini even though it also matches a frontier family.
const MINI_HINTS = ["mini", "haiku", "flash", "small", "lite", "nano", "gemma", "8b", "7b"];

// Premium families gated behind a purchased balance in managed mode.
const FRONTIER_HINTS = [
  "opus",
  "sonnet",
  "gpt-5",
  "gpt-4o",
  "o3",
  "o4",
  "gemini-3-pro",
  "grok",
  "deepseek-r1",
  "405b",
];

/** Classify a model id into a tier. Unknown ids default to mini (never gated). */
export function modelTier(modelValue: string): ModelTier {
  const v = modelValue.toLowerCase();
  if (MINI_HINTS.some((h) => v.includes(h))) return "mini";
  if (FRONTIER_HINTS.some((h) => v.includes(h))) return "frontier";
  return "mini";
}

/**
 * Is this model locked for the current token mode + entitlement?
 *
 * Only managed mode locks anything, and only the frontier tier. The account is
 * entitled to frontier when `frontierUnlocked` is true (it holds purchased
 * credits or has an active Premium subscription). BYOK never locks.
 */
export function isModelLocked(
  modelValue: string,
  mode: TokenSourceMode,
  frontierUnlocked: boolean,
): boolean {
  if (mode !== "managed") return false;
  if (frontierUnlocked) return false;
  return modelTier(modelValue) === "frontier";
}
