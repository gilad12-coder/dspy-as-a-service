"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  AlignLeft,
  ChevronRight,
  GitCompare,
  Hash,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  displayCandidateId,
  type MinibatchEntry,
  type PerExampleScore,
  type PredictionValue,
  type RejectedNode,
  type TrajectoryNode,
  type ValsetRow,
} from "../lib/types";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { HelpTip } from "@/shared/ui/help-tip";
import { RecordedChatTranscript, type ChatMessage } from "./RecordedChat";
import { UserBubble } from "@/shared/ui/agent/user-bubble";
import { AgentBubble } from "@/shared/ui/agent/agent-bubble";
import { Carousel, ToolCallRow, ToolHeader, ToolsCarousel } from "@/features/agent-panel";
import type { AgentMessage, AgentToolCall } from "@/shared/ui/agent";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/shared/ui/primitives/sheet";

const PARETO_PASS = "#8a9a5b";
const PARETO_FAIL = "#a85a3b";
const PARETO_PASS_BG = "rgba(138, 154, 91, 0.28)";
const PARETO_FAIL_BG = "rgba(168, 90, 59, 0.22)";

const DIFF_VIEW_KEY = "skynet:trajectory:prompt-view";

// Per-node map of tool name → optimized description, derived from the selected
// candidate's own ReAct overlay (see deriveToolDescriptions). Lets the shared
// tools carousel describe *this* run's tools — any optimized agent, not just the
// platform's catalogued generalist tools — when rendering an ``allowed_tools``
// roster. Empty by default, so without a provider the carousel falls back to the
// static catalog unchanged.
const ToolDescriptionsContext = createContext<Record<string, string>>({});

// Run-level tool name → approval severity, captured from the source MCP's tool
// annotations and persisted on the result's ``react_overlay.tool_severities``
// (see the optimizer's severity capture). Unlike descriptions this is invariant
// across candidates — severity is a property of the underlying tool, not the
// optimization step — so the whole drawer shares one map, looked up by name in
// ReactToolCard. Empty by default, so without a provider ToolHeader falls back
// to its catalog severity unchanged and never fabricates one.
const ToolSeveritiesContext = createContext<Record<string, string>>({});

export type DrawerSelection =
  | { kind: "candidate"; node: TrajectoryNode; parent: TrajectoryNode | null }
  | { kind: "rejected"; ghost: RejectedNode; parent: TrajectoryNode | null }
  | null;

export interface TrajectoryDrawerProps {
  selection: DrawerSelection;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  valsetRows: ValsetRow[];
  minibatch: MinibatchEntry[];
  // Per-candidate predictions on each valset example. Keyed by candidate_id →
  // example_id → prediction text. Sparse until each candidate's full eval
  // sweep arrives over the wire.
  valsetOutputs: Map<string, Map<string, PredictionValue>>;
  // Tool name → approval severity from the run's persisted react_overlay, so the
  // drawer's tool cards show the same captured severity as the Code tab.
  toolSeverities?: Record<string, string>;
}

interface NodeView {
  kind: "accepted" | "rejected";
  rawId: string;
  displayId: string;
  iteration: number | null;
  score: number;
  scoreMode: "valset" | "minibatch";
  scoreN: number;
  parentScoreOnMinibatch: number | null;
  parentId: string | null;
  prompt: Record<string, string>;
  parentPrompt: Record<string, string>;
  // Populated for accepted candidates (full valset).
  perExample: PerExampleScore[];
}

function toNodeView(selection: NonNullable<DrawerSelection>): NodeView {
  if (selection.kind === "candidate") {
    const node = selection.node;
    return {
      kind: "accepted",
      rawId: node.candidate_id,
      displayId: displayCandidateId(node.candidate_id),
      iteration: node.iteration,
      score: node.score,
      scoreMode: "valset",
      scoreN: node.per_example.length,
      parentScoreOnMinibatch: null,
      parentId: node.parent_id,
      prompt: node.prompt,
      parentPrompt: selection.parent?.prompt ?? {},
      perExample: node.per_example,
    };
  }
  const ghost = selection.ghost;
  return {
    kind: "rejected",
    rawId: ghost.rejection_id,
    displayId: String(ghost.iteration),
    iteration: ghost.iteration,
    score: ghost.proposal_score,
    scoreMode: "minibatch",
    scoreN: ghost.subsample_size,
    parentScoreOnMinibatch: ghost.parent_score,
    parentId: ghost.parent_id,
    prompt: ghost.proposal_prompt,
    parentPrompt: ghost.parent_prompt,
    perExample: [],
  };
}

export function TrajectoryDrawer({
  selection,
  open,
  onOpenChange,
  valsetRows,
  minibatch,
  valsetOutputs,
  toolSeverities,
}: TrajectoryDrawerProps) {
  if (selection === null) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent
          side="right"
          className="w-full sm:max-w-md md:max-w-[min(520px,92vw)] overflow-hidden bg-[#fbf8f3]"
        >
          <SheetHeader>
            <SheetTitle className="text-base">{TERMS.candidate}</SheetTitle>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    );
  }

  const view = toNodeView(selection);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full sm:max-w-md md:max-w-[min(520px,92vw)] overflow-hidden bg-[#fbf8f3] flex flex-col"
      >
        <NodeBody
          view={view}
          valsetRows={valsetRows}
          minibatch={minibatch}
          valsetOutputs={valsetOutputs}
          toolSeverities={toolSeverities}
        />
      </SheetContent>
    </Sheet>
  );
}

function NodeBody({
  view,
  valsetRows,
  minibatch,
  valsetOutputs,
  toolSeverities,
}: {
  view: NodeView;
  valsetRows: ValsetRow[];
  minibatch: MinibatchEntry[];
  valsetOutputs: Map<string, Map<string, PredictionValue>>;
  toolSeverities?: Record<string, string>;
}) {
  const [pinnedExampleId, setPinnedExampleId] = useState<string | null>(null);
  const [promptViewMode, setPromptViewMode] = usePromptView();

  useEffect(() => {
    setPinnedExampleId(null);
  }, [view.rawId]);

  const promptEntries = useMemo(() => Object.entries(view.prompt), [view.prompt]);
  const toolDescriptions = useMemo(() => deriveToolDescriptions(view.prompt), [view.prompt]);
  const valsetById = useMemo(() => {
    const m = new Map<string, ValsetRow>();
    for (const row of valsetRows) m.set(row.id, row);
    return m;
  }, [valsetRows]);
  const predictionsForView = useMemo(
    () =>
      view.kind === "accepted"
        ? (valsetOutputs.get(view.rawId) ?? new Map<string, PredictionValue>())
        : new Map<string, PredictionValue>(),
    [valsetOutputs, view.kind, view.rawId],
  );

  const headerTitle =
    view.kind === "accepted"
      ? formatMsg("trajectory.node.header.accepted_title", { id: view.displayId })
      : msg("trajectory.node.header.rejected_title");
  const scoreLabel =
    view.scoreMode === "valset"
      ? msg("trajectory.node.header.label.score_valset")
      : msg("trajectory.node.header.label.score_minibatch");
  const examplesSub = formatMsg("trajectory.node.header.sub.examples", {
    n: view.scoreN,
  });

  return (
    <ToolSeveritiesContext.Provider value={toolSeverities ?? {}}>
    <ToolDescriptionsContext.Provider value={toolDescriptions}>
      <SheetHeader className="border-b border-border/30">
        <SheetTitle className="flex items-center gap-2 text-base">
          {view.kind === "rejected" ? (
            <XCircle className="size-4 text-[#a85a3b]" aria-hidden="true" />
          ) : null}
          <span>{headerTitle}</span>
        </SheetTitle>
        <SheetDescription asChild>
          <div className="mt-1.5 flex items-stretch rounded-md border border-border/40 bg-background/50 overflow-hidden">
            {view.iteration !== null ? (
              <StatTile
                label={msg("trajectory.node.header.label.iteration")}
                value={String(view.iteration)}
                icon={Hash}
              />
            ) : null}
            <StatTile
              label={scoreLabel}
              value={view.score.toFixed(2)}
              sub={examplesSub}
              tone={view.kind === "rejected" ? "rejected" : "accepted"}
              emphasis
            />
            {view.parentScoreOnMinibatch !== null ? (
              <StatTile
                label={msg("trajectory.node.header.label.parent_score")}
                value={view.parentScoreOnMinibatch.toFixed(2)}
                tone="muted"
              />
            ) : null}
          </div>
        </SheetDescription>
      </SheetHeader>

      <div className="flex-1 overflow-y-auto px-4 pb-6 space-y-5">
        {view.kind === "accepted" && view.perExample.length > 0 ? (
          <Section
            title={msg("trajectory.node.section.score_detail.valset")}
            info={msg("trajectory.detail.pareto_title.explain")}
          >
            <ParetoGridSection
              examples={view.perExample}
              pinnedId={pinnedExampleId}
              onPin={(id) => setPinnedExampleId((prev) => (prev === id ? null : id))}
              valsetById={valsetById}
              predictionsForCandidate={predictionsForView}
            />
          </Section>
        ) : null}

        {view.kind === "rejected" || promptEntries.length > 0 ? (
          <Section
            title={
              view.kind === "rejected"
                ? msg("trajectory.drawer.rejected.prompt_title")
                : msg("trajectory.node.section.prompt")
            }
            info={
              view.kind === "rejected"
                ? msg("trajectory.drawer.rejected.prompt_title.explain")
                : msg("trajectory.node.section.prompt.explain")
            }
            action={
              promptEntries.length > 0 && Object.keys(view.parentPrompt).length > 0 ? (
                <PromptViewToggle view={promptViewMode} onChange={setPromptViewMode} />
              ) : undefined
            }
          >
            {promptEntries.length === 0 ? (
              <EmptyHint text={msg("trajectory.drawer.rejected.prompt_unavailable")} />
            ) : (
              <div className="space-y-2">
                {promptEntries.map(([predictor, prompt]) => (
                  <PromptEntry
                    key={predictor}
                    prompt={prompt}
                    parentPrompt={view.parentPrompt[predictor] ?? ""}
                    mode={promptViewMode}
                    hasParent={Object.keys(view.parentPrompt).length > 0}
                  />
                ))}
              </div>
            )}
          </Section>
        ) : null}

        <Section
          title={msg("trajectory.drawer.section.minibatch")}
          info={msg("trajectory.drawer.section.minibatch.explain")}
        >
          <MinibatchPanel entries={minibatch} valsetRows={valsetRows} iteration={view.iteration} />
        </Section>
      </div>
    </ToolDescriptionsContext.Provider>
    </ToolSeveritiesContext.Provider>
  );
}

function EmptyHint({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-dashed border-border/50 bg-background/40 px-3 py-2.5 text-[11px] text-muted-foreground">
      {text}
    </div>
  );
}

const TONE_COLORS = {
  accepted: { text: "#4a5a23", bg: "rgba(124, 139, 90, 0.14)" },
  rejected: { text: "#6e2e16", bg: "rgba(178, 107, 74, 0.16)" },
  muted: { text: "rgba(28, 22, 18, 0.62)", bg: "rgba(28, 22, 18, 0.05)" },
} as const;

type Tone = keyof typeof TONE_COLORS;

function StatTile({
  label,
  value,
  sub,
  icon: Icon,
  tone = "muted",
  emphasis = false,
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  tone?: Tone;
  emphasis?: boolean;
}) {
  const palette = TONE_COLORS[tone];
  return (
    <div className="flex-1 min-w-0 px-2.5 py-1.5 border-s border-border/30 first:border-s-0">
      <div className="flex items-center gap-1 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground/85">
        {Icon !== undefined ? <Icon className="size-2.5 opacity-70" aria-hidden={true} /> : null}
        <span className="truncate">{label}</span>
      </div>
      <div
        className={cn(
          "mt-0.5 font-mono tabular-nums leading-tight",
          emphasis ? "text-[15px] font-semibold" : "text-[13px] font-medium",
        )}
        style={{ color: palette.text }}
      >
        {value}
      </div>
      {sub !== undefined ? (
        <div className="mt-0.5 truncate text-[9px] tabular-nums text-muted-foreground/70">
          {sub}
        </div>
      ) : null}
      {emphasis ? (
        <div
          aria-hidden="true"
          className="mt-1 h-px w-6 rounded-full"
          style={{ background: palette.bg }}
        />
      ) : null}
    </div>
  );
}


function MinibatchPanel({
  entries,
  valsetRows,
  iteration,
}: {
  entries: MinibatchEntry[];
  valsetRows: ValsetRow[];
  // GEPA iteration of the selected node. When set, the panel shows only the
  // feedback events emitted while that iteration's propose() was running —
  // the 3-ish entries that actually informed accepting / rejecting the node.
  // null (seed candidate, or backend versions before iteration plumbing)
  // falls back to the run-wide most-recent view.
  iteration: number | null;
}) {
  const RECENT = 8;
  const recent = useMemo(() => {
    const filtered =
      iteration === null ? entries : entries.filter((e) => e.iteration === iteration);
    return filtered.slice(-RECENT).reverse();
  }, [entries, iteration]);
  const valsetById = useMemo(() => {
    const m = new Map<string, ValsetRow>();
    for (const row of valsetRows) m.set(row.id, row);
    return m;
  }, [valsetRows]);

  if (recent.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border/50 bg-background/40 px-3 py-3 text-[11px] text-muted-foreground">
        {msg("trajectory.minibatch.no_data")}
      </div>
    );
  }
  return (
    <ol className="space-y-2.5">
      {recent.map((entry) => (
        <MinibatchEntryCard
          key={entry.sequence}
          entry={entry}
          valsetRow={valsetById.get(entry.example_id) ?? null}
        />
      ))}
    </ol>
  );
}

function MinibatchEntryCard({
  entry,
  valsetRow,
}: {
  entry: MinibatchEntry;
  valsetRow: ValsetRow | null;
}) {
  const passed = entry.score > 0;
  // Render the example exactly like a validation example (ValsetFieldGroups):
  // inputs with empty fields dropped, the candidate answer as a ReAct chat turn
  // with replay metadata folded into a disclosure, and the expected answer. The
  // subsample id is a valset id, so valsetRow is normally present; when it isn't
  // (rare), fall back to just the candidate answer rendered the same way.
  const fallbackPrediction = useMemo(
    () => predictionToEntries(entry.prediction),
    [entry.prediction],
  );

  return (
    <li className="overflow-hidden rounded-lg border border-[#DDD4C8]/60 bg-background/70 shadow-[0_1px_2px_rgba(28,22,18,0.02)]">
      <div className="flex items-center justify-between gap-2 border-b border-border/30 bg-[#F8F4EF]/60 px-3 py-2">
        <div className="flex items-center gap-2 text-[11px]">
          <StatusChip passed={passed} />
          <span className="font-mono text-[10px] text-muted-foreground" dir="ltr">
            #{entry.example_id}
          </span>
        </div>
        <HelpTip text={msg("trajectory.minibatch.score_label.explain")}>
          <span className="inline-flex items-baseline gap-1.5">
            <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
              {msg("trajectory.minibatch.score_label")}
            </span>
            <span
              className={cn(
                "font-mono tabular-nums text-[11px]",
                passed ? "font-semibold text-foreground" : "text-muted-foreground",
              )}
            >
              {entry.score.toFixed(2)}
            </span>
          </span>
        </HelpTip>
      </div>

      <div className="space-y-2.5 px-3 pb-3 pt-2.5">
        {valsetRow !== null ? (
          <ValsetFieldGroups row={valsetRow} prediction={entry.prediction} />
        ) : fallbackPrediction !== null && fallbackPrediction.length > 0 ? (
          <AgentTurnGroup
            label={msg("trajectory.pareto.cell.prediction_label")}
            info={msg("trajectory.pareto.cell.prediction_label.explain")}
            entries={fallbackPrediction}
          />
        ) : null}

        {entry.feedback.length > 0 ? <FeedbackBlock body={entry.feedback} /> : null}
      </div>
    </li>
  );
}

function StatusChip({ passed }: { passed: boolean }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-[#DDD4C8]/70 bg-background/80 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-foreground/75">
      {passed ? msg("trajectory.minibatch.pass_label") : msg("trajectory.minibatch.fail_label")}
    </span>
  );
}

// Recursive-descent parser for Python literal syntax (`repr` output). Example
// field values (chat_history, wizard_state, expected steps/state) are serialized
// backend-side with str()/repr(), so they arrive as Python literals — single
// quotes, True/False/None, \' escapes — which JSON.parse rejects. Handles
// dict/list/tuple/str/number/bool/None and keyword-only constructor reprs
// (`Prediction(...)`, `History(...)`); throws on anything it can't parse so the
// caller falls back to rendering the raw string.
function parsePythonLiteral(input: string): unknown {
  let i = 0;
  const n = input.length;

  const skipWs = (): void => {
    while (i < n && /\s/.test(input.charAt(i))) i += 1;
  };

  const parseString = (): string => {
    const quote = input.charAt(i);
    i += 1;
    let out = "";
    while (i < n) {
      const ch = input.charAt(i);
      if (ch === "\\") {
        const next = input.charAt(i + 1);
        if (next === "n") out += "\n";
        else if (next === "t") out += "\t";
        else if (next === "r") out += "\r";
        else out += next; // covers \\ \' \" and any other escaped char
        i += 2;
        continue;
      }
      if (ch === quote) {
        i += 1;
        return out;
      }
      out += ch;
      i += 1;
    }
    throw new Error("unterminated string");
  };

  const parseNumber = (): number => {
    const start = i;
    if (input.charAt(i) === "+" || input.charAt(i) === "-") i += 1;
    while (i < n && /[0-9.eE+-]/.test(input.charAt(i))) i += 1;
    const text = input.slice(start, i);
    const num = Number(text);
    if (text.length === 0 || Number.isNaN(num)) throw new Error("bad number");
    return num;
  };

  const parseValue = (): unknown => {
    skipWs();
    if (i >= n) throw new Error("unexpected end");
    const c = input.charAt(i);
    if (c === "{") return parseDict();
    if (c === "[") return parseSeq("]");
    if (c === "(") return parseSeq(")");
    if (c === "'" || c === '"') return parseString();
    if (input.startsWith("True", i)) {
      i += 4;
      return true;
    }
    if (input.startsWith("False", i)) {
      i += 5;
      return false;
    }
    if (input.startsWith("None", i)) {
      i += 4;
      return null;
    }
    if (/[A-Za-z_]/.test(c)) return parseCall();
    return parseNumber();
  };

  function parseDict(): Record<string, unknown> {
    i += 1;
    const obj: Record<string, unknown> = {};
    skipWs();
    if (input.charAt(i) === "}") {
      i += 1;
      return obj;
    }
    while (i < n) {
      const key = parseValue();
      skipWs();
      if (input.charAt(i) !== ":") throw new Error("expected colon");
      i += 1;
      const val = parseValue();
      obj[typeof key === "string" ? key : String(key)] = val;
      skipWs();
      const ch = input.charAt(i);
      if (ch === ",") {
        i += 1;
        skipWs();
        if (input.charAt(i) === "}") {
          i += 1;
          return obj;
        }
        continue;
      }
      if (ch === "}") {
        i += 1;
        return obj;
      }
      throw new Error("expected comma or brace");
    }
    throw new Error("unterminated dict");
  }

  function parseSeq(close: string): unknown[] {
    i += 1;
    const arr: unknown[] = [];
    skipWs();
    if (input.charAt(i) === close) {
      i += 1;
      return arr;
    }
    while (i < n) {
      arr.push(parseValue());
      skipWs();
      const ch = input.charAt(i);
      if (ch === ",") {
        i += 1;
        skipWs();
        if (input.charAt(i) === close) {
          i += 1;
          return arr;
        }
        continue;
      }
      if (ch === close) {
        i += 1;
        return arr;
      }
      throw new Error("expected comma or close");
    }
    throw new Error("unterminated sequence");
  }

  // Constructor-style reprs — `Prediction(assistant_message=…, history=…)` or
  // `History(messages=[…])` — arrive when a field value is itself a DSPy object.
  // Treat keyword arguments as a plain object so the value renders as a tree.
  // Only keyword arguments are supported; a positional arg throws so the caller
  // falls back to rendering raw text.
  function parseCall(): Record<string, unknown> {
    while (i < n && /[A-Za-z0-9_.]/.test(input.charAt(i))) i += 1;
    skipWs();
    if (input.charAt(i) !== "(") throw new Error("expected call args");
    i += 1;
    const obj: Record<string, unknown> = {};
    skipWs();
    if (input.charAt(i) === ")") {
      i += 1;
      return obj;
    }
    while (i < n) {
      skipWs();
      const keyStart = i;
      while (i < n && /[A-Za-z0-9_]/.test(input.charAt(i))) i += 1;
      const key = input.slice(keyStart, i);
      skipWs();
      if (key.length === 0 || input.charAt(i) !== "=") {
        throw new Error("expected keyword argument");
      }
      i += 1;
      obj[key] = parseValue();
      skipWs();
      const ch = input.charAt(i);
      if (ch === ",") {
        i += 1;
        skipWs();
        if (input.charAt(i) === ")") {
          i += 1;
          return obj;
        }
        continue;
      }
      if (ch === ")") {
        i += 1;
        return obj;
      }
      throw new Error("expected comma or close paren");
    }
    throw new Error("unterminated call");
  }

  const result = parseValue();
  skipWs();
  if (i !== n) throw new Error("trailing content");
  return result;
}

// Detect a container value and parse it for the structured renderer. Tries JSON
// first (new runs / genuinely-JSON fields), then Python-literal repr (existing
// runs). A leading `Identifier(` is treated as a constructor repr (e.g.
// `History(messages=[…])`) so it parses too. Returns undefined for non-containers
// (plain text, numbers, quoted strings) so the caller renders them as text.
function parseStructuredValue(value: string): unknown {
  const trimmed = value.trim();
  const first = trimmed.charAt(0);
  const isCall = /^[A-Za-z_][\w.]*\(/.test(trimmed);
  if (first !== "{" && first !== "[" && first !== "(" && !isCall) return undefined;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (parsed !== null && typeof parsed === "object") return parsed;
  } catch {
    // Not JSON — fall through to Python-literal parsing.
  }
  try {
    const parsed = parsePythonLiteral(trimmed);
    if (parsed !== null && typeof parsed === "object") return parsed;
  } catch {
    return undefined;
  }
  return undefined;
}

// True for a field that carries nothing worth a row — a blank string or an
// empty container (`[]`, `{}`). These add only noise next to the meaningful
// fields (e.g. an empty `chat_history` sitting beside the user's message), so
// the field group drops them.
function isEmptyFieldValue(value: string): boolean {
  if (value.trim().length === 0) return true;
  const structured = parseStructuredValue(value);
  if (Array.isArray(structured)) return structured.length === 0;
  if (structured !== null && typeof structured === "object") {
    return Object.keys(structured).length === 0;
  }
  return false;
}

// Replay-provenance fields the backend keeps on each example for input context
// and gate scoring, but which add no signal when reviewing an answer: the
// turn-start wizard snapshot and the metric's before/after state. New runs omit
// them server-side (see _HIDDEN_VALSET_FIELDS); this also strips them from runs
// recorded before that change, so both validation and mini-batch cards stay
// focused on inputs, the answer and the prediction.
const HIDDEN_FIELD_KEYS: ReadonlySet<string> = new Set([
  "wizard_state",
  "state_before",
  "state_after",
]);

function isHiddenField(key: string): boolean {
  return HIDDEN_FIELD_KEYS.has(key.trim().toLowerCase());
}

// True for a field that lists tool names (e.g. ``allowed_tools``), which reads
// better as a horizontal chip carousel than a tall numbered list.
function isToolListField(fieldName: string | undefined): boolean {
  if (fieldName === undefined) return false;
  const n = fieldName.trim().toLowerCase();
  return n === "allowed_tools" || n === "tools";
}

// Parse a value into a non-empty list of strings, or null when it isn't one.
function parseStringArray(value: string): string[] | null {
  const parsed = parseStructuredValue(value);
  if (!Array.isArray(parsed) || parsed.length === 0) return null;
  if (!parsed.every((item) => typeof item === "string")) return null;
  return parsed as string[];
}

const CHAT_ROLES: ReadonlySet<string> = new Set(["user", "assistant", "system", "tool"]);

// Normalize an already-parsed value into chat turns, or null when it isn't a
// non-empty list where *every* element is a chat-role message object (so
// arbitrary object arrays don't false-trigger the chat renderer). Non-string
// content (multi-part messages) is JSON-encoded so a turn always carries text.
function normalizeChatMessages(parsed: unknown): ChatMessage[] | null {
  if (!Array.isArray(parsed) || parsed.length === 0) return null;
  const messages: ChatMessage[] = [];
  for (const item of parsed) {
    if (item === null || typeof item !== "object" || Array.isArray(item)) return null;
    const rec = item as Record<string, unknown>;
    const role = typeof rec.role === "string" ? rec.role.toLowerCase() : null;
    if (role === null || !CHAT_ROLES.has(role)) return null;
    const raw = rec.content;
    const content = typeof raw === "string" ? raw : raw == null ? "" : JSON.stringify(raw);
    messages.push({ role: role as ChatMessage["role"], content });
  }
  return messages;
}

// Recover the complete leading elements of a JSON array truncated mid-value —
// the server caps captured strings at MINIBATCH_PREDICTION_CHAR_CAP (see
// trajectory.py), so a long chat_history arrives as invalid JSON. Scans for the
// last position where a depth-1 element closed, slices there, and re-closes the
// array; quote/escape aware so braces inside string content don't miscount.
// Returns null when nothing complete remains or the prefix still won't parse.
function recoverJsonArrayPrefix(value: string): unknown[] | null {
  const s = value.trim();
  if (s.charAt(0) !== "[") return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  let lastComplete = -1;
  for (let i = 0; i < s.length; i += 1) {
    const c = s.charAt(i);
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "[" || c === "{") depth += 1;
    else if (c === "]" || c === "}") {
      depth -= 1;
      if (depth === 1) lastComplete = i + 1;
    }
  }
  if (lastComplete === -1) return null;
  try {
    const parsed: unknown = JSON.parse(`${s.slice(0, lastComplete)}]`);
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

// Detect a captured chat transcript so it renders as a conversation instead of a
// raw key/value tree. A chat_history field arrives as a serialized list of
// `{role, content}` turns (JSON or Python-literal repr). When the value was
// server-truncated mid-JSON, the complete leading turns are recovered so the
// conversation still bubbles rather than dropping to a raw string.
function parseChatHistory(value: string): ChatMessage[] | null {
  const strict = normalizeChatMessages(parseStructuredValue(value));
  if (strict !== null) return strict;
  return normalizeChatMessages(recoverJsonArrayPrefix(value));
}

type HistoryTrace =
  | { kind: "chat"; messages: ChatMessage[] }
  | { kind: "turns"; turns: Array<Array<[string, string]>> };

function safeJsonString(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return String(value);
  }
}

// Detect a captured agent history: a `{messages: [...]}` object (a serialized
// dspy History — JSON or `History(messages=[…])` repr). A history of plain
// `{role, content}` turns is a transcript; otherwise each message is a per-turn
// field map (user_message, chat_history, next_thought, tool_calls, …) the caller
// renders as one turn. Non-string field values (nested tool_calls/results) are
// JSON-encoded so they flow through the string-keyed field renderer. Returns
// null when the value isn't a messages-wrapped history.
function parseHistoryTrace(value: string): HistoryTrace | null {
  const parsed = parseStructuredValue(value);
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const messages = (parsed as Record<string, unknown>).messages;
  if (!Array.isArray(messages) || messages.length === 0) return null;
  const chat = normalizeChatMessages(messages);
  if (chat !== null) return { kind: "chat", messages: chat };
  const turns: Array<Array<[string, string]>> = [];
  for (const m of messages) {
    if (m === null || typeof m !== "object" || Array.isArray(m)) return null;
    turns.push(
      Object.entries(m as Record<string, unknown>).map(
        ([k, v]): [string, string] => [k, typeof v === "string" ? v : safeJsonString(v)],
      ),
    );
  }
  return { kind: "turns", turns };
}

const JSON_VIEW_MAX_DEPTH = 8;
const JSON_INLINE_STRING_MAX = 52;

// Warm, on-palette accents for value types (keeps the data viewer cohesive with
// the rest of the drawer rather than introducing loud syntax colors).
const JSON_NUMBER_COLOR = "#5f6b4f";
const JSON_BOOL_COLOR = "#7c6f5a";

// A structured value renders as a self-contained LTR "data viewer" so its
// layout stays consistent regardless of the RTL drawer around it: monospace
// keys with left indent guides, proportional leaf strings with dir="auto" so
// Hebrew and English content each read in their natural direction.
function JsonView({ value }: { value: unknown }) {
  return (
    <div dir="ltr" className="font-mono text-[10.5px] leading-relaxed text-foreground/80">
      <JsonNode value={value} depth={0} />
    </div>
  );
}

function JsonRaw({ value }: { value: unknown }) {
  let text: string;
  try {
    text = JSON.stringify(value, null, 2);
  } catch {
    text = String(value);
  }
  return (
    <pre className="whitespace-pre-wrap leading-snug" style={{ wordBreak: "break-word" }}>
      {text}
    </pre>
  );
}

function JsonScalar({ value }: { value: number | boolean | null }) {
  if (value === null) {
    return <span className="italic text-muted-foreground/55">{"null"}</span>;
  }
  if (typeof value === "boolean") {
    return (
      <span className="italic" style={{ color: JSON_BOOL_COLOR }}>
        {String(value)}
      </span>
    );
  }
  return (
    <span className="tabular-nums" style={{ color: JSON_NUMBER_COLOR }}>
      {String(value)}
    </span>
  );
}

function JsonString({ value }: { value: string }) {
  if (value.length === 0) {
    return (
      <span className="italic text-muted-foreground/50">{msg("trajectory.json.empty_value")}</span>
    );
  }
  return (
    <span
      className="whitespace-pre-wrap font-sans text-foreground/90"
      dir="auto"
      style={{ wordBreak: "break-word" }}
    >
      {value}
    </span>
  );
}

function JsonNode({ value, depth }: { value: unknown; depth: number }) {
  if (value === null || typeof value === "number" || typeof value === "boolean") {
    return <JsonScalar value={value} />;
  }
  if (typeof value === "string") {
    return <JsonString value={value} />;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-muted-foreground/45">{"[ ]"}</span>;
    }
    if (depth >= JSON_VIEW_MAX_DEPTH) return <JsonRaw value={value} />;
    return (
      <div className={cn("space-y-1", depth > 0 && "border-l border-border/40 pl-2.5")}>
        {value.map((item, idx) => (
          <div key={idx} className="flex gap-2">
            <span className="shrink-0 select-none pt-px text-[9px] tabular-nums text-muted-foreground/40">
              {idx}
            </span>
            <div className="min-w-0 flex-1">
              <JsonNode value={item} depth={depth + 1} />
            </div>
          </div>
        ))}
      </div>
    );
  }
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) {
    return <span className="text-muted-foreground/45">{"{ }"}</span>;
  }
  if (depth >= JSON_VIEW_MAX_DEPTH) return <JsonRaw value={value} />;
  return (
    <div className={cn("space-y-0.5", depth > 0 && "border-l border-border/40 pl-2.5")}>
      {entries.map(([k, v]) => (
        <JsonField key={k} name={k} value={v} depth={depth} />
      ))}
    </div>
  );
}

// Scalars and short single-line strings sit inline after the key (`role: user`);
// containers and long/multiline strings drop to their own block below it.
function JsonField({ name, value, depth }: { name: string; value: unknown; depth: number }) {
  const isContainer = value !== null && typeof value === "object";
  const isString = typeof value === "string";
  // A nested chat transcript (e.g. a `chat_history` string buried in a parsed
  // `History(...)`) renders as the recorded conversation rather than a raw JSON
  // blob, so an embedded chat reads as chat bubbles.
  const chat = isString ? parseChatHistory(value as string) : null;
  const inline =
    !isContainer &&
    chat === null &&
    (!isString ||
      ((value as string).length <= JSON_INLINE_STRING_MAX && !(value as string).includes("\n")));
  const key = (
    <span className="text-muted-foreground/70">
      {name}
      <span className="text-muted-foreground/35">{":"}</span>
    </span>
  );
  if (chat !== null) {
    return (
      <div>
        {key}
        <div className="mt-0.5">
          <RecordedChatTranscript messages={chat} />
        </div>
      </div>
    );
  }
  if (inline) {
    return (
      <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
        {key}
        <JsonNode value={value} depth={depth + 1} />
      </div>
    );
  }
  return (
    <div>
      {key}
      <div className="mt-0.5">
        <JsonNode value={value} depth={depth + 1} />
      </div>
    </div>
  );
}

function FieldValue({ value }: { value: string }) {
  const structured = useMemo(() => parseStructuredValue(value), [value]);
  if (structured !== undefined) {
    return <JsonView value={structured} />;
  }
  return (
    <div
      className="whitespace-pre-wrap text-[11px] leading-snug text-foreground/90"
      dir="auto"
      style={{ wordBreak: "break-word" }}
    >
      {value}
    </div>
  );
}

// True for input fields that carry a user's chat turn (e.g. ``user_message``).
// Such a turn reads better as the app's chat bubble than as a boxed data field;
// requiring ``user`` in the name keeps system/assistant-authored fields out.
function isUserMessageField(fieldName: string | undefined): boolean {
  if (fieldName === undefined) return false;
  const n = fieldName.trim().toLowerCase();
  return n.includes("user") && /message|msg|input|prompt|query/.test(n);
}

// Mirror of isUserMessageField for assistant/agent-authored turns (e.g.
// ``assistant_message``), rendered as the agent reply bubble.
function isAssistantMessageField(fieldName: string | undefined): boolean {
  if (fieldName === undefined) return false;
  const n = fieldName.trim().toLowerCase();
  return /assistant|agent/.test(n) && /message|msg|reply|response/.test(n);
}

function FieldRow({ fieldName, value }: { fieldName?: string; value: string }) {
  // A captured chat_history renders as a full-width recorded transcript — its
  // own header carries the label, so it breaks out of the bordered field cell
  // rather than sitting boxed (and inset) inside it.
  const chat = useMemo(() => parseChatHistory(value), [value]);
  // A captured agent ``history`` ({messages: […]}) breaks out the same way and
  // renders as a recorded conversation rather than a raw key/value tree.
  const trace = useMemo(() => parseHistoryTrace(value), [value]);
  const toolDescriptions = useContext(ToolDescriptionsContext);
  if (chat !== null) {
    return <RecordedChatTranscript messages={chat} />;
  }
  if (trace !== null) {
    return trace.kind === "chat" ? (
      <RecordedChatTranscript messages={trace.messages} />
    ) : (
      <HistoryTraceView turns={trace.turns} />
    );
  }
  // A user-message turn renders as the app's chat bubble — same look as the
  // live conversation and the recorded transcript — instead of a boxed field,
  // so a captured turn reads as a message rather than raw data. Read-only here:
  // there's nothing live to edit-and-resend.
  if (value.trim().length > 0 && isUserMessageField(fieldName)) {
    return <UserBubble content={value} editable={false} />;
  }
  // An assistant-message turn renders as the agent reply bubble, end-aligned and
  // markdown-rendered, exactly as in the recorded transcript.
  if (value.trim().length > 0 && isAssistantMessageField(fieldName)) {
    const agentMsg: AgentMessage = { role: "assistant", content: value };
    return (
      <div className="flex justify-end">
        <AgentBubble msg={agentMsg} className="max-w-full" />
      </div>
    );
  }
  // A tool-name list (e.g. allowed_tools) reads better as the shared tool
  // carousel than a long numbered tree — same component the agent tour uses,
  // here driven by the row's real roster instead of the curated default.
  if (isToolListField(fieldName)) {
    const tools = parseStringArray(value);
    if (tools !== null) {
      return (
        <div className="rounded border border-[#DDD4C8]/40 bg-background/80 px-1 py-1">
          <ToolsCarousel
            tools={tools}
            descriptions={toolDescriptions}
            title={msg("trajectory.pareto.cell.allowed_tools_label")}
            className="w-full"
          />
        </div>
      );
    }
  }
  return (
    <div className="rounded border border-[#DDD4C8]/40 bg-background/80 px-2 py-1.5">
      {fieldName !== undefined ? (
        <div className="mb-0.5 text-[9px] font-mono text-muted-foreground/80" dir="ltr">
          {fieldName}
        </div>
      ) : null}
      <FieldValue value={value} />
    </div>
  );
}

// True for fields carrying a recorded tool-call trajectory (the expected
// answer's ``steps``, or a candidate's ``trajectory``/``tool_calls``), which
// render as the live chat's tool-call rows rather than a raw JSON tree.
function isStepsField(fieldName: string): boolean {
  const n = fieldName.trim().toLowerCase();
  return n === "steps" || n === "trajectory" || n === "tool_calls";
}

const TOOL_CALL_STATUSES: ReadonlySet<string> = new Set(["running", "done", "error"]);

// Map a serialized step array onto the AgentToolCall shape the live chat's
// ToolCallRow renders, so a recorded turn's tool calls look identical to the
// streaming agent. The value is a JSON or Python-repr list of step objects
// (``{tool, reason, status, payload:{arguments, result}, …}``). Returns null
// when it isn't a non-empty list with at least one named tool — e.g. a
// length-capped (truncated) value that no longer parses.
function parseToolCalls(value: string): AgentToolCall[] | null {
  const parsed = parseStructuredValue(value);
  if (!Array.isArray(parsed) || parsed.length === 0) return null;
  const calls: AgentToolCall[] = [];
  parsed.forEach((item, idx) => {
    if (item === null || typeof item !== "object" || Array.isArray(item)) return;
    const rec = item as Record<string, unknown>;
    if (typeof rec.tool !== "string" || rec.tool.length === 0) return;
    const status =
      typeof rec.status === "string" && TOOL_CALL_STATUSES.has(rec.status)
        ? (rec.status as AgentToolCall["status"])
        : "done";
    const payload =
      rec.payload !== null && typeof rec.payload === "object" && !Array.isArray(rec.payload)
        ? (rec.payload as Record<string, unknown>)
        : undefined;
    calls.push({
      id: typeof rec.id === "string" ? rec.id : `step_${idx}`,
      tool: rec.tool,
      reason: typeof rec.reason === "string" ? rec.reason : "",
      status,
      startedAt: typeof rec.startedAt === "number" ? rec.startedAt : 0,
      endedAt: typeof rec.endedAt === "number" ? rec.endedAt : null,
      payload,
    });
  });
  return calls.length > 0 ? calls : null;
}

// Read-only tool-call row for recorded trajectories — same component the live
// agent chat uses, minus the interactive cards (nothing here is actionable).
function renderTrajectoryToolCall(call: AgentToolCall, ctx: { isRetry: boolean }) {
  return <ToolCallRow call={call} isRetry={ctx.isRetry} />;
}

interface AgentTurnParts {
  userMessages: string[];
  assistantMessage: string | null;
  toolCalls: AgentToolCall[] | null;
  transcripts: ChatMessage[][];
  meta: Array<[string, string]>;
}

// Split a turn's field list into the chat-relevant parts (a user/assistant
// message, the tool-call trajectory, any embedded chat transcript) and the
// replay metadata (allowed_tools, tool_schema_hashes, state snapshots, …) that
// belongs in a collapsed disclosure. Lets the expected and actual answers read
// as a conversation so a reviewer compares what the agent said/did against the
// gold turn instead of diffing raw JSON trees.
function splitAgentTurn(entries: Array<[string, string]>): AgentTurnParts {
  const userMessages: string[] = [];
  let assistantMessage: string | null = null;
  let toolCalls: AgentToolCall[] | null = null;
  const transcripts: ChatMessage[][] = [];
  const meta: Array<[string, string]> = [];
  for (const [key, value] of entries) {
    if (isHiddenField(key)) continue;
    if (isEmptyFieldValue(value)) continue;
    const chat = parseChatHistory(value);
    if (chat !== null) {
      transcripts.push(chat);
      continue;
    }
    if (value.trim().length > 0 && assistantMessage === null && isAssistantMessageField(key)) {
      assistantMessage = value;
      continue;
    }
    if (value.trim().length > 0 && isUserMessageField(key)) {
      userMessages.push(value);
      continue;
    }
    if (isStepsField(key)) {
      const calls = parseToolCalls(value);
      if (calls !== null) {
        toolCalls = toolCalls === null ? calls : [...toolCalls, ...calls];
        continue;
      }
    }
    meta.push([key, value]);
  }
  return { userMessages, assistantMessage, toolCalls, transcripts, meta };
}

// Renders a prediction/expected-answer field group as a chat turn: user turns
// and embedded transcripts first, then the assistant reply with its tool-call
// trajectory in one bubble (identical to the live agent), then the replay
// metadata folded into a collapsed disclosure. Falls back to the plain field
// list when nothing chat-shaped is present (e.g. a truncated, unparseable
// prediction) so the raw value still shows.
function turnHasChat(turn: AgentTurnParts): boolean {
  return (
    turn.userMessages.length > 0 ||
    turn.assistantMessage !== null ||
    turn.toolCalls !== null ||
    turn.transcripts.length > 0
  );
}

// The conversation body of a split turn (no section label): user bubbles, any
// embedded transcripts, the assistant reply with its tool-call trajectory, and
// the collapsed replay-metadata disclosure. Shared by the expected/actual
// answer groups and the recorded history trace.
function AgentTurnConversation({ turn }: { turn: AgentTurnParts }) {
  const agentMsg: AgentMessage | null =
    turn.assistantMessage !== null || turn.toolCalls !== null
      ? {
          role: "assistant",
          content: turn.assistantMessage ?? "",
          toolCalls: turn.toolCalls ?? undefined,
        }
      : null;
  return (
    <div className="space-y-2.5">
      {turn.userMessages.map((content, idx) => (
        <UserBubble key={`user_${idx}`} content={content} editable={false} />
      ))}
      {turn.transcripts.map((messages, idx) => (
        <RecordedChatTranscript key={`transcript_${idx}`} messages={messages} />
      ))}
      {agentMsg !== null ? (
        <div className="flex justify-end">
          <AgentBubble
            msg={agentMsg}
            className="max-w-full"
            renderToolCall={renderTrajectoryToolCall}
          />
        </div>
      ) : null}
      {turn.meta.length > 0 ? <AgentTurnMeta entries={turn.meta} /> : null}
    </div>
  );
}

function AgentTurnGroup({
  label,
  info,
  entries,
}: {
  label: string;
  info?: string;
  entries: Array<[string, string]>;
}) {
  const turn = useMemo(() => splitAgentTurn(entries), [entries]);
  if (!turnHasChat(turn)) {
    return <ValsetFieldGroup label={label} info={info} entries={entries} />;
  }
  return (
    <div className="space-y-1">
      <SectionLabel label={label} info={info} />
      <AgentTurnConversation turn={turn} />
    </div>
  );
}

// A captured agent ``history`` ({messages: [turn, …]}) rendered as a recorded
// conversation: each turn's user/assistant bubbles and embedded chat_history,
// with its reasoning/tool fields folded into the per-turn details disclosure
// (the same split as the expected/actual answer turns). Several turns get a
// light per-turn label; a lone turn renders bare.
function HistoryTraceView({ turns }: { turns: Array<Array<[string, string]>> }) {
  return (
    <div className="space-y-3">
      {turns.map((entries, idx) => (
        <HistoryTurnRow key={idx} entries={entries} index={idx} total={turns.length} />
      ))}
    </div>
  );
}

function HistoryTurnRow({
  entries,
  index,
  total,
}: {
  entries: Array<[string, string]>;
  index: number;
  total: number;
}) {
  const turn = useMemo(() => splitAgentTurn(entries), [entries]);
  const label = total > 1 ? formatMsg("trajectory.history.turn", { n: index + 1 }) : null;
  if (!turnHasChat(turn)) {
    return (
      <ValsetFieldGroup
        label={formatMsg("trajectory.history.turn", { n: index + 1 })}
        entries={entries}
      />
    );
  }
  return (
    <div className="space-y-1">
      {label !== null ? <SectionLabel label={label} /> : null}
      <AgentTurnConversation turn={turn} />
    </div>
  );
}

// Collapsed disclosure for a turn's replay metadata — off-conversation fields
// (allowed_tools, tool_schema_hashes, state snapshots) that would otherwise
// drown the chat. Closed by default; expands with the same rotating chevron and
// eased height reveal as the React tool rows so every disclosure in the drawer
// feels identical. Hand-rolled over a native <details> because <details> can't
// animate its open/close or hide its default marker without fighting the UA.
function AgentTurnMeta({ entries }: { entries: Array<[string, string]> }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="overflow-hidden rounded-md border border-[#DDD4C8]/40 bg-background/50">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full cursor-pointer select-none items-center gap-2 px-2.5 py-1.5 text-start text-[10px] font-medium text-muted-foreground/80 transition-colors hover:bg-[#F8F4EF]/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#C8A882]/45"
      >
        {msg("trajectory.pareto.cell.details_label")}
        <ChevronRight
          className={cn(
            "ms-auto size-3 shrink-0 opacity-50 transition-transform duration-200 ease-out motion-reduce:transition-none",
            // Pinned to the inline-end edge (far left in RTL) via ms-auto. Open
            // rotates to point down; collapsed mirrors by direction, unchanged.
            open ? "rotate-90 opacity-80" : "rtl:rotate-180",
          )}
          aria-hidden="true"
        />
      </button>
      <ToolDisclosureBody open={open}>
        <div className="space-y-1 px-2.5 pb-2 pt-0.5">
          {entries.map(([key, value], idx) => (
            <FieldRow
              key={key.length === 0 ? `__raw_${idx}` : key}
              fieldName={key.length > 0 ? key : undefined}
              value={value}
            />
          ))}
        </div>
      </ToolDisclosureBody>
    </div>
  );
}

function FeedbackBlock({ body }: { body: string }) {
  return (
    <div className="space-y-1">
      <HelpTip text={msg("trajectory.minibatch.feedback_label.explain")}>
        <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
          {msg("trajectory.minibatch.feedback_label")}
        </div>
      </HelpTip>
      <div
        className="rounded border border-[#DDD4C8]/40 bg-background/80 px-2.5 py-2 text-[11px] leading-relaxed text-foreground/90 whitespace-pre-wrap"
        dir="auto"
        style={{ wordBreak: "break-word" }}
      >
        {body}
      </div>
    </div>
  );
}

// Parses DSPy-style `Prediction(key='value', key2="value2", …)` reprs into
// [key, value] pairs. Returns null when the input isn't a recognizable
// `ClassName(...)` call so the caller can fall back to rendering raw text.
function parsePredictionFields(raw: string): Array<[string, string]> | null {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return null;
  const m = /^[A-Za-z_][A-Za-z0-9_]*\((.*)\)$/s.exec(trimmed);
  if (m === null) return null;
  const body = m[1] ?? "";
  const pairs: Array<[string, string]> = [];
  let i = 0;
  const n = body.length;
  while (i < n) {
    while (i < n && /\s/.test(body[i] ?? "")) i += 1;
    const keyStart = i;
    while (i < n && /[A-Za-z0-9_]/.test(body[i] ?? "")) i += 1;
    const key = body.slice(keyStart, i);
    // A trailing comma after the final field (`…, history=History(…),`) leaves
    // an empty key on the next loop — stop and keep the fields parsed so far
    // rather than discarding the whole prediction.
    if (key.length === 0) break;
    while (i < n && /\s/.test(body[i] ?? "")) i += 1;
    if (body[i] !== "=") return null;
    i += 1;
    while (i < n && /\s/.test(body[i] ?? "")) i += 1;
    const value = readValue(body, i);
    if (value === null) return null;
    pairs.push([key, value.text]);
    i = value.next;
    while (i < n && /\s/.test(body[i] ?? "")) i += 1;
    if (i < n && body[i] === ",") i += 1;
  }
  return pairs.length > 0 ? pairs : null;
}

// Reduce a candidate prediction to label/value rows for the structured cells.
// Object predictions (current runs) map field→value directly — the same shape
// as ValsetRow.outputs, so nested JSON-string fields render as trees via
// FieldValue. String predictions (legacy runs) are parsed from their Python
// repr; an unparseable repr falls back to a single unlabeled row. Returns null
// when there's nothing to show.
function predictionToEntries(prediction: PredictionValue | null): Array<[string, string]> | null {
  if (prediction === null) return null;
  if (typeof prediction === "object") return Object.entries(prediction);
  const parsed = parsePredictionFields(prediction);
  if (parsed !== null && parsed.length > 0) return parsed;
  return prediction.length > 0 ? [["", prediction]] : null;
}

function readValue(s: string, start: number): { text: string; next: number } | null {
  const ch = s[start];
  if (ch === "'" || ch === '"') {
    let i = start + 1;
    let out = "";
    while (i < s.length) {
      const c = s[i] ?? "";
      if (c === "\\" && i + 1 < s.length) {
        const next = s[i + 1] ?? "";
        out += next === "n" ? "\n" : next === "t" ? "\t" : next === "r" ? "\r" : next;
        i += 2;
        continue;
      }
      if (c === ch) return { text: out, next: i + 1 };
      out += c;
      i += 1;
    }
    return null;
  }
  // Unquoted value (number, None, True, nested call) — read until top-level
  // comma; balance brackets so nested calls don't terminate early.
  let depth = 0;
  let i = start;
  while (i < s.length) {
    const c = s[i] ?? "";
    if (depth === 0 && c === ",") break;
    if (c === "(" || c === "[" || c === "{") depth += 1;
    else if (c === ")" || c === "]" || c === "}") depth -= 1;
    i += 1;
  }
  const text = s.slice(start, i).trim();
  return text.length === 0 ? null : { text, next: i };
}

interface ReactToolView {
  name: string;
  desc: string;
  // [argName, description] for each argument that carries a description.
  args: Array<[string, string]>;
}

interface ReactOverlay {
  instructions: string;
  tools: ReactToolView[];
}

// React (agent) candidates store their prompt as a JSON blob
// `{"react": <instructions>, "tools": {<name>: {name, desc, args}}}` rather
// than a plain instruction string (see backend seed_candidate_from_program).
// Parse that shape so the drawer can render it structured instead of dumping
// raw JSON. Returns null for non-react prompts (plain scalar-run instruction
// strings) so callers fall back to the <pre> view unchanged.
function parseReactOverlay(value: string): ReactOverlay | null {
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed[0] !== "{") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;
  if (typeof obj.react !== "string") return null;
  const tools: ReactToolView[] = [];
  const toolsRaw = obj.tools;
  if (toolsRaw !== null && typeof toolsRaw === "object") {
    for (const [name, raw] of Object.entries(toolsRaw as Record<string, unknown>)) {
      if (raw === null || typeof raw !== "object") continue;
      const spec = raw as Record<string, unknown>;
      const args: Array<[string, string]> = [];
      const argsRaw = spec.args;
      if (argsRaw !== null && typeof argsRaw === "object") {
        for (const [argName, argSpec] of Object.entries(argsRaw as Record<string, unknown>)) {
          if (argSpec === null || typeof argSpec !== "object") continue;
          const desc = (argSpec as Record<string, unknown>).description;
          if (typeof desc === "string" && desc.length > 0) args.push([argName, desc]);
        }
      }
      tools.push({ name, desc: typeof spec.desc === "string" ? spec.desc : "", args });
    }
  }
  return { instructions: obj.react, tools };
}

// Merge the tool descriptions from every ReAct overlay in a node's prompt into a
// flat name → description map. This is the run's own optimized copy, so the
// shared tools carousel can describe an ``allowed_tools`` roster for any agent we
// optimize rather than only the platform's catalogued tools. First non-empty
// description per name wins; non-react predictors contribute nothing.
function deriveToolDescriptions(prompt: Record<string, string>): Record<string, string> {
  const map: Record<string, string> = {};
  for (const value of Object.values(prompt)) {
    const overlay = parseReactOverlay(value);
    if (overlay === null) continue;
    for (const tool of overlay.tools) {
      if (tool.desc.length > 0 && map[tool.name] === undefined) {
        map[tool.name] = tool.desc;
      }
    }
  }
  return map;
}

// Flatten one tool's editable text — description + argument descriptions — into
// the lines the per-tool diff compares. The tool name is the row label, so it's
// not repeated here. Diffing this surfaces the changes GEPA actually makes and
// drops minified-JSON / key-ordering noise.
function toolToText(tool: ReactToolView): string {
  const lines: string[] = [];
  if (tool.desc.length > 0) lines.push(tool.desc.trim());
  for (const [arg, desc] of tool.args) lines.push(`- ${arg}: ${desc}`);
  return lines.join("\n");
}

// One prompt predictor: its instruction/diff card, with the tool-descriptions
// section lifted out to sit *below* the card as its own block instead of nested
// inside it. Parses the react overlay once and shares it with both halves.
function PromptEntry({
  prompt,
  parentPrompt,
  mode,
  hasParent,
}: {
  prompt: string;
  parentPrompt: string;
  mode: View;
  hasParent: boolean;
}) {
  const overlay = useMemo(() => parseReactOverlay(prompt), [prompt]);
  const parentOverlay = useMemo(
    () => (parentPrompt.length > 0 ? parseReactOverlay(parentPrompt) : null),
    [parentPrompt],
  );

  return (
    <>
      <div className="overflow-hidden rounded-md border border-border/40 bg-background/60 p-3">
        <PromptBody
          prompt={prompt}
          parentPrompt={parentPrompt}
          overlay={overlay}
          parentOverlay={parentOverlay}
          mode={mode}
          hasParent={hasParent}
        />
      </div>
      {overlay !== null ? (
        <ReactToolsSection overlay={overlay} parentOverlay={parentOverlay} />
      ) : null}
    </>
  );
}

function PromptBody({
  prompt,
  parentPrompt,
  overlay,
  parentOverlay,
  mode,
  hasParent,
}: {
  prompt: string;
  parentPrompt: string;
  overlay: ReactOverlay | null;
  parentOverlay: ReactOverlay | null;
  mode: View;
  hasParent: boolean;
}) {
  if (mode === "diff" && hasParent) {
    // React overlays keep their structured shape in compare mode; non-overlay
    // prompts fall back to a flat line diff.
    if (overlay !== null && parentOverlay !== null) {
      return <ReactOverlayDiffView before={parentOverlay} after={overlay} />;
    }
    return <PromptDiff before={parentPrompt} after={prompt} />;
  }

  if (overlay !== null) {
    return <ReactOverlayView overlay={overlay} />;
  }
  return (
    <pre
      className="text-xs whitespace-pre-wrap leading-relaxed font-mono text-foreground/90"
      dir="auto"
      style={{ wordBreak: "break-word" }}
    >
      {prompt}
    </pre>
  );
}

function ReactOverlayView({ overlay }: { overlay: ReactOverlay }) {
  return (
    <div
      className="text-[11px] leading-relaxed text-foreground/90 whitespace-pre-wrap"
      dir="auto"
      style={{ wordBreak: "break-word" }}
    >
      {overlay.instructions}
    </div>
  );
}

type ToolsView = "plain" | "compare";

// The tool-descriptions section carries its own plain/compare toggle, separate
// from the panel-wide prompt/diff tabs: you can read the full descriptions while
// the instructions stay in diff mode, or compare just the tools. The toggle only
// shows when there's a parent candidate to compare against.
function ReactToolsSection({
  overlay,
  parentOverlay,
}: {
  overlay: ReactOverlay;
  parentOverlay: ReactOverlay | null;
}) {
  const canCompare = parentOverlay !== null;
  const [view, setView] = useState<ToolsView>("plain");
  const effectiveView: ToolsView = canCompare ? view : "plain";

  const beforeByName = useMemo(
    () => new Map((parentOverlay?.tools ?? []).map((t) => [t.name, t] as const)),
    [parentOverlay],
  );
  const afterByName = useMemo(
    () => new Map(overlay.tools.map((t) => [t.name, t] as const)),
    [overlay],
  );
  const orderedNames = useMemo(() => {
    const names: string[] = [];
    const seen = new Set<string>();
    for (const t of overlay.tools)
      if (!seen.has(t.name)) {
        names.push(t.name);
        seen.add(t.name);
      }
    for (const t of parentOverlay?.tools ?? [])
      if (!seen.has(t.name)) {
        names.push(t.name);
        seen.add(t.name);
      }
    return names;
  }, [overlay, parentOverlay]);

  const count = effectiveView === "compare" ? orderedNames.length : overlay.tools.length;
  if (count === 0) return null;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <SectionLabel
          label={formatMsg("trajectory.prompt.react.tools", { n: count })}
          info={msg("trajectory.prompt.react.tools.explain")}
          large
        />
        {canCompare ? <ToolsViewToggle view={view} onChange={setView} /> : null}
      </div>
      {effectiveView === "compare" ? (
        <ReactToolsDiffCarousel
          names={orderedNames}
          beforeByName={beforeByName}
          afterByName={afterByName}
        />
      ) : (
        <ReactToolsCarousel tools={overlay.tools} />
      )}
    </div>
  );
}

interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: LucideIcon;
}

// Measuring the thumb must run before paint on the client, but useLayoutEffect
// warns during SSR — fall back to useEffect on the server.
const useIsomorphicLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

// Pill segmented control with a single ``bg-background`` thumb that slides between
// options. The thumb is positioned in the radiogroup's OWN coordinate space — it
// animates ``left``/``width`` taken from the active button's offset box — rather
// than via framer's shared-layout ``layoutId``. ``layoutId`` projects in page
// coordinates, so when the control sits above content that reflows on click (the
// tool-descriptions carousel swaps plain↔compare) that vertical shift leaked into
// the slide as a downward "drop". Animating left/width is a pure horizontal slide
// in any layout context, so the prompt and tools toggles animate identically.
// Snaps instantly under reduced motion.
function SegmentedToggle<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
}: {
  value: T;
  onChange: (v: T) => void;
  options: ReadonlyArray<SegmentedOption<T>>;
  ariaLabel: string;
}) {
  const reduce = useReducedMotion();
  const containerRef = useRef<HTMLDivElement>(null);
  const btnRefs = useRef(new Map<T, HTMLButtonElement>());
  const [thumb, setThumb] = useState<{ left: number; width: number } | null>(null);

  useIsomorphicLayoutEffect(() => {
    const measure = () => {
      const el = btnRefs.current.get(value);
      if (el) setThumb({ left: el.offsetLeft, width: el.offsetWidth });
    };
    measure();
    const container = containerRef.current;
    const ro = new ResizeObserver(measure);
    if (container) ro.observe(container);
    return () => ro.disconnect();
  }, [value, options]);

  return (
    <div
      ref={containerRef}
      role="radiogroup"
      aria-label={ariaLabel}
      className="relative inline-flex shrink-0 items-center rounded-full border border-border/80 bg-muted/40 p-0.5"
    >
      {thumb ? (
        <motion.span
          aria-hidden="true"
          className="absolute inset-y-0.5 rounded-full bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
          initial={false}
          animate={{ left: thumb.left, width: thumb.width }}
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 380, damping: 32 }}
        />
      ) : null}
      {options.map((opt) => {
        const active = opt.value === value;
        const Icon = opt.icon;
        return (
          <button
            key={opt.value}
            ref={(el) => {
              if (el) btnRefs.current.set(opt.value, el);
              else btnRefs.current.delete(opt.value);
            }}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "relative inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium leading-none",
              "transition-colors duration-150 ease-out cursor-pointer",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
              active ? "text-foreground" : "text-foreground/60 hover:text-foreground",
            )}
          >
            <span className="relative z-10 inline-flex items-center gap-1.5">
              {Icon ? <Icon className="size-3.5" aria-hidden="true" /> : null}
              <span>{opt.label}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

function ToolsViewToggle({ view, onChange }: { view: ToolsView; onChange: (v: ToolsView) => void }) {
  return (
    <SegmentedToggle
      value={view}
      onChange={onChange}
      ariaLabel={msg("trajectory.prompt.react.tools.view_aria")}
      options={[
        { value: "plain", label: msg("trajectory.prompt.react.tools.view_plain"), icon: AlignLeft },
        {
          value: "compare",
          label: msg("trajectory.prompt.react.tools.view_compare"),
          icon: GitCompare,
        },
      ]}
    />
  );
}

// The expanded body shared by the plain row and the "unchanged" diff row:
// description + per-argument descriptions. Returns null for a missing tool so
// callers can pass a possibly-null side without guarding.
function ReactToolDetail({ tool }: { tool: ReactToolView | null }) {
  if (tool === null) return null;
  return (
    <div className="space-y-2 border-t border-border/30 px-2.5 py-2.5">
      {tool.desc.length > 0 ? (
        <p
          className="text-[0.75rem] leading-relaxed text-foreground/80 whitespace-pre-wrap"
          dir="auto"
          style={{ wordBreak: "break-word" }}
        >
          {tool.desc}
        </p>
      ) : null}
      {tool.args.length > 0 ? (
        <ul className="space-y-1.5">
          {tool.args.map(([arg, desc]) => (
            <li
              key={arg}
              className="flex flex-col gap-1.5 rounded-md border border-[#DDD4C8]/45 bg-[#F8F4EF]/40 px-2.5 py-1.5"
              dir="ltr"
            >
              <span className="inline-flex shrink-0 self-start rounded bg-foreground/[0.06] px-1.5 py-0.5 font-mono text-[10px] font-medium text-foreground/75">
                {arg}
              </span>
              {desc ? (
                <p
                  className="min-w-0 text-[11px] leading-relaxed text-muted-foreground/85 whitespace-pre-wrap"
                  dir="auto"
                  style={{ wordBreak: "break-word" }}
                >
                  {desc}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

// One tool inside the descriptions carousel: a pinned identity header — the same
// severity-tinted icon + friendly title the tour cards wear, via the shared
// ToolHeader — above a scrollable body holding the tool's optimized description,
// argument descriptions, or per-tool diff. The pinned header keeps the tool in
// view while a long body scrolls inside its slide.
function ReactToolCard({
  name,
  badge,
  children,
}: {
  name: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  const severities = useContext(ToolSeveritiesContext);
  return (
    // No border of its own: the carousel frame is the card now, so this is just
    // the pinned header over its body (avoids a card-in-card).
    <div>
      <ToolHeader
        toolKey={name}
        severity={severities[name]}
        trailing={badge}
        className="border-b border-border/30 px-2.5 py-2"
      />
      <div>{children}</div>
    </div>
  );
}

// Plain-view tool descriptions, paged one tool at a time through the shared
// carousel chrome instead of a long collapsible list.
function ReactToolsCarousel({ tools }: { tools: ReactToolView[] }) {
  return (
    <Carousel
      items={tools}
      itemKey={(t) => t.name}
      renderItem={(t) => (
        <ReactToolCard name={t.name}>
          <ReactToolDetail tool={t} />
        </ReactToolCard>
      )}
      ariaLabel={msg("trajectory.prompt.react.tools_carousel_aria")}
      fluid
      framed
      className="w-full"
    />
  );
}

interface ToolDiffEntry {
  name: string;
  before: ReactToolView | null;
  after: ReactToolView | null;
  status: ToolChangeStatus;
  beforeText: string;
  afterText: string;
  added: number;
  removed: number;
}

// Precompute each tool's change status and +/- line counts once, so paging
// through the diff carousel doesn't re-diff on every slide.
function buildToolDiffEntries(
  names: string[],
  beforeByName: Map<string, ReactToolView>,
  afterByName: Map<string, ReactToolView>,
): ToolDiffEntry[] {
  return names.map((name) => {
    const before = beforeByName.get(name) ?? null;
    const after = afterByName.get(name) ?? null;
    const beforeText = before !== null ? toolToText(before) : "";
    const afterText = after !== null ? toolToText(after) : "";
    const status: ToolChangeStatus =
      before === null
        ? "added"
        : after === null
          ? "removed"
          : beforeText.trim() !== afterText.trim()
            ? "changed"
            : "same";
    let added = 0;
    let removed = 0;
    if (status !== "same") {
      for (const line of diffLines(beforeText, afterText)) {
        if (line.kind === "added") added += 1;
        else if (line.kind === "removed") removed += 1;
      }
    }
    return { name, before, after, status, beforeText, afterText, added, removed };
  });
}

function ReactToolDiffCard({ entry }: { entry: ToolDiffEntry }) {
  return (
    <ReactToolCard
      name={entry.name}
      // "changed" carries no header chip: the diff body already shows its own
      // +/- line counts, so a chip would just repeat them. "added"/"removed"
      // keep the chip for the word label the diff stats don't convey.
      badge={
        entry.status === "added" || entry.status === "removed" ? (
          <ToolChangeChip status={entry.status} added={entry.added} removed={entry.removed} />
        ) : undefined
      }
    >
      {entry.status === "same" ? (
        <ReactToolDetail tool={entry.after ?? entry.before} />
      ) : (
        <div className="px-2.5 py-2">
          <PromptDiff before={entry.beforeText} after={entry.afterText} />
        </div>
      )}
    </ReactToolCard>
  );
}

// Dot tint per change kind on the diff carousel's nav strip, so the user sees at
// a glance which tools changed and can hop straight to them. Hues reuse the
// change-chip palette (added=olive, removed=rust, modified=amber); unchanged
// tools keep the faint default dot (null tone).
const DIFF_DOT_TONE: Record<ToolChangeStatus, string | null> = {
  added: "#3f4d1f",
  removed: "#6e2e16",
  changed: "#A85A1A",
  same: null,
};

// Compare-view counterpart of ReactToolsCarousel: same paging chrome, each slide
// showing one tool's change badge over its additions/removals diff. The nav strip
// colours the changed tools and a jump control hops between just those, so a few
// changes buried among many tools are one tap away instead of a long page-through.
function ReactToolsDiffCarousel({
  names,
  beforeByName,
  afterByName,
}: {
  names: string[];
  beforeByName: Map<string, ReactToolView>;
  afterByName: Map<string, ReactToolView>;
}) {
  const entries = useMemo(
    () => buildToolDiffEntries(names, beforeByName, afterByName),
    [names, beforeByName, afterByName],
  );
  const changeIndices = useMemo(
    () => entries.flatMap((e, i) => (e.status === "same" ? [] : [i])),
    [entries],
  );
  return (
    <Carousel
      items={entries}
      itemKey={(e) => e.name}
      renderItem={(e) => <ReactToolDiffCard entry={e} />}
      ariaLabel={msg("trajectory.prompt.react.tools_carousel_aria")}
      dotTone={(i) => {
        const e = entries[i];
        return e ? DIFF_DOT_TONE[e.status] : null;
      }}
      jumpIndices={changeIndices}
      fluid
      framed
      className="w-full"
    />
  );
}

// Animated height/fade reveal for a disclosure body, so the detail eases open in
// step with the chevron rotation instead of snapping. Matches the agent panel's
// ToolCallRow expand (same easing/duration) so tool rows feel identical whether
// they're in the live chat or this trajectory drawer. Under prefers-reduced-motion
// the height tween is dropped — content appears instantly, no easing — since the
// reveal is atmosphere, not information.
function ToolDisclosureBody({ open, children }: { open: boolean; children: React.ReactNode }) {
  const reduce = useReducedMotion();
  return (
    <AnimatePresence initial={false}>
      {open ? (
        <motion.div
          initial={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
          animate={reduce ? { opacity: 1 } : { height: "auto", opacity: 1 }}
          exit={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
          transition={reduce ? { duration: 0 } : { duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
          className="overflow-hidden"
        >
          {children}
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

// Compare-mode counterpart of ReactOverlayView: the instructions diff (when
// changed) or the plain instructions (when unchanged). The tool-descriptions
// section is rendered separately by PromptEntry, below the prompt card.
function ReactOverlayDiffView({ before, after }: { before: ReactOverlay; after: ReactOverlay }) {
  const instructionsChanged = before.instructions.trim() !== after.instructions.trim();

  return instructionsChanged ? (
    <PromptDiff before={before.instructions} after={after.instructions} />
  ) : (
    <div
      className="text-[11px] leading-relaxed text-foreground/90 whitespace-pre-wrap"
      dir="auto"
      style={{ wordBreak: "break-word" }}
    >
      {after.instructions}
    </div>
  );
}

type ToolChangeStatus = "added" | "removed" | "changed" | "same";

function ToolChangeChip({
  status,
  added,
  removed,
}: {
  status: Exclude<ToolChangeStatus, "same">;
  added: number;
  removed: number;
}) {
  const label =
    status === "added"
      ? msg("trajectory.prompt.react.tool.added")
      : status === "removed"
        ? msg("trajectory.prompt.react.tool.removed")
        : msg("trajectory.prompt.react.tool.changed");
  const color =
    status === "removed" ? "#6e2e16" : status === "added" ? "#3f4d1f" : "rgba(28, 22, 18, 0.7)";
  const background =
    status === "removed"
      ? PARETO_FAIL_BG
      : status === "added"
        ? PARETO_PASS_BG
        : "rgba(28, 22, 18, 0.06)";
  return (
    <span
      className="inline-flex shrink-0 items-center gap-1.5 rounded-sm px-1.5 py-[1px] text-[9px] font-semibold"
      style={{ color, background }}
    >
      {status === "changed" ? (
        // A changed tool speaks for itself through its +/- counts; the "שונה"
        // word is redundant next to them, so the chip carries only the numbers.
        <span className="inline-flex items-center gap-1 font-mono tabular-nums" dir="ltr">
          <span style={{ color: "#3f4d1f" }}>+{added}</span>
          <span style={{ color: "#6e2e16" }}>−{removed}</span>
        </span>
      ) : (
        <span>{label}</span>
      )}
    </span>
  );
}

function Section({
  title,
  info,
  action,
  children,
}: {
  title: string;
  info?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  const titleNode = (
    <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
      {title}
    </div>
  );
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        {info !== undefined ? <HelpTip text={info}>{titleNode}</HelpTip> : titleNode}
        {action}
      </div>
      {children}
    </div>
  );
}

type View = "prompt" | "diff";

function usePromptView(): readonly [View, (v: View) => void] {
  const [view, setViewState] = useState<View>("diff");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(DIFF_VIEW_KEY);
    if (stored === "prompt" || stored === "diff") setViewState(stored);
  }, []);

  const setView = (v: View) => {
    setViewState(v);
    try {
      window.localStorage.setItem(DIFF_VIEW_KEY, v);
    } catch {
      /* private mode, etc. */
    }
  };

  return [view, setView] as const;
}

function PromptViewToggle({ view, onChange }: { view: View; onChange: (v: View) => void }) {
  return (
    <SegmentedToggle
      value={view}
      onChange={onChange}
      ariaLabel={msg("trajectory.drawer.toggle.aria")}
      options={[
        { value: "prompt", label: msg("trajectory.drawer.toggle.prompt"), icon: AlignLeft },
        { value: "diff", label: msg("trajectory.drawer.toggle.diff"), icon: GitCompare },
      ]}
    />
  );
}

interface DiffLine {
  text: string;
  kind: "same" | "added" | "removed";
}

function diffLines(before: string, after: string): DiffLine[] {
  const a = before.split("\n");
  const b = after.split("\n");
  const m = a.length;
  const n = b.length;
  const stride = n + 1;
  const dp = new Int32Array((m + 1) * stride);
  for (let i = m - 1; i >= 0; i -= 1) {
    for (let j = n - 1; j >= 0; j -= 1) {
      if (a[i] === b[j]) {
        dp[i * stride + j] = (dp[(i + 1) * stride + (j + 1)] ?? 0) + 1;
      } else {
        const down = dp[(i + 1) * stride + j] ?? 0;
        const right = dp[i * stride + (j + 1)] ?? 0;
        dp[i * stride + j] = down > right ? down : right;
      }
    }
  }
  const out: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    const ai = a[i] ?? "";
    const bj = b[j] ?? "";
    if (ai === bj) {
      out.push({ text: ai, kind: "same" });
      i += 1;
      j += 1;
    } else {
      const down = dp[(i + 1) * stride + j] ?? 0;
      const right = dp[i * stride + (j + 1)] ?? 0;
      if (down >= right) {
        out.push({ text: ai, kind: "removed" });
        i += 1;
      } else {
        out.push({ text: bj, kind: "added" });
        j += 1;
      }
    }
  }
  while (i < m) {
    out.push({ text: a[i] ?? "", kind: "removed" });
    i += 1;
  }
  while (j < n) {
    out.push({ text: b[j] ?? "", kind: "added" });
    j += 1;
  }
  return out;
}

function PromptDiff({
  before,
  after,
  label,
  labelInfo,
}: {
  before: string;
  after: string;
  label?: string;
  labelInfo?: string;
}) {
  const lines = useMemo(() => diffLines(before, after), [before, after]);
  const changed = lines.some((s) => s.kind !== "same");
  if (!changed) {
    return (
      <div className="space-y-1">
        {label !== undefined ? <SectionLabel label={label} info={labelInfo} /> : null}
        <div className="rounded border border-dashed border-border/50 bg-background/40 px-3 py-2 text-[11px] text-muted-foreground">
          {msg("trajectory.detail.diff_unchanged")}
        </div>
      </div>
    );
  }
  const addedCount = lines.reduce((n, l) => n + (l.kind === "added" ? 1 : 0), 0);
  const removedCount = lines.reduce((n, l) => n + (l.kind === "removed" ? 1 : 0), 0);
  const stats = (
    <div dir="ltr" className="inline-flex items-center gap-3 text-[10px] tabular-nums opacity-80">
      <span style={{ color: "#3f4d1f" }}>+{addedCount}</span>
      <span style={{ color: "#6e2e16" }}>−{removedCount}</span>
    </div>
  );
  return (
    <div className={label !== undefined ? "space-y-1.5" : undefined}>
      {label !== undefined ? (
        // Label rides the RTL start (right); the +/- stats ride the end (left),
        // so the legend shares one horizontal line with the section header.
        <div className="flex items-center justify-between gap-3">
          <SectionLabel label={label} info={labelInfo} />
          {stats}
        </div>
      ) : null}
      <div dir="ltr" className="font-mono text-xs leading-relaxed">
        {label === undefined ? <div className="mb-1.5 flex justify-start">{stats}</div> : null}
        <div className="-mx-3">
          {lines.map((line, idx) => {
            const isAdded = line.kind === "added";
            const isRemoved = line.kind === "removed";
            const bg = isAdded ? PARETO_PASS_BG : isRemoved ? PARETO_FAIL_BG : "transparent";
            const color = isAdded ? "#3f4d1f" : isRemoved ? "#6e2e16" : "rgba(28, 22, 18, 0.78)";
            const prefix = isAdded ? "+" : isRemoved ? "−" : " ";
            return (
              <div
                key={idx}
                className="flex items-start gap-2 px-3 py-0.5"
                style={{ background: bg, color }}
              >
                <span
                  aria-hidden="true"
                  className="select-none tabular-nums opacity-60"
                  style={{ width: "0.9rem", textAlign: "center", flexShrink: 0 }}
                >
                  {prefix}
                </span>
                <span className="whitespace-pre-wrap" style={{ wordBreak: "break-word", flex: 1 }}>
                  {line.text.length === 0 ? "​" : line.text}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface ParetoGridSectionProps {
  examples: PerExampleScore[];
  pinnedId: string | null;
  onPin: (id: string) => void;
  valsetById: Map<string, ValsetRow>;
  predictionsForCandidate: Map<string, PredictionValue>;
}

function ParetoGridSection({
  examples,
  pinnedId,
  onPin,
  valsetById,
  predictionsForCandidate,
}: ParetoGridSectionProps) {
  const focused = pinnedId === null ? null : (examples.find((e) => e.id === pinnedId) ?? null);
  const focusedRow = focused === null ? null : (valsetById.get(focused.id) ?? null);
  const focusedPrediction =
    focused === null ? null : (predictionsForCandidate.get(focused.id) ?? null);
  const passed = examples.filter((e) => e.score > 0).length;

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[10px] text-muted-foreground tabular-nums">
          {formatMsg("trajectory.detail.pareto_passed", {
            count: passed,
            total: examples.length,
          })}
        </div>
      </div>
      <div
        className="flex flex-wrap gap-1"
        role="grid"
        aria-label={msg("trajectory.detail.pareto_title")}
      >
        {examples.map((ex, idx) => {
          const ok = ex.score > 0;
          const isPinned = pinnedId === ex.id;
          return (
            <button
              key={ex.id}
              type="button"
              onClick={() => onPin(ex.id)}
              aria-label={formatMsg("trajectory.detail.pareto_example_label", {
                id: ex.id,
                score: ex.score.toFixed(2),
              })}
              aria-pressed={isPinned}
              className="relative rounded-sm font-mono text-[9px] tabular-nums leading-none transition-transform hover:scale-110 focus:scale-110 focus:outline-none"
              style={{
                width: 24,
                height: 24,
                background: ok ? PARETO_PASS : PARETO_FAIL,
                color: ok ? "#2a3812" : "#4a1c0a",
                outline: isPinned ? "2px solid #1c1612" : "none",
                outlineOffset: isPinned ? "1px" : undefined,
                boxShadow: isPinned ? "inset 0 0 0 1px rgba(28, 22, 18, 0.45)" : undefined,
              }}
            >
              <span
                aria-hidden="true"
                className="absolute inset-0 flex items-center justify-center font-semibold opacity-70"
              >
                {idx + 1}
              </span>
            </button>
          );
        })}
      </div>
      {focused !== null ? (
        <div className="overflow-hidden rounded-lg border border-[#DDD4C8]/60 bg-background/70 shadow-[0_1px_2px_rgba(28,22,18,0.02)]">
          <div className="flex items-center justify-between gap-2 border-b border-border/30 bg-[#F8F4EF]/60 px-3 py-2">
            <div className="flex items-center gap-2 text-[11px]">
              <StatusChip passed={focused.score > 0} />
              <span className="font-mono text-[10px] text-muted-foreground" dir="ltr">
                #{focused.id}
              </span>
            </div>
            <HelpTip text={msg("trajectory.minibatch.score_label.explain")}>
              <span className="inline-flex items-baseline gap-1.5">
                <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {msg("trajectory.minibatch.score_label")}
                </span>
                <span
                  className={cn(
                    "font-mono tabular-nums text-[11px]",
                    focused.score > 0 ? "font-semibold text-foreground" : "text-muted-foreground",
                  )}
                >
                  {focused.score.toFixed(2)}
                </span>
              </span>
            </HelpTip>
          </div>
          <div className="space-y-2.5 px-3 pb-3 pt-2.5">
            {focusedRow === null ? (
              <div className="text-[11px] text-muted-foreground italic">
                {msg("trajectory.pareto.cell_detail_pending")}
              </div>
            ) : (
              <ValsetFieldGroups row={focusedRow} prediction={focusedPrediction} />
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ValsetFieldGroups({
  row,
  prediction,
}: {
  row: ValsetRow;
  prediction: PredictionValue | null;
}) {
  const inputs = Object.entries(row.inputs);
  const outputs = Object.entries(row.outputs);
  const predictionEntries = useMemo(() => predictionToEntries(prediction), [prediction]);
  return (
    <div className="space-y-2">
      {inputs.length > 0 ? (
        <ValsetFieldGroup
          label={msg("trajectory.pareto.cell.inputs_label")}
          info={msg("trajectory.pareto.cell.inputs_label.explain")}
          entries={inputs}
        />
      ) : null}
      {prediction === null ? (
        <PendingPredictionRow />
      ) : predictionEntries !== null && predictionEntries.length > 0 ? (
        <AgentTurnGroup
          label={msg("trajectory.pareto.cell.prediction_label")}
          info={msg("trajectory.pareto.cell.prediction_label.explain")}
          entries={predictionEntries}
        />
      ) : null}
      {outputs.length > 0 ? (
        <AgentTurnGroup
          label={msg("trajectory.pareto.cell.outputs_label")}
          info={msg("trajectory.pareto.cell.outputs_label.explain")}
          entries={outputs}
        />
      ) : null}
    </div>
  );
}

function PendingPredictionRow() {
  return (
    <div className="space-y-1">
      <SectionLabel
        label={msg("trajectory.pareto.cell.prediction_label")}
        info={msg("trajectory.pareto.cell.prediction_label.explain")}
      />
      <div className="rounded border border-dashed border-[#DDD4C8]/50 bg-background/60 px-2 py-1.5 text-[11px] italic text-muted-foreground">
        {msg("trajectory.pareto.cell.prediction_unavailable")}
      </div>
    </div>
  );
}

function SectionLabel({
  label,
  info,
  large,
}: {
  label: string;
  info?: string;
  // Match the parent Section title's size; used so the tool-descriptions label
  // lines up with the prompt section header.
  large?: boolean;
}) {
  const node = (
    <div
      className={cn(
        "font-semibold uppercase tracking-wider text-muted-foreground",
        large ? "text-[11px]" : "text-[9px]",
      )}
    >
      {label}
    </div>
  );
  return info === undefined ? node : <HelpTip text={info}>{node}</HelpTip>;
}

function ValsetFieldGroup({
  label,
  info,
  entries,
}: {
  label: string;
  info?: string;
  entries: Array<[string, string]>;
}) {
  // Drop hidden provenance keys (wizard/state snapshots) and empty
  // containers/blank values — both render as noise next to the meaningful
  // fields, so the group shows only what's worth reviewing.
  const visible = entries.filter(
    ([key, value]) => !isHiddenField(key) && !isEmptyFieldValue(value),
  );
  return (
    <div className="space-y-1">
      <SectionLabel label={label} info={info} />
      <div className="space-y-1">
        {visible.map(([key, value], idx) => (
          <FieldRow
            key={key.length === 0 ? `__raw_${idx}` : key}
            fieldName={key.length > 0 ? key : undefined}
            value={value}
          />
        ))}
      </div>
    </div>
  );
}
