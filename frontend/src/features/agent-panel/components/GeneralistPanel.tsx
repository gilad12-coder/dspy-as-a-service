"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { History, PanelLeftClose, Plus, RotateCcw, Sparkles, WandSparkles, XCircle } from "lucide-react";
import { msg } from "@/shared/lib/messages";

import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";

import { AgentThread } from "@/shared/ui/agent/agent-thread";
import { ChatTranscript } from "@/shared/ui/agent/chat-transcript";
import { Composer } from "@/shared/ui/agent/composer";
import type { AgentThinking, AgentToolCall } from "@/shared/ui/agent/types";
import { EmptyState } from "@/shared/ui/empty-state";
import { SubmitSplashOverlay, SUBMIT_SPLASH_HOLD_MS } from "@/shared/ui/submit-splash-overlay";
import { cn } from "@/shared/lib/utils";
import { formatMsg } from "@/shared/lib/messages";

import { formatShortcut, useUserPrefs } from "@/features/settings";

import { useConversationStore } from "../hooks/use-conversation-store";
import { useGeneralistAgent } from "../hooks/use-generalist-agent";
import { useCodeAuthoringAgent } from "../hooks/use-code-authoring-agent";
import { useGeneralistPanelState } from "../hooks/use-panel-state";
import { getConversation } from "../lib/conversation-api";
import { TRUST_MODE_HUE, useTrustMode } from "../hooks/use-trust-mode";
import { extractWizardPatch, useWizardStateOptional } from "../hooks/use-wizard-state";
import type { ToolEndPayload, ToolStartPayload, WizardState } from "../lib/types";
import { stageDatasetForAgent } from "@/shared/lib/api";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { registerTutorialHook } from "@/features/tutorial";

import { MAX_WIDTH, MIN_WIDTH, NARROW_VIEWPORT_QUERY } from "../constants";

import { ApprovalCard } from "./ApprovalCard";
import { ConversationDrawer } from "./ConversationDrawer";
import { DatasetUploadCard, type ConfirmedDataset } from "./DatasetUploadCard";
import { InferenceFormCard } from "./InferenceFormCard";
import { CodeAuthoringCard } from "./CodeAuthoringCard";
import { MinimizedPill } from "./MinimizedPill";
import { PresenceStrip } from "./PresenceStrip";
import { ToolCallRow } from "./ToolCallRow";
import { ToolsCarousel } from "./ToolsCarousel";
import { TrustToggle } from "./TrustToggle";
import { getToolRenderer } from "./tool-renderers";

const REQUEST_DATASET_TOOL = "request_user_dataset_datasets_request_upload_post";
const REQUEST_INFERENCE_TOOL = "request_user_inference";
const REQUEST_CODE_TOOL = "request_code_authoring";

// Submit tools whose success should mirror the manual wizard submit button:
// play the splash banner, then route to the new optimization's details.
const SUBMIT_TOOLS: ReadonlySet<string> = new Set([
  "submit_job_run_post",
  "submit_grid_search_grid_search_post",
]);

/**
 * Pull the new optimization's id out of a submit tool result. The backend's
 * ``OptimizationSubmissionResponse`` carries ``optimization_id``; results may
 * arrive raw or nested under ``result``, and ``id`` is accepted as a defensive
 * fallback for any reshaped payload.
 */
function extractOptimizationId(result: unknown): string | null {
  if (!result || typeof result !== "object") return null;
  const top = result as Record<string, unknown>;
  const inner =
    top.result && typeof top.result === "object"
      ? (top.result as Record<string, unknown>)
      : top;
  const candidate = inner.optimization_id ?? inner.id;
  return typeof candidate === "string" && candidate.length > 0 ? candidate : null;
}

function getEffectiveMinWidth(): number {
  if (typeof window === "undefined") return MIN_WIDTH;
  return Math.max(MIN_WIDTH, Math.min(360, window.innerWidth - 200));
}

function clampWidth(n: number): number {
  const min = getEffectiveMinWidth();
  return Math.max(min, Math.min(MAX_WIDTH, Math.round(n)));
}

interface GeneralistPanelProps {
  /** Optional wizard-state override. Defaults to the shared context. */
  wizardState?: WizardState;
}

/**
 * Docked generalist agent panel. Renders as an aside anchored to the document's
 * inline-end edge — left in Hebrew/RTL, right in English/LTR — that the user can
 * resize, minimize to a pill, and toggle with Ctrl+J.
 * Mounted once in the app shell so the thread survives route changes.
 */
export function GeneralistPanel({ wizardState }: GeneralistPanelProps = {}) {
  const { open, setOpen, width, setWidth } = useGeneralistPanelState();
  const { mode: trustMode, next: cycleTrust } = useTrustMode();
  const { prefs } = useUserPrefs();
  const shortcutLabel = formatShortcut(prefs.agentShortcut);
  const reduceMotion = useReducedMotion();
  const hue = TRUST_MODE_HUE[trustMode];
  const dir = getActiveDir();
  const isRtl = dir === "rtl";

  const openPanel = React.useCallback(() => {
    setOpen(true);
  }, [setOpen]);

  const closePanel = React.useCallback(() => {
    setOpen(false);
  }, [setOpen]);

  React.useEffect(
    () => registerTutorialHook("setGeneralistPanelOpen", (next) => setOpen(next)),
    [setOpen],
  );

  // Track the narrow viewport so the panel can render as a full-screen
  // overlay drawer (rather than a side-by-side aside) at <=1023px. We collapse
  // only on the wide→narrow *crossing* while open, so a panel docked at the
  // desktop layout doesn't linger as a stale full-screen sheet after a resize.
  // Crucially we no longer force-close on every render at narrow width — that
  // made the panel impossible to open at all on laptops/split-screen.
  const [isNarrow, setIsNarrow] = React.useState(false);
  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const mql = window.matchMedia(NARROW_VIEWPORT_QUERY);
    let wasNarrow = mql.matches;
    setIsNarrow(mql.matches);
    const onChange = () => {
      const narrow = mql.matches;
      setIsNarrow(narrow);
      if (narrow && !wasNarrow) setOpen(false);
      wasNarrow = narrow;
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [setOpen]);

  const wizardCtx = useWizardStateOptional();
  const effectiveWizard: WizardState = (() => {
    const base = wizardCtx?.state ?? {};
    return wizardState ? { ...base, ...wizardState } : base;
  })();

  // Agent-submit splash: a successful submit tool plays the same full-screen
  // banner the manual wizard button shows, then routes to the new job's
  // details. The panel is mounted in the app shell (survives route changes),
  // so it owns this overlay regardless of which page the agent submitted from.
  const router = useRouter();
  const searchParams = useSearchParams();
  const [submitSplash, setSubmitSplash] = React.useState(false);
  const splashTimersRef = React.useRef<Array<ReturnType<typeof setTimeout>>>([]);
  React.useEffect(() => () => splashTimersRef.current.forEach(clearTimeout), []);

  const startSubmitSplash = React.useCallback(
    (optimizationId: string) => {
      setSubmitSplash(true);
      // Drop the replay cache now so a later /submit visit can't re-stage the
      // just-submitted dataset (the backend evicts it on submit anyway).
      try {
        window.sessionStorage.removeItem("wizard:staged-dataset");
      } catch {
        // sessionStorage can be unavailable under private-mode policy.
      }
      const routeTimer = setTimeout(() => {
        router.push(`/optimizations/${optimizationId}`);
        const liftTimer = setTimeout(() => {
          // Lift the splash once the details route has had a beat to paint
          // underneath it, mirroring how the wizard's splash unmounts on nav.
          setSubmitSplash(false);
          // Reset the shared wizard state *after* navigation has unmounted any
          // mounted /submit form, so the wizard's outgoing sync effects can't
          // immediately re-push the stale values back into the context.
          wizardCtx?.reset();
        }, 600);
        splashTimersRef.current.push(liftTimer);
      }, SUBMIT_SPLASH_HOLD_MS);
      splashTimersRef.current.push(routeTimer);
    },
    [router, wizardCtx],
  );

  const handleToolEnd = React.useCallback(
    (ev: ToolEndPayload) => {
      if (ev.status === "ok" && SUBMIT_TOOLS.has(ev.tool)) {
        const optimizationId = extractOptimizationId(ev.result);
        if (optimizationId) startSubmitSplash(optimizationId);
      }
      if (!wizardCtx) return;
      const patch = extractWizardPatch(ev.result);
      if (Object.keys(patch).length > 0) wizardCtx.applyAgentPatch(patch);
    },
    [wizardCtx, startSubmitSplash],
  );

  const store = useConversationStore({ enabled: open });

  const handleConversationMeta = React.useCallback(
    (id: string, title: string) => {
      store.upsertFromMeta(id, title);
      store.setActiveId(id);
      // The user is actively in this conversation — every new turn is "seen"
      // by definition. Without this, the row would render as unread the
      // moment the server bumps its updated_at.
      store.markSeen(id);
    },
    [store],
  );

  // Arm the code-authoring mirror only on a live tool-start (not a reopened
  // historical call). The code agent itself is hosted below at panel scope, so
  // its run + timer survive the aside collapse / auto-minimize.
  const [codeArmed, setCodeArmed] = React.useState(false);
  const handedOffRef = React.useRef(false);
  const handleToolStart = React.useCallback((ev: ToolStartPayload) => {
    if (ev.tool === REQUEST_CODE_TOOL) {
      handedOffRef.current = false;
      setCodeArmed(true);
    }
  }, []);

  const agent = useGeneralistAgent({
    wizardState: effectiveWizard,
    trustMode,
    onToolStart: handleToolStart,
    onToolEnd: handleToolEnd,
    onConversationMeta: handleConversationMeta,
  });
  const streaming = agent.status === "streaming";

  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const searchDebounceRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  React.useEffect(() => {
    if (!drawerOpen) return;
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    // 400ms matches the explore SearchBar so the embedding API (Jina v4)
    // sees one request per typing pause instead of one per keystroke.
    // Shorter intervals trigger transient rate-limit failures that surface
    // to the user as "search could not happen".
    searchDebounceRef.current = setTimeout(() => {
      void store.refresh({ q: searchQuery.trim() || undefined });
    }, 400);
    return () => {
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    };
  }, [drawerOpen, searchQuery, store]);

  const handlePickConversation = React.useCallback(
    async (id: string) => {
      setDrawerOpen(false);
      const detail = await getConversation(id);
      if (!detail) return;
      agent.loadConversation(id, detail.messages);
      store.setActiveId(id);
      store.markSeen(id);
    },
    [agent, store],
  );

  // Honour ?chat=<id> deep-links (e.g. the storage page's chat rows): open the
  // panel onto that conversation, then strip the param so a refresh or back-nav
  // doesn't reopen it. The panel is client-only and globally mounted, so this
  // works from whichever route the link was clicked on.
  const chatDeepLinkRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    const chatId = searchParams.get("chat");
    if (!chatId || chatDeepLinkRef.current === chatId) return;
    chatDeepLinkRef.current = chatId;
    setOpen(true);
    void handlePickConversation(chatId);
    router.replace(window.location.pathname, { scroll: false });
  }, [searchParams, setOpen, handlePickConversation, router]);

  const handleNewConversation = React.useCallback(() => {
    agent.reset();
    store.setActiveId(null);
  }, [agent, store]);

  // Deleting the conversation you're currently viewing must drop you back to a
  // clean slate — otherwise the panel keeps showing the just-deleted thread and
  // the next message would try to append to a row that no longer exists (404).
  // Deleting some *other* thread from the drawer leaves the active one alone.
  const handleDeleteConversation = React.useCallback(
    async (id: string) => {
      const wasActive = id === (agent.conversationId ?? store.activeId);
      await store.remove(id);
      if (wasActive) {
        agent.reset();
        store.setActiveId(null);
      }
    },
    [agent, store],
  );

  const thinking: AgentThinking = {
    reasoning: agent.reasoning,
    startedAt: agent.reasoningStartedAt,
    endedAt: agent.reasoningEndedAt,
    streaming,
  };

  const [draft, setDraft] = React.useState("");
  const confirmedDatasetCallsRef = React.useRef<Set<string>>(new Set());
  // Latest dataset the user confirmed in-panel — supplies columns + roles +
  // sample rows to the hosted code agent. Kept as state (not a ref) so the
  // hosted agent re-renders into "armed" once a dataset is attached.
  const [lastConfirmedDataset, setLastConfirmedDataset] =
    React.useState<ConfirmedDataset | null>(null);

  // Host the canonical code agent at panel scope (mirroring how the wizard
  // hosts it at wizard scope) so authoring survives the aside collapse /
  // auto-minimize / conversation switch instead of restarting on remount.
  const codeAgent = useCodeAuthoringAgent({
    dataset: lastConfirmedDataset,
    armed: codeArmed,
    moduleName:
      typeof effectiveWizard.module_name === "string" ? effectiveWizard.module_name : undefined,
    optimizerName:
      typeof effectiveWizard.optimizer_name === "string"
        ? effectiveWizard.optimizer_name
        : undefined,
  });
  // The agent's turn ends BEFORE its card finishes (the code agent streams
  // after the turn), so the composer locks on this signal — not ``streaming``
  // alone — or the user could fire "submit" mid-authoring against a wizard
  // that has no validated code yet.
  const codeAuthoringActive = codeAgent.status === "streaming";

  // Drop any prior authoring session when the conversation changes (new chat or
  // one loaded from history) so its code + timer don't leak across threads.
  //
  // Guard against the *handoff turn itself* tripping this reset: the agentSend
  // below assigns a fresh conversation id (null → realId) for a brand-new
  // thread, and that materialization must NOT clear the just-set handoff latch
  // (it would let the completion effect re-fire and double-send code_ready_note
  // ~31ms apart). So reset only on a genuine thread switch — a change between
  // two distinct known ids, or a drop back to null (new chat / delete) — never
  // on the initial null → id assignment of the current thread.
  const codeAgentReset = codeAgent.reset;
  const lastConvIdRef = React.useRef<string | null>(agent.conversationId);
  React.useEffect(() => {
    const prev = lastConvIdRef.current;
    lastConvIdRef.current = agent.conversationId;
    if (prev === agent.conversationId) return;
    if (prev === null && agent.conversationId !== null) return;
    setCodeArmed(false);
    handedOffRef.current = false;
    codeAgentReset();
  }, [agent.conversationId, codeAgentReset]);

  // Auto-orchestrate: once authoring settles with valid code, write it into the
  // shared wizard state and nudge the generalist to proceed — no manual "the
  // code is ready". On validation failure we don't hand off; the card surfaces
  // the error (mirroring the code agent) and the user steers via chat.
  const applyAgentPatch = wizardCtx?.applyAgentPatch;
  const agentSend = agent.send;
  React.useEffect(() => {
    if (handedOffRef.current) return;
    if (!codeArmed) return;
    if (agent.status === "streaming") return;
    if (codeAgent.status !== "done") return;
    if (!codeAgent.signatureCode.trim() || !codeAgent.metricCode.trim()) return;
    const sigVal = codeAgent.signatureValidation;
    const metVal = codeAgent.metricValidation;
    if (!sigVal || sigVal.errors.length > 0) return;
    if (!metVal || metVal.errors.length > 0) return;
    handedOffRef.current = true;
    // Ride the authored code on this SAME nudge turn as an explicit override,
    // not just via applyAgentPatch. agentSend reads a render-lagged snapshot,
    // so without the override the code_ready_note turn would ship a
    // wizard_state with empty signature_code/metric_code — the agent would see
    // "Code incomplete", reject the Model step as out-of-order, and loop back
    // into request_code_authoring. Mirrors the dataset handoff above.
    const authoredCode = {
      signature_code: codeAgent.signatureCode,
      metric_code: codeAgent.metricCode,
    };
    applyAgentPatch?.(authoredCode);
    agentSend(
      msg("auto.features.agent.panel.components.generalistpanel.code_ready_note"),
      authoredCode,
    );
  }, [
    codeArmed,
    agent.status,
    agentSend,
    applyAgentPatch,
    codeAgent.status,
    codeAgent.signatureCode,
    codeAgent.metricCode,
    codeAgent.signatureValidation,
    codeAgent.metricValidation,
  ]);

  const handleDatasetConfirm = React.useCallback(
    async (callId: string, confirmed: ConfirmedDataset) => {
      confirmedDatasetCallsRef.current.add(callId);
      setLastConfirmedDataset(confirmed);
      // Stage the rows on the backend before notifying the agent so the
      // very next turn's wizard_state carries ``staged_dataset_id``. The
      // submit wizard's own staging path only runs when its form is
      // mounted; without this call an agent-panel-only flow ends up
      // calling ``submit_job_run_post`` with no dataset and 422s.
      let stagedDatasetId: string | undefined;
      try {
        const staged = await stageDatasetForAgent({
          dataset: confirmed.rows,
          dataset_filename: confirmed.fileName,
        });
        stagedDatasetId = staged.staged_dataset_id;
      } catch {
        // Best-effort: if staging fails the agent will surface a clear
        // 422 on submit and the user can re-attach. Falling back to no
        // staged id matches the prior behaviour for the dispatched event.
      }
      const stagedDetail = {
        dataset: confirmed.rows,
        dataset_filename: confirmed.fileName,
        staged_dataset_id: stagedDatasetId,
        wizard_state: {
          dataset_ready: true,
          columns_configured: true,
          dataset_columns: confirmed.columns,
          column_roles: confirmed.columnRoles,
          column_kinds: confirmed.columnKinds,
        },
      };
      // Persist before dispatch so the wizard can replay it after navigation;
      // the listener in use-submit-wizard reads+clears this key on mount.
      try {
        window.sessionStorage.setItem("wizard:staged-dataset", JSON.stringify(stagedDetail));
      } catch {
        // sessionStorage can throw under quota or private-mode policy; the
        // live event below still covers the same-page case.
      }
      window.dispatchEvent(
        new CustomEvent("wizard:dataset-staged", { detail: stagedDetail }),
      );
      // Write the whole dataset descriptor into the shared context (not just
      // the id) so it is the single source of truth: a wizard mounted on
      // /submit hydrates the same rows from ``staged_dataset_id`` and reads
      // columns + roles straight from here, instead of depending on the
      // (quota-fragile) sessionStorage replay.
      if (stagedDatasetId && wizardCtx) {
        wizardCtx.setField("staged_dataset_id", stagedDatasetId, "user");
        wizardCtx.setField("dataset_columns", confirmed.columns, "user");
        wizardCtx.setField("column_roles", confirmed.columnRoles, "user");
        wizardCtx.setField("dataset_ready", true, "user");
        wizardCtx.setField("columns_configured", true, "user");
      }
      const mappingLine = confirmed.columns
        .map((c) => {
          const role = confirmed.columnRoles[c];
          if (role === "ignore") return null;
          return `- ${c} → ${role}`;
        })
        .filter(Boolean)
        .join("\n");
      const note = formatMsg(
        "auto.features.agent.panel.components.generalistpanel.dataset_note",
        { p1: confirmed.fileName, p2: confirmed.rowCount, p3: mappingLine },
      );
      const wizardOverride: WizardState | undefined = stagedDatasetId
        ? {
            staged_dataset_id: stagedDatasetId,
            dataset_ready: true,
            columns_configured: true,
            dataset_columns: confirmed.columns,
            column_roles: confirmed.columnRoles,
          }
        : undefined;
      agent.send(note, wizardOverride);
    },
    [agent, wizardCtx],
  );

  const handleSubmit = React.useCallback(() => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    agent.send(trimmed);
    setDraft("");
  }, [agent, draft]);

  const handleRunCode = React.useCallback(
    (code: string, language: string) => {
      if (streaming || codeAuthoringActive) return;
      const prompt = formatMsg("shared.agent.run_code_prompt", {
        language: language || "code",
        code,
      });
      agent.send(prompt);
    },
    [agent, streaming, codeAuthoringActive],
  );

  const resizingRef = React.useRef(false);
  const startResize = React.useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      resizingRef.current = true;
      const onMove = (ev: MouseEvent) => {
        if (!resizingRef.current) return;
        // Width is the gap from the panel's docked edge to the cursor: from the
        // left when docked left (RTL), from the right when docked right (LTR).
        setWidth(clampWidth(isRtl ? ev.clientX : window.innerWidth - ev.clientX));
      };
      const onUp = () => {
        resizingRef.current = false;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [setWidth, isRtl],
  );

  const renderToolCall = React.useCallback(
    (call: AgentToolCall, ctx: { isRetry: boolean }) => {
      if (call.tool === REQUEST_DATASET_TOOL) {
        // Intentionally NOT gated on ``streaming``: the agent's expected
        // flow is to call request_user_dataset and then submit in the
        // same turn, but MiniMax sometimes stops after the dataset
        // request without finishing — that would freeze the card in
        // disabled state while the user waits for a stream that already
        // hung up. Keeping the card live lets the upload proceed; the
        // confirmation handler cancels any in-flight stream before
        // starting the next turn.
        return (
          <DatasetUploadCard
            call={call}
            alreadyConfirmed={confirmedDatasetCallsRef.current.has(call.id)}
            onConfirm={(confirmed) => handleDatasetConfirm(call.id, confirmed)}
          />
        );
      }
      if (call.tool === REQUEST_INFERENCE_TOOL) {
        return <InferenceFormCard call={call} disabled={streaming} />;
      }
      if (call.tool === REQUEST_CODE_TOOL) {
        return <CodeAuthoringCard agent={codeAgent} />;
      }
      const renderer = getToolRenderer(call.tool);
      if (renderer?.card && call.status !== "running") {
        return renderer.card(call);
      }
      const summary = renderer?.summary?.(call) ?? null;
      return <ToolCallRow call={call} isRetry={ctx.isRetry} summary={summary} />;
    },
    [handleDatasetConfirm, streaming, codeAgent],
  );

  const emptyState = (
    <EmptyState
      icon={Sparkles}
      iconWrap="circle"
      variant="compact"
      title={msg("auto.features.agent.panel.components.generalistpanel.1")}
    />
  );

  return (
    <>
      <AnimatePresence>
        {open && isNarrow && (
          <motion.div
            key="generalist-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={closePanel}
            aria-hidden="true"
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
          />
        )}
        {open && (
          <motion.aside
            key="generalist-panel"
            dir={dir}
            initial={reduceMotion ? false : { x: isRtl ? -24 : 24, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={reduceMotion ? { opacity: 0 } : { x: isRtl ? -24 : 24, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
            style={{ width: isNarrow ? "100vw" : `min(${width}px, 92vw)` }}
            className={cn(
              "fixed end-0 inset-y-0 z-40 flex h-dvh shrink-0",
              "bg-background/95 backdrop-blur-xl border-s border-border/60",
              // Shadow falls from the panel toward the page content — rightward
              // when docked left (RTL), leftward when docked right (LTR).
              isRtl
                ? "shadow-[8px_0_32px_rgba(61,46,34,0.06)]"
                : "shadow-[-8px_0_32px_rgba(61,46,34,0.06)]",
            )}
            aria-label={msg("auto.features.agent.panel.components.generalistpanel.literal.1")}
          >
            <div dir={dir} className="relative flex min-w-0 flex-1 flex-col" data-tutorial="agent-panel">
              <div className="flex items-center justify-between gap-2 border-b border-border/40 px-3 py-2.5 shrink-0">
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="inline-flex size-6 items-center justify-center rounded-full text-[#FAF8F5]"
                    style={{ backgroundColor: hue }}
                  >
                    <Sparkles className="size-3" aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5 leading-tight">
                      <span className="text-[0.8125rem] font-medium">
                        {msg("auto.features.agent.panel.components.generalistpanel.3")}
                      </span>
                      <Popover>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <PopoverTrigger asChild>
                              <button
                                type="button"
                                className="rounded-md p-1 text-muted-foreground hover:bg-accent/70 hover:text-foreground transition-colors cursor-pointer"
                                aria-label={msg(
                                  "auto.features.agent.panel.components.generalistpanel.literal.2",
                                )}
                              >
                                <WandSparkles className="size-3.5" />
                              </button>
                            </PopoverTrigger>
                          </TooltipTrigger>
                          <TooltipContent side="bottom">
                            {msg("auto.features.agent.panel.components.generalistpanel.4")}
                          </TooltipContent>
                        </Tooltip>
                        <PopoverContent side="bottom" align="center" sideOffset={8} className="p-0">
                          <ToolsCarousel />
                        </PopoverContent>
                      </Popover>
                    </div>
                    {agent.statusLabel && (
                      <div className="text-[0.6875rem] text-muted-foreground truncate">
                        {agent.statusLabel}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <TrustToggle mode={trustMode} onCycle={cycleTrust} />
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => setDrawerOpen(true)}
                        className="rounded-md p-1.5 text-muted-foreground hover:bg-accent/70 hover:text-foreground transition-colors cursor-pointer"
                        aria-label={msg(
                          "auto.features.agent.panel.components.generalistpanel.history_button",
                        )}
                      >
                        <History className="size-3.5" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      {msg("auto.features.agent.panel.components.generalistpanel.history_button")}
                    </TooltipContent>
                  </Tooltip>
                  {agent.messages.length > 0 && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          onClick={handleNewConversation}
                          className="rounded-md p-1.5 text-muted-foreground hover:bg-accent/70 hover:text-foreground transition-colors cursor-pointer"
                          aria-label={msg(
                            "auto.features.agent.panel.components.generalistpanel.new_conversation",
                          )}
                        >
                          <Plus className="size-3.5" />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent side="bottom">
                        {msg("auto.features.agent.panel.components.generalistpanel.new_conversation")}
                      </TooltipContent>
                    </Tooltip>
                  )}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={closePanel}
                        className="rounded-md p-1.5 text-muted-foreground hover:bg-accent/70 hover:text-foreground transition-colors cursor-pointer"
                        aria-label={msg(
                          "auto.features.agent.panel.components.generalistpanel.literal.4",
                        )}
                      >
                        <PanelLeftClose className="size-3.5" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      {msg("auto.features.agent.panel.components.generalistpanel.6")}{" "}
                      <span className="opacity-70 font-mono">({shortcutLabel})</span>
                    </TooltipContent>
                  </Tooltip>
                </div>
              </div>

              <AgentThread
                isEmpty={agent.messages.length === 0}
                emptyState={emptyState}
                scrollDeps={[
                  agent.messages.length,
                  agent.messages[agent.messages.length - 1]?.content,
                  agent.messages[agent.messages.length - 1]?.toolCalls?.length,
                  agent.reasoning,
                  agent.statusLabel,
                  agent.pendingApproval?.id ?? "",
                ]}
              >
                <ChatTranscript
                  messages={agent.messages}
                  streaming={streaming}
                  editAndResend={agent.editAndResend}
                  thinking={thinking}
                  renderToolCall={renderToolCall}
                  onRunCode={handleRunCode}
                  animatePairs
                  trailing={() => (
                    <>
                      {agent.pendingApproval && (
                        <ApprovalCard
                          payload={agent.pendingApproval}
                          onResolve={agent.confirmApproval}
                        />
                      )}
                      {agent.error && (
                        <div className="rounded-lg bg-[#FCEFEB]/60 border border-[#9B2C1F]/20 px-2.5 py-2 text-xs text-[#7A1E13] space-y-1.5">
                          <div className="flex items-start gap-1.5">
                            <XCircle className="size-3 shrink-0 mt-0.5 text-[#9B2C1F]" />
                            <span className="flex-1 break-words min-w-0" dir="auto">
                              {agent.error}
                            </span>
                          </div>
                          <div className="flex gap-1.5 ps-4">
                            <button
                              type="button"
                              onClick={agent.retry}
                              className="inline-flex items-center gap-1 text-[0.6875rem] text-[#7A1E13] bg-[#9B2C1F]/10 hover:bg-[#9B2C1F]/20 px-2 py-0.5 rounded cursor-pointer transition-colors"
                            >
                              <RotateCcw className="size-3" />
                              {msg("auto.features.agent.panel.components.generalistpanel.error_retry")}
                            </button>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                />
              </AgentThread>

              <Composer
                value={draft}
                onChange={setDraft}
                onSubmit={handleSubmit}
                onStop={agent.stop}
                placeholder={msg("auto.features.agent.panel.components.generalistpanel.literal.5")}
                streaming={streaming}
                disabled={codeAuthoringActive}
                sendAriaLabel={msg(
                  "auto.features.agent.panel.components.generalistpanel.literal.6",
                )}
                stopAriaLabel={msg(
                  "auto.features.agent.panel.components.generalistpanel.literal.7",
                )}
              />

              <PresenceStrip active={streaming} hue={hue} />
            </div>

            {/* Resize handle on the inline-start edge — the side that faces the
                page content, so dragging it grows/shrinks the panel. */}
            <button
              type="button"
              onMouseDown={startResize}
              aria-label={msg("auto.features.agent.panel.components.generalistpanel.literal.8")}
              tabIndex={-1}
              className={cn(
                "absolute top-0 start-0 h-full w-1 cursor-col-resize",
                "hover:bg-[#C8A882]/40 active:bg-[#C8A882]/60 transition-colors",
                "max-lg:hidden",
              )}
            />
          </motion.aside>
        )}
      </AnimatePresence>

      {!open && (
        <>
          <MinimizedPill
            onOpen={openPanel}
            active={streaming}
            statusLabel={agent.statusLabel}
            hue={hue}
          />
        </>
      )}

      <ConversationDrawer
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        conversations={store.conversations}
        loading={store.loading}
        activeId={agent.conversationId ?? store.activeId}
        unreadIds={store.unreadIds}
        query={searchQuery}
        onQueryChange={setSearchQuery}
        onPick={handlePickConversation}
        onRename={store.rename}
        onTogglePin={store.togglePin}
        onDelete={handleDeleteConversation}
      />

      <SubmitSplashOverlay show={submitSplash} />
    </>
  );
}
