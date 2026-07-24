"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ChevronDown, RefreshCw, type LucideIcon } from "lucide-react";
import { msg } from "@/shared/lib/messages";
import { CopyGlyph, useCopyToClipboard } from "@/shared/ui/copy-button";

import { cn } from "@/shared/lib/utils";
import type { AgentToolCall } from "@/shared/ui/agent/types";

import { isPlainObject } from "../lib/entry-format";
import { getToolMeta, getToolTitle } from "../lib/tool-meta";

import { EntryRow } from "./EntryRow";

interface ToolCallRowProps {
  call: AgentToolCall;
  isRetry?: boolean;
  /** Hebrew one-liner that replaces ``call.reason`` in the collapsed trigger. */
  summary?: string | null;
  /** Override for the trigger title. Defaults to ``TOOL_META.title``. */
  title?: string | null;
  /** Override for the status-glyph icon (done state). Defaults to ``TOOL_META.icon``. */
  icon?: LucideIcon;
  /**
   * Replaces the default args/result body. Caller renders whatever fits the
   * tool — e.g. a diff view. Wrapped in the same container as the default
   * body so vertical rhythm stays identical to the generalist tools.
   */
  customBody?: React.ReactNode;
}

/**
 * One tool invocation rendered as an expandable row inside the agent
 * bubble. While the call is running the row self-expands and shows a
 * pulse indicator with a live elapsed counter; once done it auto-collapses
 * into a compact chip. Errors keep the row expanded. A per-tool summary
 * string (provided by the renderer registry) replaces the raw LLM reason
 * in the collapsed view with a domain-specific Hebrew one-liner.
 */
export function ToolCallRow({
  call,
  isRetry = false,
  summary,
  title,
  icon,
  customBody,
}: ToolCallRowProps) {
  const meta = getToolMeta(call.tool);
  const derivedTitle = title ?? getToolTitle(call.tool);
  const Icon = icon ?? meta.icon;

  const initiallyOpen = call.status !== "done";
  const [open, setOpen] = React.useState(initiallyOpen);
  const autoCollapsedRef = React.useRef(false);
  const [nowTs, setNowTs] = React.useState(() => Date.now());

  React.useEffect(() => {
    if (call.status === "done" && !autoCollapsedRef.current) {
      autoCollapsedRef.current = true;
      setOpen(false);
    }
    if (call.status === "error" && !open) {
      setOpen(true);
    }
  }, [call.status]);

  React.useEffect(() => {
    if (call.status !== "running") return;
    // Skip ticks in background tabs and catch up on return (mirrors the
    // visibility-gated ticking in LiveElapsed).
    const tick = () => {
      if (document.visibilityState === "visible") setNowTs(Date.now());
    };
    const id = setInterval(tick, 200);
    document.addEventListener("visibilitychange", tick);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [call.status]);

  const elapsedMs =
    call.status === "running" && call.startedAt
      ? nowTs - call.startedAt
      : call.endedAt && call.startedAt
        ? call.endedAt - call.startedAt
        : 0;
  const elapsedLabel =
    elapsedMs >= 100 || call.status === "running" ? formatElapsed(elapsedMs) : null;

  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const args = isPlainObject(payload.arguments) ? payload.arguments : {};
  const result = payload.result;
  const argEntries = Object.entries(args);
  const hasArgs = argEntries.length > 0;
  const hasResult = result !== undefined && result !== null && result !== "";

  const triggerText = summary ?? call.reason ?? null;
  const isError = call.status === "error";
  const showReasonInBody = call.reason && call.reason !== triggerText;

  return (
    <div className={cn("overflow-hidden rounded-md", isError && "bg-[#9B2C1F]/[0.04]")}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={cn(
          "flex w-full items-center gap-2 px-2.5 py-1.5 text-start",
          "transition-colors cursor-pointer",
          "hover:bg-foreground/[0.04]",
          isError && "hover:bg-[#9B2C1F]/[0.07]",
        )}
      >
        <StatusGlyph status={call.status} Icon={Icon} isRetry={isRetry} />
        <span
          className={cn(
            "text-[0.8125rem] font-medium leading-tight truncate shrink-0",
            isError
              ? "text-[#7A1E13]"
              : call.status === "running"
                ? "text-foreground"
                : "text-foreground/80",
          )}
        >
          {derivedTitle}
        </span>
        {triggerText && !open && (
          <span
            className="min-w-0 flex-1 truncate text-[0.75rem] text-muted-foreground/70"
            title={triggerText}
          >
            · {triggerText}
          </span>
        )}
        <span className="ms-auto flex items-center gap-1.5 shrink-0">
          {elapsedLabel && (
            <span
              className={cn(
                "font-mono tabular-nums text-[0.625rem]",
                call.status === "running" ? "text-muted-foreground/80" : "text-muted-foreground/55",
              )}
            >
              {elapsedLabel}
            </span>
          )}
          <ChevronDown
            className={cn(
              "size-3.5 text-muted-foreground/50 transition-transform duration-150",
              open ? "rotate-0" : "rotate-90",
            )}
            aria-hidden="true"
          />
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 space-y-2.5">
              {isRetry && (
                <div className="flex items-center gap-2 flex-wrap">
                  <span
                    className="inline-flex items-center gap-1 text-[0.625rem] text-muted-foreground/70"
                    title={msg("auto.features.agent.panel.components.toolcallrow.literal.2")}
                  >
                    <RefreshCw className="size-2.5" aria-hidden="true" />
                    {msg("auto.features.agent.panel.components.toolcallrow.1")}
                  </span>
                </div>
              )}
              {showReasonInBody && (
                <div className="text-[0.75rem] leading-snug text-muted-foreground/85">
                  {call.reason}
                </div>
              )}
              {customBody !== undefined ? (
                customBody
              ) : (
                <>
                  {hasArgs && (
                    <Section
                      label={msg("auto.features.agent.panel.components.toolcallrow.literal.3")}
                    >
                      <dl className="space-y-1.5">
                        {argEntries.map(([k, v]) => (
                          <EntryRow key={k} argKey={k} value={v} />
                        ))}
                      </dl>
                    </Section>
                  )}
                  {hasResult && (
                    <Section
                      label={
                        isError
                          ? msg("auto.features.agent.panel.components.toolcallrow.literal.4")
                          : msg("auto.features.agent.panel.components.toolcallrow.literal.5")
                      }
                    >
                      <ResultBody result={result} isError={isError} />
                    </Section>
                  )}
                  {!hasArgs && !hasResult && call.status === "running" && (
                    <div className="text-[0.75rem] italic text-muted-foreground/75 leading-snug">
                      {msg("auto.features.agent.panel.components.toolcallrow.2")}
                    </div>
                  )}
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function StatusGlyph({
  status,
  Icon,
  isRetry,
}: {
  status: AgentToolCall["status"];
  Icon: React.ComponentType<{
    className?: string;
    strokeWidth?: number;
    "aria-hidden"?: boolean;
  }>;
  isRetry: boolean;
}) {
  const retryRing = isRetry ? "ring-1 ring-[#A85A1A]/45" : "";
  if (status === "running") {
    return (
      <span
        className={cn(
          "relative inline-flex size-4 items-center justify-center shrink-0 rounded-full",
          retryRing,
        )}
        aria-label={msg("auto.features.agent.panel.components.toolcallrow.literal.6")}
      >
        <span className="absolute inset-0 rounded-full bg-[#3D2E22]/15 animate-ping motion-reduce:animate-none" />
        <span className="relative size-2 rounded-full bg-[#3D2E22]" />
      </span>
    );
  }
  if (status === "error") {
    return (
      <span
        className={cn(
          "inline-flex size-4 items-center justify-center rounded-full bg-[#9B2C1F]/15 text-[#9B2C1F] shrink-0",
          retryRing,
        )}
        aria-label={msg("auto.features.agent.panel.components.toolcallrow.literal.7")}
      >
        <AlertTriangle className="size-2.5" strokeWidth={2.5} aria-hidden />
      </span>
    );
  }
  return (
    <span
      className={cn(
        "inline-flex size-4 items-center justify-center rounded-full bg-[#3D2E22]/15 text-[#3D2E22] shrink-0",
        retryRing,
      )}
      aria-label={msg("auto.features.agent.panel.components.toolcallrow.literal.8")}
    >
      <Icon className="size-2.5" strokeWidth={2.5} aria-hidden />
    </span>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 text-[0.6875rem]/[1.55] font-medium text-muted-foreground/70">
        {label}
      </div>
      {children}
    </div>
  );
}

function ResultBody({ result, isError }: { result: unknown; isError: boolean }) {
  if (typeof result === "string") {
    const trimmed = result.trim();
    if (!trimmed) {
      return <EmptyPlaceholder />;
    }
    return (
      <div className="relative" dir="ltr">
        <pre
          className={cn(
            "whitespace-pre-wrap break-words font-mono text-[0.6875rem]/[1.55] max-h-52 overflow-y-auto rounded-md border p-2 pe-9",
            isError
              ? "border-[#9B2C1F]/20 bg-[#FCEFEB]/60 text-[#7A1E13]"
              : "border-border/40 bg-background/70 text-foreground/90",
          )}
        >
          {trimmed}
        </pre>
        <CopyButton text={trimmed} />
      </div>
    );
  }

  if (isPlainObject(result)) {
    const entries = Object.entries(result);
    if (entries.length === 0) {
      return <EmptyPlaceholder />;
    }
    return (
      <dl className="space-y-1.5">
        {entries.map(([k, v]) => (
          <EntryRow key={k} argKey={k} value={v} />
        ))}
      </dl>
    );
  }

  if (Array.isArray(result)) {
    if (result.length === 0) {
      return <EmptyPlaceholder />;
    }
    return (
      <dl className="space-y-1.5">
        {result.slice(0, 20).map((item, idx) => (
          <EntryRow key={idx} argKey={`${idx + 1}`} value={item} />
        ))}
        {result.length > 20 && (
          <div className="text-[0.625rem] italic text-muted-foreground/60">
            {msg("auto.features.agent.panel.components.toolcallrow.3")}
            {result.length - 20}…
          </div>
        )}
      </dl>
    );
  }

  return (
    <div className="font-mono text-[0.6875rem]/[1.55] text-foreground/90" dir="ltr">
      {String(result)}
    </div>
  );
}

function EmptyPlaceholder() {
  return <div className="text-[0.6875rem]/[1.55] text-muted-foreground/60">—</div>;
}

function CopyButton({ text }: { text: string }) {
  const { copied, copy } = useCopyToClipboard();
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        void copy(text);
      }}
      aria-label={
        copied
          ? msg("auto.features.agent.panel.components.toolcallrow.literal.9")
          : msg("auto.features.agent.panel.components.toolcallrow.literal.10")
      }
      className={cn(
        "absolute end-1.5 top-1.5 inline-flex size-6 items-center justify-center rounded-md",
        "bg-background/85 text-muted-foreground/70 border border-border/40",
        "hover:bg-background hover:text-foreground transition-colors cursor-pointer",
      )}
    >
      <CopyGlyph copied={copied} className="size-3" />
    </button>
  );
}

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const sec = ms / 1000;
  if (sec < 10) return `${sec.toFixed(1)}s`;
  return `${Math.round(sec)}s`;
}
