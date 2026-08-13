"use client";

import { useRef, useState } from "react";
import { ChatText, Check, PencilSimple, Square, X } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { EmptyState } from "@/shared/ui/empty-state";
import { InlineErrorRow } from "@/shared/ui/inline-error-row";
import type { ServeInfoResponse, WorkflowNodeTrace } from "@/shared/types/api";
import { autoResizeTextarea, Composer, MessageActions } from "@/shared/ui/agent";
import { formatOutput } from "@/shared/lib";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";

export interface ServeChatProps {
  serveInfo: ServeInfoResponse;
  runHistory: Array<{
    inputs: Record<string, string>;
    outputs: Record<string, unknown>;
    model: string;
    ts: number;
    // Per-node execution trace — present only for workflow runs.
    nodeTraces?: WorkflowNodeTrace[] | null;
  }>;
  setRunHistory: React.Dispatch<React.SetStateAction<ServeChatProps["runHistory"]>>;
  streamingRun: { inputs: Record<string, string>; partial: Record<string, string> } | null;
  serveLoading: boolean;
  serveError: string | null;
  setServeError: React.Dispatch<React.SetStateAction<string | null>>;
  textareaRefs: React.MutableRefObject<Record<string, HTMLTextAreaElement | null>>;
  chatScrollRef: React.RefObject<HTMLDivElement | null>;
  handleServe: (overrideInputs?: Record<string, string>) => void;
  handleStopServe: () => void;
  demos: Array<{ inputs: Record<string, unknown> }>;
}

export function ServeChat({
  serveInfo,
  runHistory,
  setRunHistory,
  streamingRun,
  serveLoading,
  serveError,
  setServeError,
  textareaRefs,
  chatScrollRef,
  handleServe,
  handleStopServe,
  demos,
}: ServeChatProps) {
  const [editingRunTs, setEditingRunTs] = useState<number | null>(null);
  const [singleDraft, setSingleDraft] = useState("");
  const editTextareaRefs = useRef<Record<string, HTMLTextAreaElement | null>>({});
  const singleInputField =
    serveInfo.input_fields.length === 1 ? serveInfo.input_fields[0] : undefined;

  const handleEditAndResend = (runTs: number) => {
    setRunHistory((prev) => {
      const idx = prev.findIndex((r) => r.ts === runTs);
      if (idx === -1) return prev;
      return prev.slice(idx + 1);
    });
    const edited: Record<string, string> = {};
    for (const f of serveInfo.input_fields) edited[f] = editTextareaRefs.current[f]?.value ?? "";
    handleServe(edited);
    setEditingRunTs(null);
  };

  return (
    <div className="flex flex-col min-w-0 max-h-[560px] pt-2">
      <div ref={chatScrollRef} className="min-w-0 flex-1 overflow-y-auto pb-4 space-y-6">
        {runHistory.length === 0 && !streamingRun && (
          <EmptyState
            icon={ChatText}
            iconWrap="tile"
            variant="compact"
            title={msg("auto.features.optimizations.components.servechat.1")}
            description={msg("auto.features.optimizations.components.servechat.2")}
          >
            {demos.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-md w-full mt-2">
                {demos.slice(0, 4).map((demo, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      if (singleInputField) {
                        setSingleDraft(String(demo.inputs[singleInputField] ?? ""));
                        return;
                      }
                      for (const f of serveInfo.input_fields) {
                        const el = textareaRefs.current[f];
                        if (el) {
                          el.value = String(demo.inputs[f] ?? "");
                          autoResizeTextarea(el);
                        }
                      }
                    }}
                    className="text-end p-3 rounded-xl border border-[#DDD4C8]/60 hover:border-[#C8A882]/60 bg-muted/10 hover:bg-muted/20 transition-all group"
                    dir="auto"
                  >
                    <div className="text-[0.625rem] font-medium text-[#3D2E22]/50 mb-1">
                      {formatMsg("optimizations.servechat.demo_label", {
                        label: msg("auto.features.optimizations.components.servechat.3"),
                        index: String(i + 1),
                      })}
                    </div>
                    <div className="text-xs text-foreground/70 line-clamp-2 font-mono" dir="ltr">
                      {Object.entries(demo.inputs)
                        .map(([k, v]) => `${k}: ${String(v)}`)
                        .join(", ")
                        .slice(0, 80)}
                      {Object.entries(demo.inputs)
                        .map(([k, v]) => `${k}: ${String(v)}`)
                        .join(", ").length > 80
                        ? "..."
                        : ""}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </EmptyState>
        )}
        {[...runHistory].reverse().map((run) => {
          const isEditing = editingRunTs === run.ts;
          return (
            <div key={run.ts} className="space-y-3">
              {isEditing ? (
                <div className="flex justify-start">
                  <div className="max-w-[85%] w-full space-y-2">
                    {serveInfo.input_fields.map((field) => (
                      <div key={field}>
                        {serveInfo.input_fields.length > 1 && (
                          <label
                            className="text-[0.625rem] text-muted-foreground/50 font-mono px-1 mb-0.5 block"
                            dir="ltr"
                          >
                            {field}
                          </label>
                        )}
                        <textarea
                          ref={(el) => {
                            editTextareaRefs.current[field] = el;
                            autoResizeTextarea(el);
                          }}
                          dir="auto"
                          defaultValue={run.inputs[field] ?? ""}
                          onChange={(e) => {
                            autoResizeTextarea(e.target);
                          }}
                          className="w-full bg-white border border-[#DDD4C8] rounded-xl px-3 py-2 text-sm font-mono resize-none outline-none focus:border-[#C8A882] transition-colors min-h-[40px] max-h-[120px]"
                          rows={1}
                          autoFocus={serveInfo.input_fields[0] === field}
                        />
                      </div>
                    ))}
                    <div className="flex justify-start gap-1.5">
                      <button
                        onClick={() => setEditingRunTs(null)}
                        className="text-[0.6875rem] text-muted-foreground hover:text-foreground px-3 py-1 rounded-lg hover:bg-muted transition-colors"
                      >
                        {msg("auto.features.optimizations.components.servechat.4")}
                      </button>
                      <button
                        onClick={() => handleEditAndResend(run.ts)}
                        className="text-[0.6875rem] text-white bg-[#3D2E22] hover:bg-[#3D2E22]/90 disabled:opacity-40 px-3 py-1 rounded-lg transition-colors"
                      >
                        {msg("auto.features.optimizations.components.servechat.5")}
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex justify-start group/user">
                  <div
                    className="max-w-[80%] rounded-2xl rounded-br-sm bg-[#3D2E22] text-[#FAF8F5] px-4 py-3 text-sm shadow-sm"
                    dir="ltr"
                  >
                    {serveInfo.input_fields.map((k, i, arr) => (
                      <div key={k} className="font-mono leading-relaxed">
                        <span className="text-[#C8A882] text-xs">
                          {k}
                          {msg("optimizations.serve.field_separator")}
                        </span>
                        <span className="whitespace-pre-wrap break-words">
                          {run.inputs[k] ?? ""}
                        </span>
                        {i < arr.length - 1 && arr.length > 1 && (
                          <div className="h-px bg-white/10 my-1.5" />
                        )}
                      </div>
                    ))}
                  </div>
                  {!serveLoading && (
                    <button
                      onClick={() => setEditingRunTs(run.ts)}
                      className="self-center ms-1.5 opacity-0 group-hover/user:opacity-100 transition-opacity p-1.5 rounded-lg hover:bg-muted/60"
                      title={msg("auto.features.optimizations.components.servechat.literal.1")}
                    >
                      <PencilSimple className="size-3 text-muted-foreground" />
                    </button>
                  )}
                </div>
              )}
              {!isEditing && (
                <div className="px-1" dir="ltr">
                  {serveInfo.output_fields.map((k, i, arr) => (
                    <div
                      key={k}
                      className={`font-mono text-sm leading-relaxed ${arr.length > 1 ? "mb-1" : ""}`}
                    >
                      <span className="text-muted-foreground text-xs">
                        {k}
                        {msg("optimizations.serve.field_separator")}
                      </span>
                      <span className="whitespace-pre-wrap break-words">
                        {formatOutput(run.outputs[k])}
                      </span>
                    </div>
                  ))}
                  {run.nodeTraces && run.nodeTraces.length > 0 && (
                    <NodeTraceStrip traces={run.nodeTraces} />
                  )}
                  <div className="mt-1">
                    <MessageActions
                      text={serveInfo.output_fields
                        .map((k) => `${k}${msg("optimizations.serve.field_separator")}${formatOutput(run.outputs[k])}`)
                        .join("\n")}
                      model={run.model}
                      onRegenerate={
                        serveLoading
                          ? undefined
                          : () => {
                              const inputs: Record<string, string> = {};
                              for (const f of serveInfo.input_fields)
                                inputs[f] = run.inputs[f] ?? "";
                              setRunHistory((prev) => prev.filter((r) => r.ts !== run.ts));
                              handleServe(inputs);
                            }
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {streamingRun && (
          <div className="space-y-3">
            <div className="flex justify-start">
              <div
                className="max-w-[80%] rounded-2xl rounded-br-sm bg-[#3D2E22] text-[#FAF8F5] px-4 py-3 text-sm shadow-sm"
                dir="ltr"
              >
                {serveInfo.input_fields.map((k, i, arr) => (
                  <div key={k} className="font-mono leading-relaxed">
                    <span className="text-[#C8A882] text-xs">
                      {k}
                      {msg("optimizations.serve.field_separator")}
                    </span>
                    <span className="whitespace-pre-wrap break-words">
                      {streamingRun.inputs[k] ?? ""}
                    </span>
                    {i < arr.length - 1 && arr.length > 1 && (
                      <div className="h-px bg-white/10 my-1.5" />
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="px-1" dir="ltr">
              {Object.keys(streamingRun.partial).length === 0 ? (
                <div className="flex items-center gap-1.5 py-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#3D2E22]/30 animate-bounce" />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-[#3D2E22]/30 animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-[#3D2E22]/30 animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                  <span className="text-xs text-muted-foreground/40">
                    {msg("auto.features.optimizations.components.servechat.6")}
                  </span>
                </div>
              ) : (
                serveInfo.output_fields.map((k, i, arr) => (
                  <div
                    key={k}
                    className={`font-mono text-sm leading-relaxed ${arr.length > 1 ? "mb-1" : ""}`}
                  >
                    <span className="text-muted-foreground text-xs">
                      {k}
                      {msg("optimizations.serve.field_separator")}
                    </span>
                    <span className="whitespace-pre-wrap break-words">
                      {streamingRun.partial[k] ?? ""}
                    </span>
                    {streamingRun.partial[k] && (
                      <span className="inline-block w-1 h-3 bg-foreground/40 ms-0.5 animate-pulse" />
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-border/40 pt-3">
        {serveError && (
          <InlineErrorRow
            message={serveError}
            onDismiss={() => setServeError(null)}
            dismissLabel={msg("auto.features.optimizations.components.servechat.7")}
            className="mb-2 max-w-2xl mx-auto"
          />
        )}
        {singleInputField ? (
          <Composer
            value={singleDraft}
            onChange={(value) => {
              setSingleDraft(value);
              if (serveError) setServeError(null);
            }}
            onSubmit={() => {
              const value = singleDraft.trim();
              if (!value || serveLoading) return;
              handleServe({ [singleInputField]: value });
              setSingleDraft("");
            }}
            onStop={handleStopServe}
            placeholder={singleInputField}
            streaming={serveLoading}
            sendAriaLabel={msg("auto.features.optimizations.components.servechat.literal.2")}
            className="mx-auto max-w-2xl border-t-0 p-0"
          />
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleServe();
            }}
            className="max-w-2xl mx-auto"
          >
            <div
              className={`flex gap-2 ${getActiveDir() === "ltr" ? "flex-row-reverse" : ""} items-center`}
            >
              <Button
                type={serveLoading ? "button" : "submit"}
                onClick={serveLoading ? handleStopServe : undefined}
                size="icon"
                className="shrink-0 rounded-full !size-[42px]"
                aria-label={
                  serveLoading
                    ? msg("auto.shared.ui.agent.composer.literal.2")
                    : msg("auto.features.optimizations.components.servechat.literal.2")
                }
              >
                {serveLoading ? (
                  <Square className="size-3 fill-current" />
                ) : (
                  <svg viewBox="0 0 24 24" fill="currentColor" className="size-4">
                    <path
                      d="M12 2L12 22M12 2L5 9M12 2L19 9"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      fill="none"
                    />
                  </svg>
                )}
              </Button>
              <div className="flex-1 space-y-2">
                {serveInfo.input_fields.map((field) => (
                  <div key={field} className="flex-1 min-w-0">
                    <label
                      htmlFor={`serve-${field}`}
                      className="text-[0.625rem] text-muted-foreground/50 font-mono px-3 mb-0.5 block"
                      dir="ltr"
                    >
                      {field}
                    </label>
                    <textarea
                      id={`serve-${field}`}
                      ref={(el) => {
                        textareaRefs.current[field] = el;
                      }}
                      dir="auto"
                      placeholder={field}
                      defaultValue=""
                      onChange={(e) => {
                        autoResizeTextarea(e.target);
                        if (serveError) setServeError(null);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          const allFilled = serveInfo.input_fields.every((f) =>
                            textareaRefs.current[f]?.value?.trim(),
                          );
                          if (!serveLoading && allFilled) handleServe();
                        }
                      }}
                      rows={1}
                      className="block w-full bg-muted/20 rounded-2xl border border-[#DDD4C8] px-4 py-[11px] text-sm font-mono leading-[20px] outline-none ring-0 shadow-none resize-none overflow-hidden h-[42px] max-h-[120px] focus:outline-none focus-visible:outline-none focus-visible:ring-0 focus:border-[#C8A882] transition-colors placeholder:text-muted-foreground/40"
                    />
                  </div>
                ))}
              </div>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

/**
 * Compact per-node execution trace under a workflow run's outputs: one row
 * per node with status, latency, and a hoverable error. The graph itself
 * lives in the wizard canvas; here a linear replay keeps the chat readable.
 */
function NodeTraceStrip({ traces }: { traces: WorkflowNodeTrace[] }) {
  return (
    <div className="mt-2 rounded-lg border border-border/50 bg-muted/20 px-3 py-2" dir="ltr">
      <p className="mb-1 text-[0.625rem] font-medium uppercase tracking-wider text-muted-foreground">
        {msg("workflow.playground.trace_title")}
      </p>
      <div className="space-y-0.5">
        {traces.map((trace) => (
          <div
            key={trace.node_id}
            className="flex items-center gap-2 text-xs"
            title={trace.error ?? undefined}
          >
            {trace.error ? (
              <X className="size-3 shrink-0 text-destructive" />
            ) : (
              <Check className="size-3 shrink-0 text-[#5A7247]" />
            )}
            <span className="truncate font-mono text-foreground">{trace.name}</span>
            <span className="text-[0.625rem] uppercase text-muted-foreground/70">{trace.kind}</span>
            <span className="ms-auto tabular-nums text-muted-foreground">
              {Math.round(trace.elapsed_ms)} {msg("workflow.trace.ms")}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
