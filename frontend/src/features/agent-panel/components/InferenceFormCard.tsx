"use client";

import * as React from "react";
import { Loader2, Sparkles, XCircle } from "lucide-react";
import { msg } from "@/shared/lib/messages";

import { Button } from "@/shared/ui/primitives/button";
import { autoResizeTextarea } from "@/shared/ui/agent";
import { formatOutput } from "@/shared/lib";
import { getServeInfo, serveProgram } from "@/shared/lib/api";
import type { AgentToolCall } from "@/shared/ui/agent/types";
import type { ServeInfoResponse, ServeResponse } from "@/shared/types/api";

interface InferenceFormCardProps {
  call: AgentToolCall;
  disabled?: boolean;
}

function getArgs(call: AgentToolCall): Record<string, unknown> {
  const p = (call.payload ?? {}) as Record<string, unknown>;
  const a = p.arguments;
  return a && typeof a === "object" && !Array.isArray(a) ? (a as Record<string, unknown>) : {};
}

function getInitialResult(call: AgentToolCall): ServeResponse | null {
  const p = (call.payload ?? {}) as Record<string, unknown>;
  const r = p.result;
  if (!r || typeof r !== "object") return null;
  const candidate = r as Partial<ServeResponse>;
  if (typeof candidate.optimization_id !== "string") return null;
  return candidate as ServeResponse;
}

function inputsFromArgs(args: Record<string, unknown>): Record<string, string> {
  const raw = args.inputs;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    out[k] = typeof v === "string" ? v : v == null ? "" : String(v);
  }
  return out;
}

export function InferenceFormCard({ call, disabled }: InferenceFormCardProps) {
  const args = getArgs(call);
  const optimizationId =
    typeof args.optimization_id === "string"
      ? args.optimization_id
      : typeof args.id === "string"
        ? args.id
        : "";

  const initialResult = getInitialResult(call);
  const initialInputs = inputsFromArgs(args);

  const [info, setInfo] = React.useState<ServeInfoResponse | null>(null);
  const [infoError, setInfoError] = React.useState<string | null>(null);
  const [running, setRunning] = React.useState(false);
  const [result, setResult] = React.useState<ServeResponse | null>(initialResult);
  const [submittedInputs, setSubmittedInputs] = React.useState<Record<string, string> | null>(
    initialResult ? initialInputs : null,
  );
  const [runError, setRunError] = React.useState<string | null>(null);

  // Textarea refs (uncontrolled), keyed by field name. We use refs so the
  // auto-resize logic can run on every change without forcing a re-render
  // per keystroke — same pattern as ServeChat.
  const textareaRefs = React.useRef<Record<string, HTMLTextAreaElement | null>>({});

  React.useEffect(() => {
    if (!optimizationId) return;
    let cancelled = false;
    getServeInfo(optimizationId)
      .then((data) => {
        if (cancelled) return;
        setInfo(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setInfoError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [optimizationId]);

  const fields = info?.input_fields ?? Object.keys(initialInputs);

  const collectValues = React.useCallback((): Record<string, string> => {
    const vals: Record<string, string> = {};
    for (const f of fields) vals[f] = textareaRefs.current[f]?.value ?? "";
    return vals;
  }, [fields]);

  const handleSubmit = React.useCallback(
    async (e?: React.FormEvent) => {
      if (e) e.preventDefault();
      if (disabled || running || !optimizationId || fields.length === 0) return;
      const values = collectValues();
      if (!fields.every((f) => (values[f] ?? "").trim().length > 0)) return;
      setRunning(true);
      setRunError(null);
      try {
        const response = await serveProgram(optimizationId, values);
        setResult(response);
        setSubmittedInputs(values);
        for (const f of fields) {
          const el = textareaRefs.current[f];
          if (el) {
            el.value = "";
            autoResizeTextarea(el);
          }
        }
      } catch (err) {
        setRunError(err instanceof Error ? err.message : String(err));
      } finally {
        setRunning(false);
      }
    },
    [collectValues, disabled, fields, optimizationId, running],
  );

  const outputFields =
    result?.output_fields && result.output_fields.length > 0
      ? result.output_fields
      : result
        ? Object.keys(result.outputs)
        : [];

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-[#C8A882]/40 bg-gradient-to-br from-[#FAF8F5] to-[#F5EFE6] shadow-[0_4px_16px_rgba(61,46,34,0.06)] overflow-hidden">
        <div className="px-4 pt-3.5 pb-2.5 border-b border-[#C8A882]/25 bg-white/40">
          <div className="flex items-start gap-2.5">
            <span className="shrink-0 size-7 rounded-full inline-flex items-center justify-center bg-[#C8A882]/25 text-[#3D2E22]">
              <Sparkles className="size-3.5" aria-hidden="true" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-[0.8125rem] font-semibold text-[#3D2E22] leading-tight">
                {msg("auto.features.agent.panel.lib.tool.meta.literal.70")}
              </div>
            </div>
          </div>
        </div>

        <div className="px-4 py-4">
          {!info && !infoError && (
            <div className="text-[0.75rem] text-[#6B5B4A] flex items-center justify-center gap-1.5 py-3">
              <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
              {msg("auto.features.agent.panel.components.inferenceformcard.literal.8")}
            </div>
          )}

          {infoError && (
            <div className="text-[0.75rem] text-[#7A1E13] flex items-start gap-1.5 py-2">
              <XCircle className="size-3.5 shrink-0 mt-0.5 text-[#9B2C1F]" aria-hidden="true" />
              <span className="break-words min-w-0" dir="auto">
                {infoError}
              </span>
            </div>
          )}

          {info && fields.length === 0 && (
            <div className="text-[0.75rem] text-[#6B5B4A] py-2">
              {msg("auto.features.agent.panel.components.inferenceformcard.literal.6")}
            </div>
          )}

          {result && submittedInputs && (
            <div className="space-y-3 mb-4">
              <div className="flex justify-start">
                <div
                  className="max-w-[80%] rounded-2xl rounded-br-sm bg-[#3D2E22] text-[#FAF8F5] px-4 py-3 text-sm shadow-sm"
                  dir="ltr"
                >
                  {fields.map((k, i, arr) => (
                    <div key={k} className="font-mono leading-relaxed">
                      <span className="text-[#C8A882] text-xs">{k}: </span>
                      <span className="whitespace-pre-wrap break-words">
                        {submittedInputs[k] ?? ""}
                      </span>
                      {i < arr.length - 1 && arr.length > 1 && (
                        <div className="h-px bg-white/10 my-1.5" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
              <div className="px-1" dir="ltr">
                {outputFields.map((k, i, arr) => (
                  <div
                    key={k}
                    className={`font-mono text-sm leading-relaxed ${arr.length > 1 ? "mb-1" : ""}`}
                  >
                    <span className="text-muted-foreground text-xs">{k}: </span>
                    <span className="whitespace-pre-wrap break-words">
                      {formatOutput(result.outputs[k])}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {runError && (
            <div className="mb-3 text-[0.75rem] text-[#7A1E13] flex items-start gap-1.5">
              <XCircle className="size-3.5 shrink-0 mt-0.5 text-[#9B2C1F]" aria-hidden="true" />
              <span className="break-words min-w-0" dir="auto">
                {msg("auto.features.agent.panel.components.inferenceformcard.literal.7")} {runError}
              </span>
            </div>
          )}

          {fields.length > 0 && (
            <form onSubmit={handleSubmit}>
              <div
                className={`flex gap-2 ${fields.length > 1 ? "items-center" : "items-start"}`}
              >
                <Button
                  type="submit"
                  size="icon"
                  className="shrink-0 rounded-full !size-[42px]"
                  disabled={running || disabled || !optimizationId}
                  aria-label={msg("auto.features.agent.panel.components.inferenceformcard.literal.2")}
                >
                  {running ? (
                    <Loader2 className="size-4 animate-spin" />
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
                <div
                  className={`flex-1 ${fields.length > 1 ? "space-y-2" : "flex gap-2 items-start"}`}
                >
                  {fields.map((field) => (
                    <div key={field} className="flex-1 min-w-0">
                      {fields.length > 1 && (
                        <label
                          htmlFor={`inference-${call.id}-${field}`}
                          className="text-[0.625rem] text-muted-foreground/50 font-mono px-3 mb-0.5 block"
                          dir="ltr"
                        >
                          {field}
                        </label>
                      )}
                      <textarea
                        id={`inference-${call.id}-${field}`}
                        ref={(el) => {
                          textareaRefs.current[field] = el;
                          if (el && el.value === "" && initialInputs[field]) {
                            el.value = initialInputs[field];
                            autoResizeTextarea(el);
                          }
                        }}
                        dir="auto"
                        placeholder={field}
                        defaultValue={initialInputs[field] ?? ""}
                        onChange={(e) => {
                          autoResizeTextarea(e.target);
                          if (runError) setRunError(null);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            const allFilled = fields.every((f) =>
                              textareaRefs.current[f]?.value?.trim(),
                            );
                            if (!running && allFilled) void handleSubmit();
                          }
                        }}
                        rows={1}
                        disabled={running || disabled}
                        className="block w-full bg-muted/20 rounded-2xl border border-[#DDD4C8] px-4 py-[11px] text-sm font-mono leading-[20px] outline-none ring-0 shadow-none resize-none overflow-hidden h-[42px] max-h-[120px] focus:outline-none focus-visible:outline-none focus-visible:ring-0 focus:border-[#C8A882] transition-colors placeholder:text-muted-foreground/40 disabled:opacity-60"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
