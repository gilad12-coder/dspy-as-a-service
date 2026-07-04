"use client";

/**
 * Dry-run input dialog: collects one value per workflow input field
 * (prefilled from the first dataset row) and fires the billed test
 * execution over SSE, painting the answer live as tokens arrive. The final
 * result is handed back to the canvas, which paints per-node trace badges.
 */

import * as React from "react";
import { AlertTriangle, CheckCircle2, Cpu, Loader2, Play } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import type { WorkflowDryRunStreamHandlers } from "@/shared/lib/api";
import type { WorkflowDryRunResponse } from "@/shared/types/api";

interface DryRunDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  inputFields: string[];
  outputFields: string[];
  sampleInputs: Record<string, string>;
  /** Selected model, surfaced as a chip so the pick made from the canvas is
      visible — and changeable via `onPickModel` — without leaving the dialog. */
  modelName?: string | null;
  onPickModel?: () => void;
  run: (inputs: Record<string, unknown>, handlers: WorkflowDryRunStreamHandlers) => Promise<void>;
  onResult: (result: WorkflowDryRunResponse) => void;
}

/** One output field's live text: grows with the stream, sticks to the bottom. */
function AnswerFieldBox({
  field,
  text,
  streaming,
}: {
  field: string;
  text: string;
  streaming: boolean;
}) {
  const boxRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const el = boxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [text]);

  return (
    <div className="space-y-1">
      <Label className="font-mono text-[0.6875rem] text-muted-foreground" dir="ltr">
        {field}
      </Label>
      <div
        ref={boxRef}
        dir="auto"
        className="max-h-48 overflow-y-auto rounded-lg border border-border/60 bg-background/80 px-3 py-2 text-sm leading-relaxed break-words whitespace-pre-wrap"
      >
        {text}
        {streaming && (
          <span className="ms-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-[1px] bg-foreground/60 align-middle" />
        )}
      </div>
    </div>
  );
}

export function DryRunDialog({
  open,
  onOpenChange,
  inputFields,
  outputFields,
  sampleInputs,
  modelName,
  onPickModel,
  run,
  onResult,
}: DryRunDialogProps) {
  const [values, setValues] = React.useState<Record<string, string>>({});
  const [running, setRunning] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [streamed, setStreamed] = React.useState<Record<string, string>>({});
  const [activeField, setActiveField] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<WorkflowDryRunResponse | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    if (!open) {
      abortRef.current?.abort();
      return;
    }
    setValues((prev) => {
      const next: Record<string, string> = {};
      for (const field of inputFields) {
        next[field] = prev[field] ?? sampleInputs[field] ?? "";
      }
      return next;
    });
    setError(null);
    setStreamed({});
    setActiveField(null);
    setResult(null);
  }, [open, inputFields, sampleInputs]);

  React.useEffect(() => () => abortRef.current?.abort(), []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    setStreamed({});
    setActiveField(null);
    setResult(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await run(values, {
        signal: controller.signal,
        onToken: (field, chunk) => {
          setActiveField(field);
          setStreamed((s) => ({ ...s, [field]: (s[field] ?? "") + chunk }));
        },
        onFinal: (r) => {
          setResult(r);
          onResult(r);
        },
        onError: (message) => setError(message),
      });
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setRunning(false);
      setActiveField(null);
    }
  };

  const started = running || result !== null || Object.keys(streamed).length > 0;
  // The final outputs are authoritative — the fallback path emits no tokens,
  // and the stream can be cut mid-chunk.
  const answerFor = (field: string): string => {
    if (result?.outputs && result.outputs[field] != null) return String(result.outputs[field]);
    return streamed[field] ?? "";
  };
  const visibleAnswerFields = outputFields.filter(
    (field) => running || answerFor(field).length > 0,
  );

  return (
    <Dialog open={open} onOpenChange={(o) => !running && onOpenChange(o)}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{msg("workflow.dryrun.title")}</DialogTitle>
          <DialogDescription>{msg("workflow.dryrun.description")}</DialogDescription>
        </DialogHeader>
        {modelName && onPickModel && (
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border/60 bg-muted/40 px-3 py-2">
            <div className="flex min-w-0 items-center gap-2">
              <Cpu className="size-3.5 shrink-0 text-muted-foreground" />
              <span className="truncate font-mono text-xs text-foreground" dir="ltr">
                {modelName}
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 shrink-0 px-2 text-xs text-muted-foreground hover:text-foreground"
              disabled={running}
              onClick={onPickModel}
            >
              {msg("workflow.dryrun.change_model")}
            </Button>
          </div>
        )}
        <div className="max-h-80 space-y-3 overflow-y-auto py-1">
          {inputFields.map((field) => (
            <div key={field} className="space-y-1.5">
              <Label className="font-mono text-xs" dir="ltr">
                {field}
              </Label>
              <textarea
                dir="auto"
                rows={2}
                value={values[field] ?? ""}
                disabled={running}
                onChange={(e) => setValues((v) => ({ ...v, [field]: e.target.value }))}
                className="flex w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:border-ring disabled:cursor-not-allowed disabled:opacity-60"
              />
            </div>
          ))}
        </div>
        {started && (
          <div className="space-y-2 border-t border-border/60 pt-3">
            <div
              className={cn(
                "flex items-center gap-1.5 text-[0.6875rem] font-medium",
                running && "text-muted-foreground",
                !running && result && !result.error && "text-[#5A7247]",
                !running && result?.error && "text-[#A3512B]",
              )}
              role="status"
            >
              {running && (
                <>
                  <Loader2 className="size-3 shrink-0 animate-spin" />
                  {msg("workflow.dryrun.running")}
                </>
              )}
              {!running && result && !result.error && (
                <>
                  <CheckCircle2 className="size-3 shrink-0" />
                  {msg("workflow.dryrun.succeeded")}
                </>
              )}
              {!running && result?.error && (
                <>
                  <AlertTriangle className="size-3 shrink-0" />
                  {msg("workflow.dryrun.failed")}
                </>
              )}
            </div>
            <div className="max-h-72 space-y-3 overflow-y-auto">
              {visibleAnswerFields.map((field) => (
                <AnswerFieldBox
                  key={field}
                  field={field}
                  text={answerFor(field)}
                  streaming={running && activeField === field}
                />
              ))}
            </div>
            {result?.error && (
              <p className="break-words text-xs text-destructive" dir="ltr">
                {result.error}
              </p>
            )}
          </div>
        )}
        {error && (
          <p className="break-words text-xs text-destructive" dir="ltr">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (running) {
                abortRef.current?.abort();
                setRunning(false);
              } else {
                onOpenChange(false);
              }
            }}
          >
            {msg(running ? "workflow.dryrun.cancel" : "workflow.dryrun.close")}
          </Button>
          <Button size="sm" className="gap-1.5" disabled={running} onClick={handleRun}>
            {running ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Play className="size-3.5" />
            )}
            {msg(running ? "workflow.dryrun.running" : "workflow.dryrun.run")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
