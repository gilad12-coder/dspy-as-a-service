"use client";

/**
 * Dry-run input dialog: collects one value per workflow input field
 * (prefilled from the first dataset row) and fires the billed test
 * execution. Results are handed back to the canvas, which paints per-node
 * trace badges.
 */

import * as React from "react";
import { Cpu, Loader2, Play } from "lucide-react";

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
import { msg } from "@/shared/lib/messages";
import type { WorkflowDryRunResponse } from "@/shared/types/api";

interface DryRunDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  inputFields: string[];
  sampleInputs: Record<string, string>;
  /** Selected model, surfaced as a chip so the pick made from the canvas is
      visible — and changeable via `onPickModel` — without leaving the dialog. */
  modelName?: string | null;
  onPickModel?: () => void;
  run: (inputs: Record<string, unknown>) => Promise<WorkflowDryRunResponse>;
  onResult: (result: WorkflowDryRunResponse) => void;
}

export function DryRunDialog({
  open,
  onOpenChange,
  inputFields,
  sampleInputs,
  modelName,
  onPickModel,
  run,
  onResult,
}: DryRunDialogProps) {
  const [values, setValues] = React.useState<Record<string, string>>({});
  const [running, setRunning] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!open) return;
    setValues((prev) => {
      const next: Record<string, string> = {};
      for (const field of inputFields) {
        next[field] = prev[field] ?? sampleInputs[field] ?? "";
      }
      return next;
    });
    setError(null);
  }, [open, inputFields, sampleInputs]);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await run(values);
      onResult(result);
      onOpenChange(false);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setRunning(false);
    }
  };

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
                onChange={(e) => setValues((v) => ({ ...v, [field]: e.target.value }))}
                className="flex w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:border-ring"
              />
            </div>
          ))}
        </div>
        {error && (
          <p className="break-words text-xs text-destructive" dir="ltr">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button
            variant="outline"
            size="sm"
            disabled={running}
            onClick={() => onOpenChange(false)}
          >
            {msg("workflow.dryrun.cancel")}
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
