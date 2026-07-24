"use client";

import * as React from "react";
import { CheckCircle2, Database, Loader2, XCircle } from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { getDatasetRows, type DatasetSummary } from "@/shared/lib/api";
import { DatasetPickerDialog } from "@/features/datasets";

import type { AgentToolCall } from "@/shared/ui/agent/types";

/**
 * What the picker resolves once the user chooses a saved dataset: the id the
 * run submits by reference plus the columns/roles the wizard gate needs. No
 * rows — the backend materializes them from ``source_dataset_id``.
 */
export interface ConfirmedLibraryDataset {
  sourceDatasetId: string;
  name: string;
  columns: string[];
  columnRoles: Record<string, "input" | "output" | "ignore">;
  columnKinds: Record<string, "text" | "image">;
  rowCount: number;
}

interface LibraryDatasetCardProps {
  call: AgentToolCall;
  alreadyConfirmed: boolean;
  onConfirm: (dataset: ConfirmedLibraryDataset) => void;
}

type Phase = "idle" | "loading" | "error" | "done";

/**
 * Inline card for ``request_user_dataset_from_library`` — the by-reference twin
 * of the upload card. Opens the shared library picker; on selection it loads
 * the dataset's rows + saved column schema and hands the caller a
 * ``ConfirmedLibraryDataset`` (carrying ``source_dataset_id``) so the run
 * submits against the saved dataset by reference instead of a fresh upload.
 */
export function LibraryDatasetCard({ call, alreadyConfirmed, onConfirm }: LibraryDatasetCardProps) {
  const [open, setOpen] = React.useState(false);
  const [phase, setPhase] = React.useState<Phase>(alreadyConfirmed ? "done" : "idle");
  const [pickedName, setPickedName] = React.useState<string | null>(null);

  const args = (call.payload?.arguments ?? {}) as Record<string, unknown>;
  const prompt = typeof args.prompt === "string" ? args.prompt.trim() : "";

  const handlePick = React.useCallback(
    async (dataset: DatasetSummary) => {
      setPhase("loading");
      setPickedName(dataset.name);
      try {
        const res = await getDatasetRows(dataset.id);
        const schema = res.column_schema ?? {};
        // A saved dataset ships its column roles; any column the schema doesn't
        // cover defaults to "ignore" so the run only trains on chosen columns
        // (mirrors the /submit wizard's handlePickFromLibrary).
        const savedRoles = schema.column_roles ?? {};
        const columnRoles: Record<string, "input" | "output" | "ignore"> = {};
        for (const col of res.columns) columnRoles[col] = savedRoles[col] ?? "ignore";
        onConfirm({
          sourceDatasetId: dataset.id,
          name: dataset.name,
          columns: res.columns,
          columnRoles,
          columnKinds: schema.column_kinds ?? {},
          rowCount: res.row_count,
        });
        setPhase("done");
      } catch {
        setPhase("error");
      }
    },
    [onConfirm],
  );

  if (phase === "done") {
    return (
      <div className="flex items-center gap-2 rounded-2xl border border-[#5E7A5E]/25 bg-[#F0F4EC] px-4 py-3 text-[0.8125rem] text-[#2F3E32]">
        <CheckCircle2 className="size-3.5 shrink-0 text-[#3E5240]" aria-hidden="true" />
        <span dir="auto" className="min-w-0 flex-1 truncate">
          {pickedName
            ? formatMsg("auto.features.agent.panel.components.librarydatasetcard.picked", {
                p1: pickedName,
              })
            : msg("auto.features.agent.panel.components.librarydatasetcard.title")}
        </span>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-[#C8A882]/30 bg-[#FAF8F5] shadow-sm">
      <div className="flex items-center gap-2 border-b border-[#3D2E22]/10 px-4 py-2.5 text-[0.8125rem] font-medium text-[#3D2E22]">
        <Database className="size-3.5 text-[#3D2E22]" aria-hidden="true" />
        {msg("auto.features.agent.panel.components.librarydatasetcard.title")}
      </div>

      <div className="space-y-3 px-4 py-3">
        {prompt && (
          <p dir="auto" className="text-[0.75rem] leading-snug text-foreground/75">
            {prompt}
          </p>
        )}

        {phase === "error" && (
          <div className="flex items-center gap-1.5 text-[0.75rem] text-[#7A1E13]">
            <XCircle className="size-3 shrink-0 text-[#9B2C1F]" aria-hidden="true" />
            <span>{msg("auto.features.agent.panel.components.librarydatasetcard.error")}</span>
          </div>
        )}

        <button
          type="button"
          onClick={() => setOpen(true)}
          disabled={phase === "loading"}
          className="inline-flex items-center gap-1.5 rounded-lg border border-[#C8A882]/50 bg-white/70 px-3 py-1.5 text-[0.75rem] font-medium text-[#3D2E22] transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
        >
          {phase === "loading" ? (
            <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
          ) : (
            <Database className="size-3.5" aria-hidden="true" />
          )}
          {msg("auto.features.agent.panel.components.librarydatasetcard.pick")}
        </button>
      </div>

      <DatasetPickerDialog open={open} onOpenChange={setOpen} onPick={handlePick} />
    </div>
  );
}
