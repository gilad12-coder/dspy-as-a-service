"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { FileUp, Loader2 } from "lucide-react";
import { toast } from "react-toastify";
import { profileDataset, stageDatasetForAgent } from "@/shared/lib/api";
import { parseDatasetFile, type ParsedDataset } from "@/shared/lib/parse-dataset";
import { formatMsg, msg } from "@/shared/lib/messages";
import { useLocale } from "@/shared/providers";
import { formatCredits } from "@/features/billing";
import type { ColumnMapping } from "@/shared/types/api";
import { MIN_ONBOARDING_ROWS } from "../constants";

/**
 * Heuristic input/output split for the onboarding profile call.
 *
 * The user hasn't mapped columns yet, so we apply the common "last column is the
 * target" convention purely to get an honest held-out test count from the
 * profiler. The real mapping is confirmed in the wizard; nothing here is sent
 * with the run.
 */
function guessMapping(columns: string[]): ColumnMapping {
  const inputs: Record<string, string> = {};
  const outputs: Record<string, string> = {};
  columns.forEach((col, i) => {
    if (i === columns.length - 1) outputs[col] = col;
    else inputs[col] = col;
  });
  return { inputs, outputs };
}

/** Roles derived from the same heuristic, in the shape the wizard hydrates. */
function guessRoles(columns: string[]): Record<string, "input" | "output"> {
  const roles: Record<string, "input" | "output"> = {};
  columns.forEach((col, i) => {
    roles[col] = i === columns.length - 1 ? "output" : "input";
  });
  return roles;
}

/**
 * Upload → instant baseline framing, then the free first run.
 *
 * Parses the file in the browser (nothing leaves the device until the run),
 * calls the profiler to learn how many held-out examples the program will be
 * scored against, and surfaces that count as the personal gap the optimizer has
 * to close. "Optimize — your first run is free" stages the rows and hands off to
 * the submit wizard, where the guaranteed first run actually executes.
 */
export function UploadBaseline() {
  const router = useRouter();
  const { locale } = useLocale();
  const inputRef = React.useRef<HTMLInputElement>(null);

  const [parsing, setParsing] = React.useState(false);
  const [staging, setStaging] = React.useState(false);
  const [dataset, setDataset] = React.useState<ParsedDataset | null>(null);
  const [filename, setFilename] = React.useState<string>("");
  const [testCount, setTestCount] = React.useState<number | null>(null);

  const onFile = async (file: File) => {
    setParsing(true);
    setDataset(null);
    setTestCount(null);
    try {
      const parsed = await parseDatasetFile(file);
      if (parsed.rowCount < MIN_ONBOARDING_ROWS || parsed.columns.length < 2) {
        throw new Error("too-small");
      }
      setDataset(parsed);
      setFilename(file.name);
      // The held-out test count is the baseline gap we promise to measure. A
      // failed profile shouldn't block the flow — the wizard re-profiles — so
      // we just leave the count unset and show the row/column summary instead.
      try {
        const { plan } = await profileDataset({
          dataset: parsed.rows as Array<Record<string, unknown>>,
          column_mapping: guessMapping(parsed.columns),
        });
        setTestCount(plan.counts.test);
      } catch {
        setTestCount(null);
      }
    } catch {
      toast.error(msg("onboarding.upload.error"));
    } finally {
      setParsing(false);
    }
  };

  const onOptimize = async () => {
    if (!dataset) return;
    setStaging(true);
    const roles = guessRoles(dataset.columns);
    let stagedId: string | null = null;
    try {
      const res = await stageDatasetForAgent({
        dataset: dataset.rows as Array<Record<string, unknown>>,
        dataset_filename: filename || "dataset.json",
      });
      stagedId = res.staged_dataset_id;
    } catch {
      // Staging is an optimization for large datasets; the wizard can still
      // hydrate from the inline rows below if the id never minted.
    }
    try {
      window.sessionStorage.setItem(
        "wizard:staged-dataset",
        JSON.stringify({
          dataset: dataset.rows,
          dataset_filename: filename || "dataset.json",
          ...(stagedId ? { staged_dataset_id: stagedId } : {}),
          wizard_state: { dataset_columns: dataset.columns, column_roles: roles },
        }),
      );
    } catch {
      /* sessionStorage unavailable — the wizard simply starts empty */
    }
    router.push("/submit");
  };

  return (
    <div className="flex flex-col gap-4">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.json,.jsonl,.xlsx,.xls"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void onFile(file);
          e.target.value = "";
        }}
      />

      {!dataset && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={parsing}
          className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/70 bg-muted/20 px-6 py-10 text-center transition-colors hover:border-[#C8A882]/60 hover:bg-muted/40 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait"
        >
          {parsing ? (
            <Loader2 className="size-6 animate-spin text-muted-foreground" aria-hidden="true" />
          ) : (
            <FileUp className="size-6 text-muted-foreground" aria-hidden="true" />
          )}
          <span className="text-sm font-semibold text-foreground">
            {parsing ? msg("onboarding.upload.parsing") : msg("onboarding.upload.dropzone_label")}
          </span>
          {!parsing && (
            <span className="text-xs text-muted-foreground">
              {msg("onboarding.upload.dropzone_hint")}
            </span>
          )}
        </button>
      )}

      {dataset && (
        <div className="flex flex-col gap-4 rounded-xl border border-border/60 bg-card p-5">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold text-foreground">
              {testCount != null
                ? formatMsg("onboarding.upload.baseline_title", {
                    p1: formatCredits(testCount, locale),
                  })
                : msg("onboarding.upload.title")}
            </span>
            <p className="text-xs text-muted-foreground">
              {testCount != null
                ? formatMsg("onboarding.upload.baseline_desc", {
                    p1: formatCredits(testCount, locale),
                  })
                : formatMsg("onboarding.upload.rows_columns", {
                    p1: formatCredits(dataset.rowCount, locale),
                    p2: formatCredits(dataset.columns.length, locale),
                  })}
            </p>
          </div>
          <span dir="auto" className="text-xs text-muted-foreground" title={filename}>
            {filename} ·{" "}
            {formatMsg("onboarding.upload.rows_columns", {
              p1: formatCredits(dataset.rowCount, locale),
              p2: formatCredits(dataset.columns.length, locale),
            })}
          </span>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <button
          type="button"
          onClick={onOptimize}
          disabled={!dataset || staging}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#3D2E22] px-4 py-2.5 text-sm font-semibold text-[#FAF8F5] transition-colors duration-200 hover:bg-[#2A1F17] cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {staging && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
          {msg("onboarding.cta.run")}
        </button>
        <p className="text-xs text-muted-foreground">{msg("onboarding.cta.free_note")}</p>
      </div>
    </div>
  );
}
