"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CaretDown, DownloadSimple, FileCode, FileText, FileXls, Package } from "@/shared/ui/icons";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { downloadProgramExport } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { track, TelemetryEvent } from "@/shared/lib/telemetry";
import type {
  OptimizationLogEntry,
  OptimizationStatusResponse,
  OptimizedPredictor,
} from "@/shared/types/api";

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  // Defer revoke so the browser has time to start the download in slow paths.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function escapeCsvField(value: string | number | null | undefined): string {
  const s = value == null ? "" : String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function exportPromptAsJson(prompt: OptimizedPredictor, optimizationId: string) {
  downloadFile(
    JSON.stringify(prompt, null, 2),
    `prompt_${optimizationId.slice(0, 8)}.json`,
    "application/json",
  );
}

/** Save the GEPA-rewritten Flex module source as a standalone .py download. */
export function exportModuleAsPython(
  moduleSrc: string,
  optimizationId: string,
  componentPath?: string,
) {
  const suffix = componentPath ? `_${componentPath}` : "";
  downloadFile(
    moduleSrc.endsWith("\n") ? moduleSrc : `${moduleSrc}\n`,
    `optimized_module_${optimizationId.slice(0, 8)}${suffix}.py`,
    "text/x-python",
  );
}

/** Decode the artifact's base64 pickle and hand it to the browser as a .pkl download. */
export function downloadProgramPickle(pickleBase64: string, optimizationId: string) {
  const blob = new Blob([Uint8Array.from(atob(pickleBase64), (c) => c.charCodeAt(0))], {
    type: "application/octet-stream",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `program_${optimizationId.slice(0, 8)}.pkl`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportLogsAsCsv(logs: OptimizationLogEntry[], optimizationId: string) {
  const header = "timestamp,level,logger,message\n";
  const rows = logs
    .map(
      (l) =>
        `${escapeCsvField(l.timestamp)},${escapeCsvField(l.level)},${escapeCsvField(
          l.logger,
        )},${escapeCsvField(l.message)}`,
    )
    .join("\n");
  downloadFile(header + rows, `logs_${optimizationId.slice(0, 8)}.csv`, "text/csv");
}

export function ExportMenu({
  job,
  optimizedPrompt,
  optimizedModuleSrc,
  optimizedComponentSrcs,
  pickleBase64,
  programPairIndex,
  isShare,
}: {
  job: OptimizationStatusResponse;
  optimizedPrompt: OptimizedPredictor | null;
  /** GEPA-rewritten Flex module source, offered as a standalone .py download. */
  optimizedModuleSrc?: string | null;
  /** Per-submodule Flex sources (a workflow's flex nodes), one .py download each. */
  optimizedComponentSrcs?: Record<string, string>;
  /** Pair-scoped pickle override; falls back to the run / grid-best artifact. */
  pickleBase64?: string | null;
  /** Pair selected for a runnable grid-search export; omitted for single runs. */
  programPairIndex?: number;
  /** The /program-export endpoint is authed, so the public share view hides the ZIP item. */
  isShare?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const pklB64 =
    pickleBase64 ??
    job.result?.program_artifact?.program_pickle_base64 ??
    job.grid_result?.best_pair?.program_artifact?.program_pickle_base64 ??
    null;
  const hasPkl = !!pklB64;
  const selectedGridPair =
    job.grid_result?.pair_results.find((pair) => pair.pair_index === programPairIndex) ??
    job.grid_result?.best_pair;
  const hasProgram =
    !isShare &&
    !!(
      job.result?.program_artifact?.program_state_json ||
      selectedGridPair?.program_artifact?.program_state_json
    );
  // A top-level Flex is one nameless download; a workflow's flex nodes are one each.
  const moduleDownloads: Array<[string, string]> = optimizedModuleSrc
    ? [["", optimizedModuleSrc]]
    : Object.entries(optimizedComponentSrcs ?? {});
  const hasModuleSrc = moduleDownloads.length > 0;
  const itemCls =
    "flex min-h-[44px] w-full items-center gap-2.5 px-3.5 py-2 text-[0.75rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors";
  const iconCls = "size-4 shrink-0 text-muted-foreground/60";
  const extCls = "text-muted-foreground/60 font-mono text-[0.625rem] ms-auto";
  const divider = <div className="h-px bg-border/40 mx-2 my-1" />;

  return (
    <div className="relative" ref={ref}>
      <Button
        size="sm"
        onClick={() => setOpen((o) => !o)}
        data-telemetry="results-export-menu"
        className="min-h-[44px] gap-1.5 sm:min-h-0 [@media(hover:none)_and_(pointer:coarse)]:min-h-[44px]"
      >
        <DownloadSimple className="size-4" />
        {msg("auto.features.optimizations.components.exportmenu.1")}
        <CaretDown
          className={`size-3.5 transition-transform duration-150 ${open ? "rotate-180" : ""}`}
        />
      </Button>
      <AnimatePresence>
        {open && (
          <motion.div
            role="menu"
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -4 }}
            transition={{ duration: 0.12 }}
            className="absolute end-0 top-full mt-1.5 z-50 min-w-[180px] max-w-[min(240px,90vw)] rounded-2xl border border-border/40 bg-card shadow-[0_4px_24px_rgba(28,22,18,0.1)] py-1.5"
          >
            {hasProgram && (
              <button
                type="button"
                role="menuitem"
                onClick={async () => {
                  setOpen(false);
                  try {
                    await downloadProgramExport(job.optimization_id, programPairIndex);
                    track(TelemetryEvent.ArtifactDownloaded, { kind: "program_zip" });
                  } catch (err) {
                    toast.error(
                      err instanceof Error ? err.message : msg("optimization.file.parse_error"),
                    );
                  }
                }}
                className={itemCls}
              >
                <FileCode className={iconCls} />
                <span className="flex-1">
                  {msg("auto.features.optimizations.components.exportmenu.8")}
                </span>
                <span className={extCls}>
                  {msg("auto.features.optimizations.components.exportmenu.9")}
                </span>
              </button>
            )}
            {hasPkl && (
              <>
                {hasProgram && divider}
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    if (!pklB64) return;
                    try {
                      downloadProgramPickle(pklB64, job.optimization_id);
                      track(TelemetryEvent.ArtifactDownloaded, { kind: "program_pickle" });
                    } catch {
                      toast.error(msg("optimization.file.parse_error"));
                    }
                  }}
                  className={itemCls}
                >
                  <Package className={iconCls} />
                  <span className="flex-1">
                    {msg("auto.features.optimizations.components.exportmenu.2")}
                  </span>
                  <span className={extCls}>
                    {msg("auto.features.optimizations.components.exportmenu.3")}
                  </span>
                </button>
              </>
            )}
            {optimizedPrompt && (
              <>
                {(hasProgram || hasPkl) && divider}
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    exportPromptAsJson(optimizedPrompt, job.optimization_id);
                    track(TelemetryEvent.ArtifactDownloaded, { kind: "prompt_json" });
                  }}
                  className={itemCls}
                >
                  <FileText className={iconCls} />
                  <span className="flex-1">
                    {msg("auto.features.optimizations.components.exportmenu.4")}
                  </span>
                  <span className={extCls}>
                    {msg("auto.features.optimizations.components.exportmenu.5")}
                  </span>
                </button>
              </>
            )}
            {hasModuleSrc && (
              <>
                {(hasProgram || hasPkl || optimizedPrompt) && divider}
                {moduleDownloads.map(([path, source]) => (
                  <button
                    key={path}
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setOpen(false);
                      exportModuleAsPython(source, job.optimization_id, path || undefined);
                      track(TelemetryEvent.ArtifactDownloaded, { kind: "module_python" });
                    }}
                    className={itemCls}
                  >
                    <FileCode className={iconCls} />
                    <span className="flex-1">{msg("optimizations.flex.optimized_code")}</span>
                    {path && (
                      <span className="font-mono text-[0.6875rem] text-muted-foreground" dir="ltr">
                        {path}
                      </span>
                    )}
                    <span className={extCls}>{msg("optimizations.flex.py_ext")}</span>
                  </button>
                ))}
              </>
            )}
            {job.logs && job.logs.length > 0 && (
              <>
                {(hasProgram || hasPkl || optimizedPrompt || hasModuleSrc) && divider}
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    exportLogsAsCsv(job.logs, job.optimization_id);
                    track(TelemetryEvent.ArtifactDownloaded, { kind: "logs_csv" });
                  }}
                  className={itemCls}
                >
                  <FileXls className={iconCls} />
                  <span className="flex-1">
                    {msg("auto.features.optimizations.components.exportmenu.6")}
                  </span>
                  <span className={extCls}>
                    {msg("auto.features.optimizations.components.exportmenu.7")}
                  </span>
                </button>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
