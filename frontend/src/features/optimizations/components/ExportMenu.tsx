"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Download, FileCode, FileJson, FileSpreadsheet, Package } from "lucide-react";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { downloadProgramExport } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
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

export function exportLogsAsCsv(logs: OptimizationLogEntry[], optimizationId: string) {
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
}: {
  job: OptimizationStatusResponse;
  optimizedPrompt: OptimizedPredictor | null;
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

  const hasPkl = !!(
    job.result?.program_artifact?.program_pickle_base64 ||
    job.grid_result?.best_pair?.program_artifact?.program_pickle_base64
  );
  // The runnable export reconstructs from state JSON + signature_code, which the
  // /program-export endpoint serves for single-run (non-grid) jobs only.
  const hasProgram = !!job.result?.program_artifact?.program_state_json;
  const itemCls =
    "w-full flex items-center gap-2.5 px-3.5 py-2 text-[0.75rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors";
  const iconCls = "size-4 shrink-0 text-muted-foreground/60";
  const extCls = "text-muted-foreground/60 font-mono text-[0.625rem] ms-auto";
  const divider = <div className="h-px bg-border/40 mx-2 my-1" />;

  return (
    <div className="relative" ref={ref}>
      <Button size="sm" onClick={() => setOpen((o) => !o)} className="gap-1.5">
        <Download className="size-4" />
        {msg("auto.features.optimizations.components.exportmenu.1")}
        <ChevronDown
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
                    await downloadProgramExport(job.optimization_id);
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
                    const b64 =
                      job.result?.program_artifact?.program_pickle_base64 ??
                      job.grid_result?.best_pair?.program_artifact?.program_pickle_base64;
                    if (!b64) return;
                    try {
                      const blob = new Blob([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], {
                        type: "application/octet-stream",
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = `program_${job.optimization_id.slice(0, 8)}.pkl`;
                      a.click();
                      setTimeout(() => URL.revokeObjectURL(url), 0);
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
                  }}
                  className={itemCls}
                >
                  <FileJson className={iconCls} />
                  <span className="flex-1">
                    {msg("auto.features.optimizations.components.exportmenu.4")}
                  </span>
                  <span className={extCls}>
                    {msg("auto.features.optimizations.components.exportmenu.5")}
                  </span>
                </button>
              </>
            )}
            {job.logs && job.logs.length > 0 && (
              <>
                {(hasProgram || hasPkl || optimizedPrompt) && divider}
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    exportLogsAsCsv(job.logs, job.optimization_id);
                  }}
                  className={itemCls}
                >
                  <FileSpreadsheet className={iconCls} />
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
