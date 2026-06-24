"use client";

import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  ArrowRight,
  Check,
  CheckCircle2,
  FileSpreadsheet,
  Image as ImageIcon,
  Loader2,
  RotateCw,
  Type as TypeIcon,
  Upload,
} from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { parseDatasetFile } from "@/shared/lib/parse-dataset";

import type { AgentToolCall } from "@/shared/ui/agent/types";

export type ColumnRole = "input" | "output" | "ignore";
export type ColumnKind = "text" | "image";

export interface ConfirmedDataset {
  fileName: string;
  rows: Array<Record<string, unknown>>;
  rowCount: number;
  columns: string[];
  columnRoles: Record<string, ColumnRole>;
  columnKinds: Record<string, ColumnKind>;
}

interface ParsedFile {
  fileName: string;
  rows: Array<Record<string, unknown>>;
  rowCount: number;
  columns: string[];
}

interface DatasetUploadCardProps {
  call: AgentToolCall;
  disabled?: boolean;
  alreadyConfirmed?: boolean;
  onConfirm: (confirmed: ConfirmedDataset) => void;
}

const FILE_ACCEPT = ".csv,.json,.jsonl,.xlsx,.xls";

function defaultRoleFor(idx: number, total: number): ColumnRole {
  if (total === 1) return "input";
  return idx === total - 1 ? "output" : "input";
}

export function DatasetUploadCard({
  call,
  disabled,
  alreadyConfirmed,
  onConfirm,
}: DatasetUploadCardProps) {
  const reduceMotion = useReducedMotion();
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [parsed, setParsed] = React.useState<ParsedFile | null>(null);
  const [roles, setRoles] = React.useState<Record<string, ColumnRole>>({});
  const [kinds, setKinds] = React.useState<Record<string, ColumnKind>>({});
  const [parsing, setParsing] = React.useState(false);
  const [dragOver, setDragOver] = React.useState(false);
  const [parseError, setParseError] = React.useState<string | null>(null);
  const [validationError, setValidationError] = React.useState<string | null>(null);
  const [confirmed, setConfirmed] = React.useState(!!alreadyConfirmed);

  React.useEffect(() => {
    if (alreadyConfirmed) setConfirmed(true);
  }, [alreadyConfirmed]);

  const handleFile = React.useCallback(async (file: File) => {
    setParsing(true);
    setParseError(null);
    setValidationError(null);
    try {
      const result = await parseDatasetFile(file);
      if (!result.columns.length) throw new Error("empty");
      const initialRoles: Record<string, ColumnRole> = {};
      result.columns.forEach((c, idx) => {
        initialRoles[c] = defaultRoleFor(idx, result.columns.length);
      });
      setParsed({
        fileName: file.name,
        rows: result.rows,
        rowCount: result.rowCount,
        columns: result.columns,
      });
      setRoles(initialRoles);
      const initialKinds: Record<string, ColumnKind> = {};
      result.columns.forEach((c) => {
        initialKinds[c] = "text";
      });
      setKinds(initialKinds);
    } catch {
      setParsed(null);
      setRoles({});
      setKinds({});
      setParseError(msg("auto.features.agent.panel.components.datasetuploadcard.parse_error"));
    } finally {
      setParsing(false);
    }
  }, []);

  const onPickClick = React.useCallback(() => {
    inputRef.current?.click();
  }, []);

  const onInputChange = React.useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) await handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = React.useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);
  const onDragLeave = React.useCallback(() => setDragOver(false), []);
  const onDrop = React.useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) await handleFile(file);
    },
    [handleFile],
  );

  const setRoleFor = React.useCallback(
    (col: string, role: ColumnRole) => {
      if (confirmed) return;
      setRoles((prev) => ({ ...prev, [col]: role }));
      setValidationError(null);
    },
    [confirmed],
  );

  const toggleKindFor = React.useCallback(
    (col: string) => {
      if (confirmed) return;
      setKinds((prev) => ({
        ...prev,
        [col]: (prev[col] ?? "text") === "image" ? "text" : "image",
      }));
    },
    [confirmed],
  );

  const counts = React.useMemo(() => {
    let input = 0;
    let output = 0;
    let ignore = 0;
    Object.values(roles).forEach((r) => {
      if (r === "input") input += 1;
      else if (r === "output") output += 1;
      else ignore += 1;
    });
    return { input, output, ignore };
  }, [roles]);

  const canConfirm = parsed !== null && counts.input >= 1 && counts.output >= 1 && !confirmed;

  const handleConfirm = React.useCallback(() => {
    if (!parsed) return;
    if (counts.input < 1 || counts.output < 1) {
      setValidationError(msg("auto.features.agent.panel.components.datasetuploadcard.need_io"));
      return;
    }
    setConfirmed(true);
    const finalKinds: Record<string, ColumnKind> = {};
    for (const col of parsed.columns) {
      finalKinds[col] = roles[col] === "input" ? (kinds[col] ?? "text") : "text";
    }
    onConfirm({
      fileName: parsed.fileName,
      rows: parsed.rows,
      rowCount: parsed.rowCount,
      columns: parsed.columns,
      columnRoles: { ...roles },
      columnKinds: finalKinds,
    });
  }, [counts.input, counts.output, kinds, onConfirm, parsed, roles]);

  const handleReset = React.useCallback(() => {
    if (confirmed) return;
    setParsed(null);
    setRoles({});
    setKinds({});
    setParseError(null);
    setValidationError(null);
  }, [confirmed]);

  const fade = reduceMotion
    ? { initial: false, animate: { opacity: 1 }, exit: { opacity: 0 } }
    : {
        initial: { opacity: 0, y: 4 },
        animate: { opacity: 1, y: 0 },
        exit: { opacity: 0, y: -4 },
        transition: { duration: 0.18, ease: [0.2, 0.8, 0.2, 1] as const },
      };

  return (
    <div className="w-full">
      <motion.div
        layout={!reduceMotion}
        className={cn(
          "relative overflow-hidden rounded-2xl border bg-gradient-to-br from-[#FAF8F5] to-[#F5EFE6]",
          "border-[#C8A882]/40",
          "shadow-[0_4px_16px_rgba(61,46,34,0.06)]",
          confirmed && "from-[#F4F0E8] to-[#EEE6D7] border-[#3D2E22]/30",
        )}
      >
        <div className="px-4 pt-3.5 pb-2.5 border-b border-[#C8A882]/25 bg-white/40">
          <div className="flex items-start gap-2.5">
            <div
              className={cn(
                "shrink-0 size-7 rounded-full inline-flex items-center justify-center",
                confirmed
                  ? "bg-[#3D2E22] text-[#FAF8F5]"
                  : "bg-[#C8A882]/25 text-[#3D2E22]",
              )}
            >
              {confirmed ? <Check className="size-3.5" /> : <Upload className="size-3.5" />}
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[0.8125rem] font-semibold text-[#3D2E22] leading-tight">
                {confirmed
                  ? msg("auto.features.agent.panel.components.datasetuploadcard.confirmed")
                  : msg("auto.features.agent.panel.components.datasetuploadcard.title")}
              </div>
            </div>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={FILE_ACCEPT}
          className="hidden"
          onChange={onInputChange}
          disabled={disabled || parsing || confirmed}
        />

        <AnimatePresence mode="wait" initial={false}>
          {!parsed && !parsing && (
            <motion.div key="empty" {...fade} className="px-4 py-3.5">
              <button
                type="button"
                onClick={onPickClick}
                onDragOver={onDragOver}
                onDragEnter={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                disabled={disabled}
                className={cn(
                  "group w-full rounded-xl border-2 border-dashed",
                  "px-4 py-5 flex flex-col items-center justify-center gap-2",
                  "cursor-pointer transition-all duration-150 outline-none",
                  "focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#FAF8F5]",
                  dragOver
                    ? "border-[#3D2E22] bg-[#C8A882]/15 scale-[1.01]"
                    : "border-[#C8A882]/50 bg-white/50 hover:border-[#3D2E22]/60 hover:bg-white/80",
                  "disabled:opacity-50 disabled:cursor-not-allowed",
                )}
              >
                <div
                  className={cn(
                    "size-9 rounded-full inline-flex items-center justify-center transition-colors",
                    dragOver
                      ? "bg-[#3D2E22] text-[#FAF8F5]"
                      : "bg-[#C8A882]/20 text-[#3D2E22] group-hover:bg-[#C8A882]/35",
                  )}
                >
                  <Upload className="size-4" />
                </div>
                <div className="text-center">
                  <div className="text-[0.8125rem] font-medium text-[#3D2E22]">
                    {msg("auto.features.agent.panel.components.datasetuploadcard.drop_label")}
                  </div>
                </div>
              </button>
              {parseError && (
                <div className="mt-2 text-[0.6875rem] text-red-600/90 bg-red-50/80 border border-red-100 rounded-md px-2.5 py-1.5">
                  {parseError}
                </div>
              )}
            </motion.div>
          )}

          {parsing && (
            <motion.div
              key="parsing"
              {...fade}
              className="px-4 py-6 flex items-center justify-center gap-2 text-[0.75rem] text-[#6B5B4A]"
            >
              <Loader2 className="size-3.5 animate-spin" />
              {msg("auto.features.agent.panel.components.datasetuploadcard.parsing")}
            </motion.div>
          )}

          {parsed && (
            <motion.div key="parsed" {...fade} className="px-4 py-3 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <FileSpreadsheet className="size-3.5 text-[#3D2E22] shrink-0" />
                  <div className="min-w-0">
                    <div className="text-[0.75rem] font-medium text-[#3D2E22] truncate">
                      {parsed.fileName}
                    </div>
                    <div className="text-[0.6875rem] text-[#8A7866] tabular-nums leading-tight">
                      {parsed.rowCount}{" "}
                      {msg("auto.features.agent.panel.components.datasetuploadcard.rows_suffix")} ·{" "}
                      {parsed.columns.length}{" "}
                      {msg("auto.features.agent.panel.components.datasetuploadcard.columns_suffix")}
                    </div>
                  </div>
                </div>
                {!confirmed && (
                  <button
                    type="button"
                    onClick={handleReset}
                    aria-label={msg(
                      "auto.features.agent.panel.components.datasetuploadcard.replace",
                    )}
                    className={cn(
                      "shrink-0 inline-flex items-center gap-1 text-[0.6875rem]",
                      "rounded-md px-1.5 py-1 text-[#6B5B4A] hover:text-[#3D2E22]",
                      "hover:bg-white/70 transition-colors cursor-pointer",
                    )}
                  >
                    <RotateCw className="size-3" />
                    {msg("auto.features.agent.panel.components.datasetuploadcard.replace")}
                  </button>
                )}
              </div>

              {!confirmed && (
                <div className="text-[0.6875rem] text-[#8A7866]">
                  {msg("auto.features.agent.panel.components.datasetuploadcard.role_legend")}
                </div>
              )}

              <div className="space-y-1.5">
                {parsed.columns.map((col) => {
                  const role = roles[col] ?? "ignore";
                  const kind = kinds[col] ?? "text";
                  return (
                    <ColumnRoleRow
                      key={col}
                      column={col}
                      role={role}
                      kind={kind}
                      disabled={confirmed}
                      onChangeRole={(next) => setRoleFor(col, next)}
                      onToggleKind={() => toggleKindFor(col)}
                    />
                  );
                })}
              </div>

              {validationError && (
                <div className="text-[0.6875rem] text-red-600/90 bg-red-50/80 border border-red-100 rounded-md px-2.5 py-1.5">
                  {validationError}
                </div>
              )}

              {!confirmed && (
                <button
                  type="button"
                  onClick={handleConfirm}
                  disabled={!canConfirm || disabled}
                  className={cn(
                    "w-full inline-flex items-center justify-center gap-1.5",
                    "rounded-xl px-3 py-2 text-[0.8125rem] font-medium cursor-pointer",
                    "bg-[#3D2E22] text-[#FAF8F5] hover:bg-[#2A1F16] transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#FAF8F5]",
                    "disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-[#3D2E22]",
                  )}
                >
                  {msg("auto.features.agent.panel.components.datasetuploadcard.confirm")}
                  <ArrowRight className="size-3.5 rotate-180" />
                </button>
              )}

              {confirmed && (
                <div className="flex items-center gap-1.5 text-[0.6875rem] text-[#3D2E22]/80 bg-white/50 rounded-md px-2.5 py-1.5">
                  <CheckCircle2 className="size-3 text-[#3D2E22]" />
                  {formatMsg("auto.features.agent.panel.components.datasetuploadcard.confirmed", {})}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

interface ColumnRoleRowProps {
  column: string;
  role: ColumnRole;
  kind: ColumnKind;
  disabled?: boolean;
  onChangeRole: (role: ColumnRole) => void;
  onToggleKind: () => void;
}

// Three-segment role switch with a sliding active indicator.
// Mirrors the wizard's column-role toggle in DatasetStep so the two
// surfaces share the same visual grammar.
function ColumnRoleRow({
  column,
  role,
  kind,
  disabled,
  onChangeRole,
  onToggleKind,
}: ColumnRoleRowProps) {
  const options = [
    ["input", msg("auto.features.agent.panel.components.datasetuploadcard.role_input")],
    ["output", msg("auto.features.agent.panel.components.datasetuploadcard.role_output")],
    ["ignore", msg("auto.features.agent.panel.components.datasetuploadcard.role_skip")],
  ] as const;
  const activeIdx = options.findIndex(([v]) => v === role);
  const pillLeft = activeIdx >= 0 ? `calc(${activeIdx} * 100% / 3 + 2px)` : "2px";
  const isInput = role === "input";
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="text-[0.75rem] font-mono truncate" dir="ltr">
          {column}
        </span>
        {isInput && (
          <button
            type="button"
            onClick={onToggleKind}
            disabled={disabled}
            className={cn(
              "shrink-0 inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[0.625rem] font-medium transition-colors cursor-pointer",
              kind === "image"
                ? "border-primary/40 bg-primary/10 text-primary hover:bg-primary/15"
                : "border-border/60 bg-muted/40 text-muted-foreground hover:border-primary/30 hover:text-foreground",
              disabled && "cursor-not-allowed opacity-60",
            )}
            title={
              kind === "image"
                ? msg("submit.dataset.column_kind.image")
                : msg("submit.dataset.column_kind.text_manual_hint")
            }
          >
            {kind === "image" ? (
              <ImageIcon className="size-3" />
            ) : (
              <TypeIcon className="size-3" />
            )}
            <span>
              {kind === "image"
                ? msg("submit.dataset.column_kind.image")
                : msg("submit.dataset.column_kind.text")}
            </span>
          </button>
        )}
      </div>
      <div
        className="relative inline-grid grid-cols-3 shrink-0 rounded-lg bg-muted p-0.5 gap-0.5"
      >
        <div
          className="absolute top-0.5 bottom-0.5 rounded-md bg-stone-500/15 shadow-sm transition-[inset-inline-start] duration-100 ease-out"
          style={{
            width: "calc((100% - 6px) / 3)",
            insetInlineStart: pillLeft,
          }}
        />
        {options.map(([val, label]) => (
          <button
            key={val}
            type="button"
            onClick={() => onChangeRole(val)}
            disabled={disabled}
            className={cn(
              "relative z-10 rounded-md px-2.5 py-1 text-[0.6875rem] font-medium text-center transition-colors duration-100 cursor-pointer",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40",
              role === val ? "text-stone-600" : "text-muted-foreground hover:text-foreground",
              disabled && "cursor-not-allowed opacity-60",
            )}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

