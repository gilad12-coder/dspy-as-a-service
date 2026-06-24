"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ChevronRight,
  ChevronLeft,
  SkipBack,
  CircleMinus,
  Database,
  Download,
  Keyboard,
  Loader2,
} from "lucide-react";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardTitle } from "@/shared/ui/primitives/card";
import { Badge } from "@/shared/ui/primitives/badge";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/shared/ui/primitives/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Popover as PopoverPrimitive } from "radix-ui";
import { cn } from "@/shared/lib/utils";
import { exportAnnotations, buildLibraryRows } from "../lib/export-csv";
import type { DataField, DataRow, Annotation, TaggerConfig } from "../lib/types";
import { isStorageQuotaError, saveDataset } from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";

interface Props {
  config: TaggerConfig;
  data: DataRow[];
  columns: string[];
  annotations: Record<string, Annotation>;
  currentIndex: number;
  taggedCount: number;
  onNavigate: (dir: 1 | -1) => void;
  onGoTo: (idx: number) => void;
  onJumpUntagged: () => void;
  onToggleBinary: (id: string, value: "yes" | "no") => void;
  onToggleCategory: (id: string, catId: string) => void;
  onSetFreetext: (id: string, text: string) => void;
  onBack: () => void;
}

export function TaggerAnnotation({
  config,
  data,
  columns,
  annotations,
  currentIndex,
  taggedCount,
  onNavigate,
  onGoTo,
  onJumpUntagged,
  onToggleBinary,
  onToggleCategory,
  onSetFreetext,
  onBack,
}: Props) {
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [exportConfirm, setExportConfirm] = useState<"csv" | "json" | "xlsx" | "xls" | null>(null);
  const [nameDialogOpen, setNameDialogOpen] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [savingToLibrary, setSavingToLibrary] = useState(false);
  const confettiFired = useRef(false);
  const confettiTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const item = data[currentIndex];
  const id = item ? String(item.id) : "";
  const pct = data.length > 0 ? (taggedCount / data.length) * 100 : 0;
  const currentAnn = annotations[id];

  const showConfettiBriefly = useCallback(() => {
    setShowConfetti(true);
    if (confettiTimerRef.current) clearTimeout(confettiTimerRef.current);
    confettiTimerRef.current = setTimeout(() => setShowConfetti(false), 4000);
  }, []);

  useEffect(
    () => () => {
      if (confettiTimerRef.current) clearTimeout(confettiTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    if (taggedCount === data.length && data.length > 0 && !confettiFired.current) {
      confettiFired.current = true;
      showConfettiBriefly();
    }
    if (taggedCount < data.length) confettiFired.current = false;
  }, [taggedCount, data.length, showConfettiBriefly]);

  const doExport = useCallback(
    (format: "csv" | "json" | "xlsx" | "xls") => {
      void exportAnnotations(data, columns, annotations, config, format).then(
        showConfettiBriefly,
      );
    },
    [data, columns, annotations, config, showConfettiBriefly],
  );

  const handleExport = useCallback(
    (format: "csv" | "json" | "xlsx" | "xls") => {
      if (taggedCount < data.length) {
        setExportConfirm(format);
      } else {
        doExport(format);
      }
    },
    [data.length, taggedCount, doExport],
  );

  const defaultLibraryName = useCallback(
    () => `tagging_${config.mode}_${new Date().toISOString().slice(0, 10)}`,
    [config.mode],
  );

  const openNameDialog = useCallback(() => {
    setDatasetName(defaultLibraryName());
    setNameDialogOpen(true);
  }, [defaultLibraryName]);

  const handleSaveToLibrary = useCallback(async () => {
    if (savingToLibrary) return;
    const name = datasetName.trim() || defaultLibraryName();
    setSavingToLibrary(true);
    try {
      const { rows, columnOrder, columnRoles } = buildLibraryRows(
        data,
        columns,
        annotations,
        config,
      );
      const res = await saveDataset({
        name,
        source: "tagger",
        dataset: rows,
        column_schema: { column_order: columnOrder, column_roles: columnRoles },
      });
      toast.success(
        res.deduplicated
          ? msg("datasets.toast.deduplicated")
          : formatMsg("tagger.library.saved", { name: res.dataset.name }),
      );
      setNameDialogOpen(false);
    } catch (err) {
      if (!isStorageQuotaError(err)) {
        toast.error(err instanceof Error ? err.message : msg("tagger.library.save_failed"));
      }
    } finally {
      setSavingToLibrary(false);
    }
  }, [savingToLibrary, datasetName, defaultLibraryName, data, columns, annotations, config]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "h") {
        e.preventDefault();
        setShowShortcuts((v) => !v);
        return;
      }
      if (e.key === "Escape") {
        if (showShortcuts) {
          setShowShortcuts(false);
          e.preventDefault();
          return;
        }
      }

      const tag = (e.target as HTMLElement).tagName;
      if (tag === "TEXTAREA" || tag === "INPUT") {
        if (e.key === "Escape") (e.target as HTMLElement).blur();
        return;
      }

      if (!id) return;

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onNavigate(1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onNavigate(-1);
      } else if (e.key === "Home") {
        e.preventDefault();
        onGoTo(0);
      } else if (e.key === "u" || e.key === "U") {
        e.preventDefault();
        onJumpUntagged();
      } else if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        handleExport("csv");
      } else if (config.mode === "binary") {
        if (e.key === "y" || e.key === "Y") {
          e.preventDefault();
          onToggleBinary(id, "yes");
        } else if (e.key === "n" || e.key === "N") {
          e.preventDefault();
          onToggleBinary(id, "no");
        }
      } else if (config.mode === "multiclass") {
        const num = parseInt(e.key);
        if (num >= 1 && num <= 9) {
          const cats = config.categories ?? [];
          if (num <= cats.length) {
            e.preventDefault();
            onToggleCategory(id, cats[num - 1]!.id);
          }
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    config,
    id,
    showShortcuts,
    onNavigate,
    onGoTo,
    onJumpUntagged,
    onToggleBinary,
    onToggleCategory,
    handleExport,
  ]);

  if (!item) return null;

  return (
    <div className="flex h-[calc(100dvh-var(--header-height,53px)-3rem)] flex-col overflow-hidden md:h-[calc(100dvh-var(--header-height,53px)-4rem)]">
      <div className="flex items-center gap-2 px-5 pt-3 pb-1.5">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${pct}%`,
              background: "var(--gradient-progress)",
            }}
          />
        </div>
        <span className="text-xs text-muted-foreground tabular-nums shrink-0">
          <span className="font-semibold text-primary">{taggedCount}</span>/{data.length}
        </span>
      </div>

      <div className="flex flex-1 flex-col gap-3 p-5 overflow-hidden">
        <Card className="flex flex-1 min-h-0 flex-col">
          <CardContent
            className="flex-1 overflow-y-auto px-6 py-5 text-base leading-relaxed text-foreground"
            dir="auto"
          >
            {item.fields && item.fields.length > 1 ? (
              <FieldsView fields={item.fields} />
            ) : (
              <div className="whitespace-pre-wrap">{item.text}</div>
            )}
          </CardContent>
        </Card>

        <Card className="flex flex-1 min-h-0 flex-col p-5">
          <CardTitle className="mb-3 text-center text-sm font-medium text-muted-foreground">
            {config.mode === "binary" &&
              (config.question ??
                msg("auto.features.tagger.components.taggerannotation.literal.1"))}
            {config.mode === "multiclass" &&
              msg("auto.features.tagger.components.taggerannotation.literal.2")}
            {config.mode === "freetext" &&
              (config.prompt ?? msg("auto.features.tagger.components.taggerannotation.literal.3"))}
          </CardTitle>

          {config.mode === "binary" && (
            <div className="flex flex-1 min-h-0 flex-col gap-2">
              <Button
                variant={currentAnn === "yes" ? "default" : "outline"}
                onClick={() => onToggleBinary(id, "yes")}
                className={cn(
                  "flex-1 text-base font-medium rounded-xl gap-2 focus-visible:ring-0 focus-visible:border-transparent",
                  currentAnn === "yes" &&
                    "bg-emerald-600/15 hover:bg-emerald-600/20 border-emerald-600/40 text-emerald-700",
                )}
              >
                <Badge variant="ghost" size="sm" className="opacity-40 font-mono">
                  {msg("auto.features.tagger.components.taggerannotation.4")}
                </Badge>
                {msg("auto.features.tagger.components.taggerannotation.5")}
              </Button>
              <Button
                variant={currentAnn === "no" ? "default" : "outline"}
                onClick={() => onToggleBinary(id, "no")}
                className={cn(
                  "flex-1 text-base font-medium rounded-xl gap-2 focus-visible:ring-0 focus-visible:border-transparent",
                  currentAnn === "no" &&
                    "bg-red-500/15 hover:bg-red-500/20 border-red-500/40 text-red-600",
                )}
              >
                <Badge variant="ghost" size="sm" className="opacity-40 font-mono">
                  {msg("auto.features.tagger.components.taggerannotation.6")}
                </Badge>
                {msg("auto.features.tagger.components.taggerannotation.7")}
              </Button>
            </div>
          )}

          {config.mode === "multiclass" && (
            <div
              className={cn(
                "flex flex-1 min-h-0 flex-col gap-1.5 overflow-y-auto",
                (config.categories?.length ?? 0) >= 7 && "gap-1",
              )}
            >
              {(config.categories ?? []).map((cat, i) => {
                const selected = Array.isArray(currentAnn) && currentAnn.includes(cat.id);
                return (
                  <Button
                    key={cat.id}
                    variant="outline"
                    onClick={() => onToggleCategory(id, cat.id)}
                    className={cn(
                      "flex-1 min-h-0 min-w-0 rounded-xl gap-2 whitespace-normal focus-visible:ring-0 focus-visible:border-transparent",
                      (config.categories?.length ?? 0) >= 7 ? "text-sm" : "text-base",
                      "font-medium",
                      selected && "bg-primary/10 border-primary/40 text-primary",
                    )}
                  >
                    {i < 9 && (
                      <Badge
                        variant="ghost"
                        size="sm"
                        className={cn("font-mono", selected ? "opacity-70" : "opacity-40")}
                      >
                        {i + 1}
                      </Badge>
                    )}
                    <span className="min-w-0 break-words">{cat.label}</span>
                  </Button>
                );
              })}
            </div>
          )}

          {config.mode === "freetext" && (
            <textarea
              value={typeof currentAnn === "string" ? currentAnn : ""}
              onChange={(e) => onSetFreetext(id, e.target.value)}
              className="flex-1 min-h-0 resize-none rounded-xl border border-input/90 bg-background/75 px-4 py-3 text-sm leading-relaxed shadow-[inset_0_1px_0_rgba(255,255,255,0.72)] backdrop-blur-sm transition-[color,box-shadow,border-color] outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50"
              dir="auto"
            />
          )}
        </Card>

        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={() => onNavigate(-1)}
            disabled={currentIndex === 0}
            className="gap-2"
          >
            <ChevronRight className="size-4" />
            {msg("auto.features.tagger.components.taggerannotation.8")}
          </Button>

          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => onGoTo(0)}
                  aria-label={msg("auto.features.tagger.components.taggerannotation.9")}
                >
                  <SkipBack className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {msg("auto.features.tagger.components.taggerannotation.9")}
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={onJumpUntagged}
                  disabled={taggedCount === data.length}
                  aria-label={msg("auto.features.tagger.components.taggerannotation.10")}
                >
                  <CircleMinus className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {msg("auto.features.tagger.components.taggerannotation.10")}
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => setShowShortcuts(true)}
                  aria-label={msg("auto.features.tagger.components.taggerannotation.11")}
                >
                  <Keyboard className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {msg("auto.features.tagger.components.taggerannotation.11")}
              </TooltipContent>
            </Tooltip>
            <PopoverPrimitive.Root>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverPrimitive.Trigger asChild>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={msg("auto.features.tagger.components.taggerannotation.12")}
                    >
                      <Download className="size-4" />
                    </Button>
                  </PopoverPrimitive.Trigger>
                </TooltipTrigger>
                <TooltipContent>
                  {msg("auto.features.tagger.components.taggerannotation.12")}
                </TooltipContent>
              </Tooltip>
              <PopoverPrimitive.Portal>
                <PopoverPrimitive.Content
                  side="bottom"
                  sideOffset={8}
                  className="z-50 w-44 rounded-lg border bg-background p-1 shadow-lg animate-in fade-in-0 zoom-in-95"
                >
                  <PopoverPrimitive.Close asChild>
                    <button
                      type="button"
                      onClick={openNameDialog}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium text-foreground cursor-pointer transition-colors hover:bg-accent"
                    >
                      <Database className="size-3.5 shrink-0" />
                      {msg("tagger.library.save")}
                    </button>
                  </PopoverPrimitive.Close>
                  <div className="my-1 h-px bg-border" />
                  {(["csv", "json", "xlsx", "xls"] as const).map((fmt) => (
                    <PopoverPrimitive.Close key={fmt} asChild>
                      <button
                        type="button"
                        onClick={() => handleExport(fmt)}
                        className="flex w-full items-center rounded-md px-3 py-1.5 text-xs font-medium text-foreground cursor-pointer transition-colors hover:bg-accent"
                      >
                        {fmt.toUpperCase()}
                      </button>
                    </PopoverPrimitive.Close>
                  ))}
                </PopoverPrimitive.Content>
              </PopoverPrimitive.Portal>
            </PopoverPrimitive.Root>
          </div>

          <Button
            variant="outline"
            onClick={() => onNavigate(1)}
            disabled={currentIndex === data.length - 1}
            className="gap-2"
          >
            {msg("auto.features.tagger.components.taggerannotation.13")}
            <ChevronLeft className="size-4" />
          </Button>
        </div>
      </div>

      <Dialog open={showShortcuts} onOpenChange={setShowShortcuts}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{msg("auto.features.tagger.components.taggerannotation.14")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-1.5">
            {config.mode === "binary" && (
              <>
                <ShortcutRow
                  keys="Y"
                  label={msg("auto.features.tagger.components.taggerannotation.literal.4")}
                />
                <ShortcutRow
                  keys="N"
                  label={msg("auto.features.tagger.components.taggerannotation.literal.5")}
                />
              </>
            )}
            {config.mode === "multiclass" &&
              (config.categories ?? [])
                .slice(0, 9)
                .map((cat, i) => (
                  <ShortcutRow key={cat.id} keys={String(i + 1)} label={cat.label} />
                ))}
            <Separator className="my-2" />
            <ShortcutRow
              keys="← / →"
              label={msg("auto.features.tagger.components.taggerannotation.literal.6")}
            />
            <ShortcutRow
              keys="Home"
              label={msg("auto.features.tagger.components.taggerannotation.literal.7")}
            />
            <ShortcutRow
              keys="U"
              label={msg("auto.features.tagger.components.taggerannotation.literal.8")}
            />
            <ShortcutRow
              keys="E"
              label={msg("auto.features.tagger.components.taggerannotation.literal.9")}
            />
            <ShortcutRow
              keys="Ctrl+H"
              label={msg("auto.features.tagger.components.taggerannotation.literal.10")}
            />
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!exportConfirm}
        onOpenChange={(open) => {
          if (!open) setExportConfirm(null);
        }}
      >
        <DialogContent className="max-w-md sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{msg("auto.features.tagger.components.taggerannotation.15")}</DialogTitle>
            <DialogDescription>
              {msg("auto.features.tagger.components.taggerannotation.16")}{" "}
              <span className="font-mono font-medium text-foreground">
                {data.length - taggedCount}
              </span>{" "}
              {msg("auto.features.tagger.components.taggerannotation.17")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setExportConfirm(null)}
              className="w-full justify-center"
            >
              {msg("auto.features.tagger.components.taggerannotation.18")}
            </Button>
            <Button
              onClick={() => {
                if (exportConfirm) doExport(exportConfirm);
                setExportConfirm(null);
              }}
              className="w-full justify-center"
            >
              {msg("auto.features.tagger.components.taggerannotation.19")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={nameDialogOpen}
        onOpenChange={(open) => {
          if (!savingToLibrary) setNameDialogOpen(open);
        }}
      >
        <DialogContent className="max-w-md sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{msg("tagger.library.name_title")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="tagger-dataset-name">{msg("tagger.library.name_label")}</Label>
            <Input
              id="tagger-dataset-name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && datasetName.trim() && !savingToLibrary) {
                  e.preventDefault();
                  void handleSaveToLibrary();
                }
              }}
              autoFocus
              dir="auto"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setNameDialogOpen(false)}
              disabled={savingToLibrary}
              className="w-full justify-center"
            >
              {msg("tagger.library.name_cancel")}
            </Button>
            <Button
              onClick={() => void handleSaveToLibrary()}
              disabled={savingToLibrary || !datasetName.trim()}
              className="w-full justify-center"
            >
              {savingToLibrary ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                msg("tagger.library.name_save")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {showConfetti && <Confetti />}
    </div>
  );
}

function ShortcutRow({ keys, label }: { keys: string; label: string }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <Badge variant="outline" size="sm" className="font-mono">
        {keys}
      </Badge>
    </div>
  );
}

interface ConfettiPiece {
  shape: string;
  color: string;
  left: number;
  delay: number;
  duration: number;
  size: number;
  elongated: boolean;
}

function Confetti() {
  const [pieces, setPieces] = useState<ConfettiPiece[]>([]);

  useEffect(() => {
    const colors = ["#3d2e22", "#5c4d40", "#8c7a6b", "#a69585", "#ddd6cc"];
    const shapes = ["rounded-full", "rounded-sm", "rounded-sm"];
    setPieces(
      Array.from({ length: 50 }, (_, i) => ({
        shape: shapes[i % shapes.length]!,
        color: colors[i % colors.length]!,
        left: Math.random() * 100,
        delay: Math.random() * 0.5,
        duration: 2 + Math.random() * 2,
        size: 6 + Math.random() * 10,
        elongated: i % 3 === 2,
      })),
    );
  }, []);

  return (
    <div className="pointer-events-none fixed inset-0 z-[9999] overflow-hidden">
      <style>{`
        @keyframes confetti-fall {
          0% { opacity: 1; transform: translateY(-100px) rotate(0deg); }
          100% { opacity: 0; transform: translateY(100vh) rotate(720deg); }
        }
      `}</style>
      {pieces.map((p, i) => (
        <div
          key={i}
          className={cn("absolute opacity-0", p.shape)}
          style={{
            backgroundColor: p.color,
            left: `${p.left}%`,
            width: `${p.size}px`,
            height: p.elongated ? `${p.size * 1.6}px` : `${p.size}px`,
            animation: `confetti-fall ${p.duration}s ${p.delay}s ease-out forwards`,
          }}
        />
      ))}
    </div>
  );
}

function FieldsView({ fields }: { fields: DataField[] }) {
  return (
    <dl className="flex flex-col">
      {fields.map((field, i) => (
        <div
          key={`${field.column}-${i}`}
          className={cn(
            "flex flex-col gap-2 py-3.5 first:pt-0 last:pb-0",
            i > 0 && "border-t border-border/40",
          )}
        >
          <dt>
            <span
              dir="ltr"
              className="inline-flex items-center rounded-md bg-muted/55 px-2 py-0.5 text-[10.5px] font-mono uppercase tracking-[0.08em] text-muted-foreground"
            >
              {field.column}
            </span>
          </dt>
          <dd className="text-foreground" dir="auto">
            <FieldValue value={field.value} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function FieldValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || value === undefined || value === "") {
    return <Empty />;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <Empty />;
    const primitive = value.every((v) => v === null || typeof v !== "object");
    if (primitive) {
      return (
        <ul className="space-y-1 ps-5 list-disc marker:text-muted-foreground/50">
          {value.map((item, i) => (
            <li key={i} className="leading-relaxed">
              <FieldValue value={item} depth={depth + 1} />
            </li>
          ))}
        </ul>
      );
    }
    return (
      <ol className="space-y-1.5">
        {value.map((item, i) => (
          <li
            key={i}
            className="relative rounded-lg border border-border/40 bg-muted/30 px-3 py-2"
          >
            <span
              dir="ltr"
              className="absolute -top-2 start-2 rounded bg-background px-1.5 text-[10px] font-mono tabular-nums text-muted-foreground/80"
            >
              {i + 1}
            </span>
            <FieldValue value={item} depth={depth + 1} />
          </li>
        ))}
      </ol>
    );
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return <Empty />;
    return (
      <dl
        className={cn(
          "grid gap-2",
          depth === 0 && "rounded-lg border border-border/40 bg-muted/30 p-3",
          depth > 0 && "border-s border-border/50 ps-3",
        )}
      >
        {entries.map(([k, v]) => (
          <div key={k} className="grid gap-1">
            <dt>
              <span
                dir="ltr"
                className="inline-flex items-center rounded bg-background/80 px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-[0.06em] text-muted-foreground"
              >
                {k}
              </span>
            </dt>
            <dd className="min-w-0">
              <FieldValue value={v} depth={depth + 1} />
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  if (typeof value === "boolean") {
    return (
      <span
        dir="ltr"
        className={cn(
          "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-mono",
          value
            ? "bg-emerald-600/10 text-emerald-700"
            : "bg-muted text-muted-foreground",
        )}
      >
        {value ? "true" : "false"}
      </span>
    );
  }
  if (typeof value === "number") {
    return <span className="font-mono tabular-nums">{value}</span>;
  }
  const str = String(value);
  const trimmed = str.trim();
  if (
    (trimmed.startsWith("[") && trimmed.endsWith("]")) ||
    (trimmed.startsWith("{") && trimmed.endsWith("}"))
  ) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed !== null && typeof parsed === "object") {
        return <FieldValue value={parsed} depth={depth} />;
      }
    } catch {
      /* not JSON — fall through to plain text */
    }
  }
  return <span className="whitespace-pre-wrap break-words leading-relaxed">{str}</span>;
}

function Empty() {
  return <span className="text-muted-foreground/55">—</span>;
}
