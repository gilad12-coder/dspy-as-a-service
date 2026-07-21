"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  Upload,
  Binary,
  ListChecks,
  TextCursorInput,
  Plus,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Check,
  Database,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/shared/ui/primitives/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/shared/ui/primitives/card";
import { Separator } from "@/shared/ui/primitives/separator";
import { cn } from "@/shared/lib/utils";
import { HelpTip } from "@/shared/ui/help-tip";
import { tip } from "@/shared/lib/tooltips";
import { parseDatasetFile } from "@/shared/lib/parse-dataset";
import { getDatasetRows, listDatasets, type DatasetSummary } from "@/shared/lib/api";
import { registerTutorialHook, registerTutorialQuery } from "@/features/tutorial";
import { ModelPicker } from "@/features/submit";
import { useUserPrefs } from "@/features/settings";
import type {
  AnnotationMode,
  TaggerAssistMode,
  TaggerConfig,
  DataRow,
  Category,
} from "../lib/types";
import { isTaggerAssistEnabled } from "../lib/feature-flag";
import { REVIEW_BATCH_SIZE, calibrationTarget } from "../lib/assist";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";
import { getActiveDir } from "@/shared/lib/runtime-locale";

interface TaggerSetupProps {
  onStart: (
    config: TaggerConfig,
    rows: DataRow[],
    columns: string[],
    assistMode?: TaggerAssistMode,
    assistModel?: string,
  ) => void;
}

const BASE_STEPS = perLocale(
  () =>
    [{ id: "data", label: msg("auto.features.tagger.components.taggersetup.literal.1") }] as const,
);

const TASK_STEP = perLocale(
  () =>
    ({
      id: "task",
      label: msg("auto.features.tagger.components.taggersetup.literal.2"),
    }) as const,
);

const ASSIST_STEP = perLocale(
  () => ({ id: "assist", label: msg("tagger.assist.setup.step_label") }) as const,
);

const ASSIST_OPTIONS: Array<{
  mode: TaggerAssistMode;
  label: string;
  desc: string;
  recommended?: boolean;
}> = perLocale(() => [
  {
    mode: "manual",
    label: msg("tagger.assist.setup.manual_label"),
    desc: msg("tagger.assist.setup.manual_desc"),
  },
  {
    mode: "copilot",
    label: msg("tagger.assist.setup.copilot_label"),
    desc: msg("tagger.assist.setup.copilot_desc"),
    recommended: true,
  },
  {
    mode: "autopilot",
    label: msg("tagger.assist.setup.autopilot_label"),
    desc: msg("tagger.assist.setup.autopilot_desc"),
  },
]);

/** "support_tickets.csv" → "support tickets" — a readable session-card name. */
function cleanSourceName(fileName: string): string {
  const base = fileName
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .trim();
  return base || fileName;
}

const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? -80 : 80,
    opacity: 0,
    scale: 0.97,
  }),
  center: { x: 0, opacity: 1, scale: 1 },
  exit: (direction: number) => ({
    x: direction > 0 ? 80 : -80,
    opacity: 0,
    scale: 0.97,
  }),
};

const MODE_OPTIONS: Array<{
  mode: AnnotationMode;
  label: string;
  desc: string;
  icon: typeof Binary;
}> = perLocale(() => [
  {
    mode: "binary",
    label: msg("auto.features.tagger.components.taggersetup.literal.4"),
    desc: msg("auto.features.tagger.components.taggersetup.literal.5"),
    icon: Binary,
  },
  {
    mode: "multiclass",
    label: msg("auto.features.tagger.components.taggersetup.literal.6"),
    desc: msg("auto.features.tagger.components.taggersetup.literal.7"),
    icon: ListChecks,
  },
  {
    mode: "freetext",
    label: msg("auto.features.tagger.components.taggersetup.literal.8"),
    desc: msg("auto.features.tagger.components.taggersetup.literal.9"),
    icon: TextCursorInput,
  },
]);

export function TaggerSetup({ onStart }: TaggerSetupProps) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState(1);

  const [file, setFile] = useState<File | null>(null);
  const [parsedRows, setParsedRows] = useState<DataRow[]>([]);
  const [parsedCols, setParsedCols] = useState<string[]>([]);
  const [inputCols, setInputCols] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<AnnotationMode | null>(null);
  const [assistMode, setAssistMode] = useState<TaggerAssistMode>("copilot");
  // Empty = the server's default tagging model.
  const [assistModel, setAssistModel] = useState("");
  const [libraryName, setLibraryName] = useState<string | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryOptions, setLibraryOptions] = useState<DatasetSummary[]>([]);
  const [pickedId, setPickedId] = useState<string | null>(null);

  // Whether the user chose an approach themselves — an explicit pick must
  // never be overridden by the size-hint default below.
  const assistPicked = useRef(false);

  const { prefs } = useUserPrefs();
  const assistAvailable = isTaggerAssistEnabled() && prefs.taggerAssist;
  const effectiveAssistMode = assistAvailable ? assistMode : "manual";
  // Assisted flows leave the answer style to the interview, so only manual
  // flows get the task-definition step (interface + question/categories).
  const needsTaskStep = effectiveAssistMode === "manual";
  const activeSteps = [
    ...BASE_STEPS,
    ...(assistAvailable ? [ASSIST_STEP] : []),
    ...(needsTaskStep ? [TASK_STEP] : []),
  ];

  // The approach step precedes any interface choice, so the tiny-dataset
  // caveat sizes against the provisional calibration target.
  const tinyTarget = calibrationTarget({
    mode: "freetext",
    modeProvisional: true,
    inputColumns: inputCols,
  });
  const tinyDataset = parsedRows.length > 0 && parsedRows.length <= tinyTarget + REVIEW_BATCH_SIZE;

  // The pre-selected approach follows the size hint: while the hint says
  // "Manual or Autopilot will serve you better", Co-pilot must not stay
  // selected by default — the screen would contradict itself.
  useEffect(() => {
    if (assistPicked.current) return;
    setAssistMode(tinyDataset ? "manual" : "copilot");
  }, [tinyDataset]);

  const [question, setQuestion] = useState(
    msg("auto.features.tagger.components.taggersetup.literal.10"),
  );
  const [categories, setCategories] = useState<Category[]>([
    { id: "cat1", label: msg("auto.features.tagger.components.taggersetup.literal.11") },
    { id: "cat2", label: msg("auto.features.tagger.components.taggersetup.literal.12") },
  ]);

  // Tutorial hooks — let the guided tour inject demo data and navigate steps
  useEffect(
    () =>
      registerTutorialHook("setTaggerStep", (s: number) => {
        setDirection(s > step ? 1 : -1);
        setStep(s);
      }),
    [step],
  );
  useEffect(
    () =>
      registerTutorialHook("setTaggerDemoData", (data) => {
        setFile(new File([""], "demo_dataset.csv"));
        setParsedRows(data.rows as DataRow[]);
        setParsedCols(data.cols);
        setInputCols(Array.isArray(data.textCol) ? data.textCol : [data.textCol]);
      }),
    [],
  );
  useEffect(
    () => registerTutorialQuery("hasTaggerData", () => parsedRows.length > 0),
    [parsedRows],
  );

  const handleFile = useCallback(async (f: File) => {
    setError(null);
    setFile(f);
    setLibraryName(null);
    setPickedId(null);
    try {
      const { columns, rows } = await parseDatasetFile(f);
      setParsedRows(rows as DataRow[]);
      setParsedCols(columns);
      const guessText = columns.find((c) => c.toLowerCase() === "text") ?? columns[0];
      setInputCols(guessText ? [guessText] : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : msg("tagger.upload.parse_failed"));
    }
  }, []);

  // Deep link from the dataset editor ("Tag with AI"): /tagger?dataset={id}
  // loads the library dataset by reference; the saved column roles pre-select
  // the input columns (a tagger-saved dataset marks its own label/provenance
  // columns output/ignore, so they stay unselected).
  const loadLibraryDataset = useCallback(async (datasetId: string, name: string | null) => {
    setError(null);
    setLibraryLoading(true);
    try {
      const detail = await getDatasetRows(datasetId);
      setParsedRows(detail.rows as DataRow[]);
      setParsedCols(detail.columns);
      setFile(null);
      setPickedId(datasetId);
      setLibraryName(name || msg("tagger.setup.library_fallback_name"));
      const roles = detail.column_schema?.column_roles ?? {};
      const inputs = detail.columns.filter((c) => roles[c] === "input");
      if (inputs.length > 0) {
        setInputCols(inputs);
      } else {
        const guessText =
          detail.columns.find((c) => c.toLowerCase() === "text") ?? detail.columns[0];
        setInputCols(guessText ? [guessText] : []);
      }
    } catch {
      setError(msg("tagger.setup.library_error"));
    } finally {
      setLibraryLoading(false);
    }
  }, []);

  // window.location (not useSearchParams) keeps the page statically renderable.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const datasetId = params.get("dataset");
    if (datasetId) void loadLibraryDataset(datasetId, params.get("name"));
  }, [loadLibraryDataset]);

  useEffect(() => {
    let alive = true;
    listDatasets()
      .then((res) => {
        if (alive) setLibraryOptions(res.datasets);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) void handleFile(f);
    },
    [handleFile],
  );

  const addCategory = () => {
    setCategories((prev) => [...prev, { id: crypto.randomUUID(), label: "" }]);
  };

  const removeCategory = (id: string) => {
    setCategories((prev) => prev.filter((c) => c.id !== id));
  };

  const updateCategory = (id: string, label: string) => {
    setCategories((prev) => prev.map((c) => (c.id === id ? { ...c, label } : c)));
  };

  const toggleInputCol = (col: string) => {
    setInputCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  };

  // Which steps exist depends on the chosen approach (assisted flows drop the
  // task step), so gate on the step's id rather than its index.
  const validateStep = (s: number): boolean => {
    const id = activeSteps[s]?.id;
    if (id === "data") return parsedRows.length > 0 && inputCols.length > 0;
    if (id === "assist") return assistAvailable;
    if (id === "task") {
      if (!mode) return false;
      if (mode === "binary") return question.trim().length > 0;
      if (mode === "multiclass") return categories.filter((c) => c.label.trim()).length >= 2;
      return true;
    }
    return false;
  };

  // The substantive gate for starting: data selected, plus — for manual flows
  // only — a fully-defined task. Assisted flows define the task (including the
  // answer style) in the interview instead.
  const canStart = (): boolean =>
    parsedRows.length > 0 &&
    inputCols.length > 0 &&
    (effectiveAssistMode !== "manual" ||
      (!!mode &&
        (mode !== "binary" || question.trim().length > 0) &&
        (mode !== "multiclass" || categories.filter((c) => c.label.trim()).length >= 2)));

  const maxReachableStep = (() => {
    for (let i = 0; i < activeSteps.length; i++) {
      if (!validateStep(i)) return i;
    }
    return activeSteps.length - 1;
  })();

  const goTo = (idx: number) => {
    setDirection(idx > step ? 1 : -1);
    setStep(idx);
  };

  const handleNext = () => {
    if (step < activeSteps.length - 1 && validateStep(step)) {
      setDirection(1);
      setStep(step + 1);
    }
  };

  const goPrev = () => {
    if (step > 0) {
      setDirection(-1);
      setStep(step - 1);
    }
  };

  const handleTabClick = (idx: number) => {
    if (idx <= step || idx <= maxReachableStep) goTo(idx);
  };

  const handleStart = () => {
    if (!canStart()) return;
    const mapped: DataRow[] = parsedRows.map((row, i) => {
      const fields = inputCols.map((col) => ({ column: col, value: row[col] }));
      // ``text`` stays as a flat string for CSV export / search / single-col
      // fallbacks. The structured ``fields`` array is what the annotation UI
      // renders so JSON-shaped values (arrays, objects) get proper layout.
      const text = fields
        .map(({ column, value }) => {
          const flat =
            value === undefined || value === null
              ? ""
              : typeof value === "object"
                ? JSON.stringify(value)
                : String(value);
          return inputCols.length > 1 ? `${column}: ${flat}` : flat;
        })
        .join("\n");
      return { ...row, id: i + 1, text, fields };
    });
    // Assisted sessions start with a provisional answer style; the interview
    // infers the real one and stores it in the task override. Manual freetext
    // carries no prompt — annotation falls back to a default header.
    const config: TaggerConfig =
      effectiveAssistMode === "manual"
        ? { mode: mode!, inputColumns: inputCols }
        : { mode: "freetext", modeProvisional: true, inputColumns: inputCols };
    if (effectiveAssistMode === "manual" && mode === "binary") config.question = question.trim();
    if (effectiveAssistMode === "manual" && mode === "multiclass") {
      config.categories = categories.filter((c) => c.label.trim());
    }
    config.assistMode = effectiveAssistMode;
    const source = libraryName ?? (file ? cleanSourceName(file.name) : null);
    if (source) config.sourceName = source;
    onStart(
      config,
      mapped,
      parsedCols,
      effectiveAssistMode,
      effectiveAssistMode === "manual" ? undefined : assistModel.trim() || undefined,
    );
  };

  const steps = [
    <Card key="data">
      <CardHeader>
        <CardTitle className="text-base">
          <HelpTip text={tip("tagger.upload_file")}>
            {msg("auto.features.tagger.components.taggersetup.1")}
          </HelpTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <label
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 cursor-pointer transition-all duration-300 group",
            file ? "border-primary/40 bg-primary/5" : "hover:border-primary/50 hover:bg-muted/30",
          )}
        >
          <Upload className="size-8 text-muted-foreground group-hover:text-primary/70 transition-colors duration-300" />
          {file ? (
            <div className="text-center">
              <p className="font-medium text-foreground">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {parsedRows.length}
                {msg("auto.features.tagger.components.taggersetup.2")}
              </p>
            </div>
          ) : libraryName ? (
            <div className="text-center">
              <p className="font-medium text-foreground" dir="auto">
                {libraryName}
              </p>
              <p className="text-sm text-muted-foreground">
                {formatMsg("datasets.count.rows", { count: parsedRows.length })}
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              {msg("auto.features.tagger.components.taggersetup.3")}
            </p>
          )}
          <input
            type="file"
            accept=".json,.csv,.xlsx,.xls"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFile(f);
            }}
          />
        </label>
        {libraryLoading && (
          <p className="text-sm text-muted-foreground">{msg("tagger.setup.library_loading")}</p>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}

        {libraryOptions.length > 0 && (
          <>
            <div className="flex items-center gap-3">
              <Separator className="flex-1" />
              <span className="text-xs text-muted-foreground">
                {msg("tagger.setup.library_or")}
              </span>
              <Separator className="flex-1" />
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">{msg("tagger.setup.library_pick")}</p>
              <div className="max-h-44 space-y-1 overflow-y-auto">
                {libraryOptions.map((ds) => {
                  const selected = pickedId === ds.id;
                  return (
                    <button
                      key={ds.id}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => void loadLibraryDataset(ds.id, ds.name)}
                      className={cn(
                        "flex w-full min-w-0 items-center gap-2 rounded-lg px-3 py-2 text-sm transition-all cursor-pointer",
                        selected
                          ? "bg-primary/10 border border-primary/40 text-primary font-medium"
                          : "border border-transparent text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                      )}
                    >
                      <Database className="size-3.5 shrink-0" />
                      <span className="truncate min-w-0" dir="auto">
                        {ds.name}
                      </span>
                      <span className="ms-auto shrink-0 text-xs text-muted-foreground">
                        {formatMsg("datasets.count.rows", { count: ds.row_count })}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {parsedCols.length > 0 && (
          <>
            <Separator />
            <div className="space-y-3">
              <p className="text-sm font-medium">
                <HelpTip text={tip("tagger.text_column")}>
                  {msg("auto.features.tagger.components.taggersetup.4")}
                </HelpTip>
              </p>
              <div className="space-y-1">
                {parsedCols.map((col, i) => {
                  const selected = inputCols.includes(col);
                  return (
                    <button
                      key={`${col}-${i}`}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => toggleInputCol(col)}
                      className={cn(
                        "flex w-full min-w-0 items-center gap-2 rounded-lg px-3 py-2 text-sm transition-all cursor-pointer",
                        selected
                          ? "bg-primary/10 border border-primary/40 text-primary font-medium"
                          : "border border-transparent text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                      )}
                    >
                      <span
                        className="size-3 rounded-[3px] border-2 flex items-center justify-center shrink-0"
                        style={{ borderColor: selected ? "var(--primary)" : "var(--border)" }}
                      >
                        {selected && <Check className="size-2 text-primary" strokeWidth={3.5} />}
                      </span>
                      <span className="font-mono text-xs truncate min-w-0" dir="ltr">
                        {col}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>,
  ];

  // The manual task step: pick the answer style, then define it inline —
  // one decision surface instead of the old separate mode + settings steps.
  const taskCard = (
    <Card key="task" data-tutorial={assistAvailable ? undefined : "tagger-modes"}>
      <CardHeader>
        <CardTitle className="text-base">
          <HelpTip text={tip("tagger.mode")}>
            {msg("auto.features.tagger.components.taggersetup.5")}
          </HelpTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {MODE_OPTIONS.map((opt) => (
            <button
              key={opt.mode}
              type="button"
              aria-pressed={mode === opt.mode}
              onClick={() => setMode(opt.mode)}
              className={cn(
                "flex min-w-0 flex-col items-center gap-2 rounded-xl border p-4 text-center transition-all cursor-pointer",
                "hover:border-primary/40 hover:bg-primary/5",
                mode === opt.mode ? "border-primary bg-primary/10 shadow-sm" : "border-border/50",
              )}
            >
              <opt.icon
                className={cn(
                  "size-6",
                  mode === opt.mode ? "text-primary" : "text-muted-foreground",
                )}
              />
              <span
                className={cn(
                  "text-sm font-medium",
                  mode === opt.mode ? "text-primary" : "text-foreground",
                )}
              >
                {opt.label}
              </span>
              <span className="text-xs text-muted-foreground">{opt.desc}</span>
            </button>
          ))}
        </div>
        <AnimatePresence mode="wait" initial={false}>
          {mode === "binary" && (
            <motion.div
              key="binary"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-2"
            >
              <p className="text-sm font-medium">
                <HelpTip text={tip("tagger.binary_question")}>
                  {msg("auto.features.tagger.components.taggersetup.6")}
                </HelpTip>
              </p>
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder={msg("auto.features.tagger.components.taggersetup.literal.15")}
                dir="auto"
              />
            </motion.div>
          )}
          {mode === "multiclass" && (
            <motion.div
              key="multiclass"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-2"
            >
              <p className="text-sm font-medium">
                <HelpTip text={tip("tagger.multiclass_categories")}>
                  {msg("auto.features.tagger.components.taggersetup.7")}
                </HelpTip>
              </p>
              <p className="text-xs text-muted-foreground">
                {msg("auto.features.tagger.components.taggersetup.9")}
              </p>
              {categories.map((cat) => (
                <div key={cat.id} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={cat.label}
                    onChange={(e) => updateCategory(cat.id, e.target.value)}
                    className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                    placeholder={msg("auto.features.tagger.components.taggersetup.literal.16")}
                    dir="auto"
                  />
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={() => removeCategory(cat.id)}
                    disabled={categories.length <= 2}
                    aria-label={msg("auto.features.tagger.components.taggersetup.16")}
                  >
                    <Trash2 className="size-3.5 text-muted-foreground" />
                  </Button>
                </div>
              ))}
              <Button
                variant="outline"
                size="sm"
                onClick={addCategory}
                className="mt-1 w-full"
                title={msg("auto.features.tagger.components.taggersetup.literal.17")}
                aria-label={msg("auto.features.tagger.components.taggersetup.literal.17")}
              >
                <Plus className="size-3.5" />
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
  if (assistAvailable) {
    steps.push(
      <Card key="assist" data-tutorial="tagger-modes">
        <CardHeader>
          <CardTitle className="text-base">{msg("tagger.assist.setup.title")}</CardTitle>
          <CardDescription>{msg("tagger.assist.setup.description")}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {ASSIST_OPTIONS.map((opt) => {
            const selected = assistMode === opt.mode;
            const discouraged = tinyDataset && opt.mode === "copilot";
            return (
              <button
                key={opt.mode}
                type="button"
                aria-pressed={selected}
                onClick={() => {
                  assistPicked.current = true;
                  setAssistMode(opt.mode);
                }}
                className={cn(
                  "flex min-w-0 flex-col gap-0.5 rounded-xl border p-3.5 text-start transition-all cursor-pointer",
                  "hover:border-primary/40 hover:bg-primary/5",
                  selected ? "border-primary bg-primary/10 shadow-sm" : "border-border/50",
                )}
              >
                <span className="flex items-center gap-2">
                  <span
                    className={cn(
                      "text-sm font-medium",
                      selected ? "text-primary" : "text-foreground",
                    )}
                  >
                    {opt.label}
                  </span>
                  {opt.recommended && !discouraged && (
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                      {msg("tagger.assist.setup.recommended")}
                    </span>
                  )}
                </span>
                <span className="text-xs text-muted-foreground">{opt.desc}</span>
              </button>
            );
          })}
          {tinyDataset && (
            <p className="mt-1 text-xs text-muted-foreground">
              {msg("tagger.assist.setup.tiny_dataset")}
            </p>
          )}
          {assistMode !== "manual" && (
            <div className="mt-2 space-y-1.5">
              <p className="text-sm font-medium">{msg("tagger.assist.model.title")}</p>
              <p className="text-xs text-muted-foreground">{msg("tagger.assist.model.hint")}</p>
              <ModelPicker
                value={assistModel}
                onChange={setAssistModel}
                placeholder={msg("tagger.assist.model.placeholder")}
              />
            </div>
          )}
        </CardContent>
      </Card>,
    );
  }

  if (needsTaskStep) steps.push(taskCard);

  const isLastStep = step === activeSteps.length - 1;

  const rtl = getActiveDir() === "rtl";
  const BackIcon = rtl ? ChevronRight : ChevronLeft;
  const NextIcon = rtl ? ChevronLeft : ChevronRight;

  return (
    <div className="space-y-6 max-w-2xl mx-auto pb-8 -mt-2 md:-mt-4" data-tutorial="tagger-setup">
      <div className="relative">
        <div className="flex items-center justify-between">
          {activeSteps.map((s, i) => {
            const reachable = i <= maxReachableStep;
            const completed = i < step && validateStep(i);
            const active = i === step;
            return (
              <div key={s.id} className="flex flex-col items-center relative z-10 flex-1">
                <button
                  type="button"
                  onClick={() => handleTabClick(i)}
                  disabled={!reachable && i > step}
                  className={cn(
                    "relative flex items-center justify-center rounded-full transition-all duration-300 cursor-pointer",
                    "size-9 sm:size-10 text-sm font-semibold",
                    active
                      ? "bg-primary text-primary-foreground shadow-[0_0_16px_rgba(124,99,80,0.4)] scale-110"
                      : completed
                        ? "bg-primary/15 text-primary hover:bg-primary/25"
                        : reachable
                          ? "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                          : "bg-muted/50 text-muted-foreground/30 cursor-not-allowed",
                  )}
                >
                  {completed ? <Check className="size-4" /> : i + 1}
                  {active && (
                    <motion.span
                      layoutId="tagger-step-ring"
                      className="absolute inset-0 rounded-full border-2 border-primary"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                </button>
                <span
                  className={cn(
                    "mt-2 text-[0.6875rem] font-medium transition-colors duration-200 hidden sm:block text-center",
                    active
                      ? "text-foreground"
                      : completed
                        ? "text-primary"
                        : "text-muted-foreground",
                  )}
                >
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>
        <div className="absolute top-[18px] sm:top-5 inset-x-[10%] h-[2px] bg-muted -z-0 rounded-full">
          <motion.div
            className="h-full rounded-full"
            style={{ background: "var(--gradient-progress)" }}
            initial={{ width: 0 }}
            animate={{ width: `${(step / (activeSteps.length - 1)) * 100}%` }}
            transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
          />
        </div>
      </div>

      {/* x-clip (not hidden): the step slide animates horizontally only, and
          the model picker's dropdown must escape the wrapper vertically. */}
      <div className="relative overflow-x-clip pt-[10px]">
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={step}
            custom={direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.1 }}
          >
            {steps[step]}
          </motion.div>
        </AnimatePresence>
      </div>

      {!isLastStep ? (
        <div className="flex items-center justify-between">
          <Button variant="outline" onClick={goPrev} disabled={step === 0} className="gap-2">
            <BackIcon className="h-4 w-4" />
            {msg("auto.features.tagger.components.taggersetup.13")}
          </Button>
          <span className="text-xs text-muted-foreground tabular-nums">
            {step + 1} / {activeSteps.length}
          </span>
          <Button onClick={handleNext} disabled={!validateStep(step)} className="gap-2">
            {msg("auto.features.tagger.components.taggersetup.14")}
            <NextIcon className="h-4 w-4" />
          </Button>
        </div>
      ) : (
        <Button onClick={handleStart} disabled={!canStart()} size="lg" className="w-full">
          {msg("auto.features.tagger.components.taggersetup.15")}
        </Button>
      )}
    </div>
  );
}
