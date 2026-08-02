"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import {
  EditorView,
  ViewPlugin,
  Decoration,
  type DecorationSet,
  type ViewUpdate,
} from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import { type Range } from "@codemirror/state";
import { tags } from "@lezer/highlight";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { highlightSelectionMatches } from "@codemirror/search";
import type { CompletionContext } from "@codemirror/autocomplete";
import { autocompletion, acceptCompletion, type CompletionResult } from "@codemirror/autocomplete";
import { indentWithTab } from "@codemirror/commands";
import { Prec } from "@codemirror/state";
import { keymap } from "@codemirror/view";
import { linkifyMessage } from "@/shared/lib/linkify";
import { CopyGlyph, useCopyToClipboard } from "@/shared/ui/copy-button";
import {
  CaretUp,
  CaretDown,
  Play,
  CircleNotch,
  X,
  Eraser,
} from "@/shared/ui/icons";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { useLiteMode } from "@/features/settings";

const beigeEditorTheme = EditorView.theme(
  {
    "&": {
      backgroundColor: "#FAF6F0",
      color: "#3D2E22",
      fontSize: "12.5px",
    },
    ".cm-content": {
      caretColor: "#7C6350",
      fontFamily: "ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, monospace",
      fontSize: "12.5px",
    },
    ".cm-cursor, .cm-dropCursor": {
      borderLeftColor: "#7C6350",
    },
    /* Raise the selection layer so the highlight shows over the beige
       background (its default negative z-index hides behind it). Lifting it
       above .cm-content means it must stay click-through: the layer carries
       no pointer-events:none of its own, so once on top a select-all would
       blanket the editor and swallow every click — leaving it unclickable and
       "stuck". pointer-events:none restores normal caret/click behaviour, the
       same guard CM applies to its own .cm-cursorLayer. */
    ".cm-selectionLayer": {
      zIndex: "1 !important",
      pointerEvents: "none",
    },
    /* Selection highlight — must be visually stronger than matches */
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
      backgroundColor: "rgba(140, 105, 50, 0.40) !important",
    },
    /* Other occurrences of the selected word */
    ".cm-selectionMatch": {
      backgroundColor: "rgba(160, 120, 70, 0.15) !important",
      borderRadius: "2px",
      outline: "1px solid rgba(160, 120, 70, 0.35)",
    },
    ".cm-focused .cm-cursor": {
      borderLeftColor: "#3D2E22",
      borderLeftWidth: "2px",
    },
    ".cm-activeLine": {
      backgroundColor: "rgba(245, 237, 228, 0.3)",
    },
    ".cm-gutters": {
      backgroundColor: "#F3ECE3",
      color: "#B09878",
      border: "none",
      borderRight: "1px solid #E5DDD4",
      fontSize: "12px",
    },
    ".cm-activeLineGutter": {
      backgroundColor: "#EDE4D8",
      color: "#7C6350",
    },
    ".cm-foldPlaceholder": {
      backgroundColor: "#E8DDD4",
      border: "none",
      color: "#7C6350",
    },
    ".cm-matchingBracket": {
      backgroundColor: "#E8DDD4",
      outline: "1px solid #C8B8A4",
    },
    /* ── Autocomplete tooltip ── */
    ".cm-tooltip": {
      backgroundColor: "#FAF6F0",
      border: "1px solid #E5DDD4",
      borderRadius: "8px",
      boxShadow: "0 4px 12px rgba(60, 46, 34, 0.1)",
    },
    ".cm-tooltip.cm-tooltip-autocomplete": {
      backgroundColor: "#FAF6F0",
    },
    ".cm-tooltip-autocomplete > ul": {
      fontFamily: "ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, monospace",
      fontSize: "12px",
    },
    ".cm-tooltip-autocomplete > ul > li": {
      padding: "3px 8px",
      color: "#3D2E22",
    },
    ".cm-tooltip-autocomplete > ul > li[aria-selected]": {
      backgroundColor: "#E8DDD4",
      color: "#3D2E22",
    },
    ".cm-completionIcon": {
      color: "#8C7A6B",
    },
    ".cm-completionLabel": {
      color: "#3D2E22",
    },
    ".cm-completionMatchedText": {
      color: "#8B5E3C",
      fontWeight: "bold",
      textDecoration: "none",
    },
  },
  { dark: false },
);

const beigeHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: "#8B5E3C", fontWeight: "bold" },
  { tag: tags.operator, color: "#7C6350" },
  { tag: tags.special(tags.variableName), color: "#6B4226" },
  { tag: tags.typeName, color: "#8B6914" },
  { tag: tags.atom, color: "#8B6914" },
  { tag: tags.number, color: "#986832" },
  { tag: tags.definition(tags.variableName), color: "#3D2E22" },
  { tag: tags.string, color: "#5A7247" },
  { tag: tags.special(tags.string), color: "#5A7247" },
  { tag: tags.comment, color: "#B09878", fontStyle: "italic" },
  { tag: tags.variableName, color: "#3D2E22" },
  { tag: tags.bracket, color: "#8C7A6B" },
  { tag: tags.tagName, color: "#8B5E3C" },
  { tag: tags.attributeName, color: "#8B6914" },
  { tag: tags.propertyName, color: "#6B4226" },
  { tag: tags.className, color: "#6B4226" },
  { tag: tags.function(tags.variableName), color: "#7C5030" },
  { tag: tags.bool, color: "#8B6914" },
  { tag: tags.null, color: "#8B6914" },
  { tag: tags.self, color: "#8B5E3C", fontStyle: "italic" },
  { tag: tags.punctuation, color: "#8C7A6B" },
]);

const beigeTheme: Extension = [beigeEditorTheme, syntaxHighlighting(beigeHighlightStyle)];

const dspyCompletions = [
  { label: "dspy.Signature", type: "class", detail: "Base signature class" },
  { label: "dspy.InputField", type: "function", detail: "Declare an input field" },
  { label: "dspy.OutputField", type: "function", detail: "Declare an output field" },
  { label: "dspy.Example", type: "class", detail: "DSPy example" },
  { label: "dspy.Prediction", type: "class", detail: "DSPy prediction" },
  { label: "dspy.Module", type: "class", detail: "Base module class" },
  { label: "dspy.ChainOfThought", type: "class", detail: "Chain of thought module" },
  { label: "dspy.Predict", type: "class", detail: "Predict module" },
  { label: "import dspy", type: "keyword", detail: "Import DSPy" },
  {
    label: "def metric(example, pred, trace=None):",
    type: "function",
    detail: "Metric function template",
  },
  { label: "with_inputs", type: "method", detail: "Set input fields on Example" },
  { label: "with_outputs", type: "method", detail: "Set output fields on Example" },
];

function dspyCompletion(context: CompletionContext): CompletionResult | null {
  const word = context.matchBefore(/[\w.]+/);
  if (!word || (word.from === word.to && !context.explicit)) return null;
  return {
    from: word.from,
    options: dspyCompletions,
    validFor: /^[\w.]*$/,
  };
}

const pyExtensions = [
  python(),
  highlightSelectionMatches(),
  Prec.highest(keymap.of([{ key: "Tab", run: acceptCompletion }])),
  keymap.of([indentWithTab]),
  autocompletion({
    override: [dspyCompletion],
    activateOnTyping: true,
    maxRenderedOptions: 15,
  }),
];

/* ── Streaming line-fade decoration ─────────────────────────────────────
   Applies a fade-in class to lines that appear during a streaming session.
   The class's CSS animation is only active when prefers-reduced-motion
   is "no-preference" (see .cm-line-fresh style below). */
const lineFadeDecoration = Decoration.line({ class: "cm-line-fresh" });

function streamingLineFadeExtension(): Extension {
  return ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;
      prev: number;
      constructor(view: EditorView) {
        this.prev = view.state.doc.lines;
        this.decorations = Decoration.none;
      }
      update(u: ViewUpdate) {
        if (!u.docChanged) return;
        const now = u.state.doc.lines;
        if (now > this.prev) {
          const ranges: Array<Range<Decoration>> = [];
          for (let i = this.prev + 1; i <= now; i++) {
            ranges.push(lineFadeDecoration.range(u.state.doc.line(i).from));
          }
          this.decorations = Decoration.set(ranges);
          this.prev = now;
        } else if (now < this.prev) {
          this.decorations = Decoration.none;
          this.prev = now;
        }
      }
    },
    { decorations: (v) => v.decorations },
  );
}

/* ── Changed-line flash decoration ──────────────────────────────────────
   Highlights a static set of line numbers (1-based) with a brief yellow
   flash after a refinement. The caller is expected to clear the set after
   the CSS animation completes so decorations don't linger. */
const flashLineDecoration = Decoration.line({ class: "cm-line-flash" });

function flashLinesExtension(lines: readonly number[]): Extension {
  return ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;
      constructor(view: EditorView) {
        this.decorations = this.build(view);
      }
      update(u: ViewUpdate) {
        if (u.docChanged || u.viewportChanged) {
          this.decorations = this.build(u.view);
        }
      }
      build(view: EditorView): DecorationSet {
        const total = view.state.doc.lines;
        const ranges: Array<Range<Decoration>> = [];
        for (const lineNum of lines) {
          if (lineNum >= 1 && lineNum <= total) {
            ranges.push(flashLineDecoration.range(view.state.doc.line(lineNum).from));
          }
        }
        return Decoration.set(ranges);
      }
    },
    { decorations: (v) => v.decorations },
  );
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  signature_fields?: { inputs: string[]; outputs: string[] };
}

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  height?: string;
  readOnly?: boolean;
  onRun?: () => Promise<ValidationResult | null>;
  runLabel?: string;
  runningLabel?: string;
  label?: React.ReactNode;
  /** External validation result — synced into the internal state so callers
   *  (e.g. the Next button) can trigger the inline error panel without
   *  going through the editor's own Run button. */
  validationResult?: ValidationResult | null;
  /** When true, new lines fade in as they appear. */
  streaming?: boolean;
  /** 1-based line numbers to flash briefly (post-refinement highlight). */
  flashLines?: readonly number[];
}

export function CodeEditor({
  value,
  onChange,
  height = "200px",
  readOnly = false,
  onRun,
  runLabel = msg("shared.code_editor.run"),
  runningLabel = msg("shared.code_editor.running"),
  label = "Python",
  validationResult,
  streaming = false,
  flashLines,
}: CodeEditorProps) {
  const extensions = useMemo(() => {
    const base = streaming ? [...pyExtensions, streamingLineFadeExtension()] : pyExtensions;
    if (flashLines && flashLines.length > 0) {
      return [...base, flashLinesExtension(flashLines)];
    }
    return base;
  }, [streaming, flashLines]);
  const [collapsed, setCollapsed] = useState(false);
  const { copied, copy: copyToClipboard } = useCopyToClipboard();
  const { copied: consoleCopied, copy: copyConsoleToClipboard } = useCopyToClipboard();
  const [running, setRunning] = useState(false);
  const [formatting, setFormatting] = useState(false);
  const [result, setResult] = useState<ValidationResult | null>(null);
  const lite = useLiteMode();

  useEffect(() => {
    if (validationResult !== undefined) setResult(validationResult);
  }, [validationResult]);

  const lineCount = value.split("\n").length;

  const handleCopy = useCallback(async () => {
    await copyToClipboard(value);
  }, [copyToClipboard, value]);

  const handleCopyConsole = useCallback(async () => {
    if (!result) return;
    const lines: string[] = [];
    for (const err of result.errors) lines.push(`[error] ${err}`);
    for (const w of result.warnings) lines.push(`[warn]  ${w}`);
    if (lines.length === 0) return;
    await copyConsoleToClipboard(lines.join("\n"));
  }, [copyConsoleToClipboard, result]);

  const handleRun = useCallback(async () => {
    if (!onRun || running) return;
    setRunning(true);
    setResult(null);
    try {
      const res = await onRun();
      setResult(res);
    } catch {
      setResult({ valid: false, errors: [msg("shared.code_editor.validation_failed")], warnings: [] });
    } finally {
      setRunning(false);
    }
  }, [onRun, running]);

  const handleFormat = useCallback(async () => {
    if (formatting || !value.trim()) return;
    setFormatting(true);
    try {
      const API = getRuntimeEnv().apiUrl;
      const res = await fetch(`${API}/format-code`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: value }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.changed) onChange(data.code);
      }
    } catch (error) {
      console.warn("Failed to format code", error);
    } finally {
      setFormatting(false);
    }
  }, [value, formatting, onChange]);

  const handleChange = useCallback(
    (v: string) => {
      onChange(v);
      if (result) setResult(null);
    },
    [onChange, result],
  );

  const hasOutput = result !== null || running;
  const hasConsoleBody =
    running || (result !== null && (result.errors.length > 0 || result.warnings.length > 0));

  return (
    <div
      className="rounded-xl border border-border/60 overflow-visible flex flex-col shadow-sm w-full"
      style={readOnly ? undefined : { maxHeight: "60vh" }}
      dir="ltr"
    >
      <div className="flex items-center gap-1 px-3 py-1.5 bg-[#F3ECE3] text-[0.6875rem] text-[#8C7A6B] border-b border-[#E5DDD4] rounded-t-xl">
        <span className="flex-1 font-semibold text-[#7C6350] tracking-wide flex items-center gap-1.5">
          {label}
        </span>

        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-black/5 transition-colors cursor-pointer"
        >
          {collapsed ? <CaretDown className="size-3" /> : <CaretUp className="size-3" />}
          {collapsed
            ? formatMsg("shared.code_editor.lines_count", { count: lineCount })
            : msg("shared.code_editor.collapse")}
        </button>

        {onRun && (
          <button
            type="button"
            onClick={handleRun}
            disabled={running || !value.trim()}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-black/5 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {running ? <CircleNotch className="size-3 animate-spin" /> : <Play className="size-3" />}
            {running ? runningLabel : runLabel}
          </button>
        )}

        <button
          type="button"
          onClick={handleFormat}
          disabled={formatting || !value.trim()}
          className="flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-black/5 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {formatting ? <CircleNotch className="size-3 animate-spin" /> : <Eraser className="size-3" />}
          {msg("shared.code_editor.format")}
        </button>

        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-black/5 transition-colors cursor-pointer"
        >
          <CopyGlyph copied={copied} className="size-3" checkClassName="text-[#3D2E22]" />
          {copied ? msg("shared.code_editor.copied") : msg("shared.code_editor.copy")}
        </button>
      </div>

      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="editor"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden rounded-b-xl"
          >
            <div
              className="relative overflow-y-auto [&_.cm-editor]:!outline-none"
              style={readOnly ? undefined : { maxHeight: "calc(60vh - 4rem)" }}
            >
              {lite ? (
                readOnly ? (
                  <pre
                    dir="ltr"
                    className="m-0 overflow-auto bg-[#FAF6F0] px-4 py-3 font-mono text-[12.5px] leading-relaxed text-[#3D2E22] whitespace-pre"
                    style={{ minHeight: height }}
                  >
                    {value}
                  </pre>
                ) : (
                  <textarea
                    dir="ltr"
                    value={value}
                    onChange={(e) => handleChange(e.target.value)}
                    spellCheck={false}
                    className="block w-full resize-y bg-[#FAF6F0] px-4 py-3 font-mono text-[12.5px] leading-relaxed text-[#3D2E22] outline-none"
                    style={{ height }}
                  />
                )
              ) : (
                <CodeMirror
                  value={value}
                  height={height}
                  theme={beigeTheme}
                  extensions={extensions}
                  onChange={handleChange}
                  readOnly={readOnly}
                  basicSetup={{
                    lineNumbers: true,
                    foldGutter: true,
                    bracketMatching: true,
                    autocompletion: true,
                    completionKeymap: true,
                    searchKeymap: false,
                    highlightActiveLine: true,
                    highlightSelectionMatches: false,
                    indentOnInput: true,
                    tabSize: 4,
                  }}
                />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      {collapsed && (
        <div className="bg-[#FAF6F0] text-[#B09878] text-xs italic px-4 py-2">
          {formatMsg("shared.code_editor.lines_hidden", { count: lineCount })}
        </div>
      )}

      {hasOutput && (
        <div dir="ltr" className="bg-[#EFE5D5] border-t border-[#E5DDD4] text-left">
          {!running && result && (
            <div
              className={`flex items-center justify-between gap-2 px-3 py-1.5 ${
                hasConsoleBody ? "border-b border-[#E5DDD4]/60" : ""
              }`}
            >
              <div className="flex items-center gap-1.5 min-w-0">
                {result.errors.length > 0 && (
                  <span className="inline-flex items-center gap-1.5 text-[0.6875rem] font-medium px-2 py-0.5 rounded-full bg-[#E8D7C5] text-[#3D2E22] tabular-nums">
                    <span className="size-1.5 rounded-full bg-[#6B4226]" aria-hidden="true" />
                    {formatMsg("shared.code_editor.errors_count", { count: result.errors.length })}
                  </span>
                )}
                {result.warnings.length > 0 && (
                  <span className="inline-flex items-center gap-1.5 text-[0.6875rem] font-medium px-2 py-0.5 rounded-full bg-[#EBE0B8] text-[#5C4509] tabular-nums">
                    <span className="size-1.5 rounded-full bg-[#8B6914]" aria-hidden="true" />
                    {formatMsg("shared.code_editor.warnings_count", { count: result.warnings.length })}
                  </span>
                )}
                {result.valid && result.errors.length === 0 && result.warnings.length === 0 && (
                  <span className="inline-flex items-center gap-1.5 text-[0.6875rem] font-medium px-2 py-0.5 rounded-full bg-[#EDE4D8] text-[#3D2E22]">
                    <span className="size-1.5 rounded-full bg-[#7C6350]" aria-hidden="true" />
                    {msg("shared.code_editor.valid")}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-0.5 shrink-0">
                {(result.errors.length > 0 || result.warnings.length > 0) && (
                  <TooltipButton
                    tooltip={
                      consoleCopied
                        ? msg("shared.code_editor.console_copied")
                        : msg("shared.code_editor.copy_console")
                    }
                    side="top"
                  >
                    <button
                      type="button"
                      onClick={handleCopyConsole}
                      className="close-button"
                      aria-label={msg("shared.code_editor.copy_console")}
                    >
                      <CopyGlyph copied={consoleCopied} checkClassName="text-[#3D2E22]" />
                    </button>
                  </TooltipButton>
                )}
                <TooltipButton tooltip={msg("shared.code_editor.console_clear")} side="top">
                  <button
                    type="button"
                    onClick={() => setResult(null)}
                    className="close-button"
                    aria-label={msg("shared.code_editor.console_clear")}
                  >
                    <X />
                  </button>
                </TooltipButton>
              </div>
            </div>
          )}
          {hasConsoleBody && (
            <div className="py-1 max-h-48 overflow-y-auto">
              {running && (
                <div className="flex items-center gap-2 px-4 py-1.5 text-xs text-[#8C7A6B]">
                  <CircleNotch className="size-3.5 animate-spin" />
                  {msg("shared.code_editor.validating")}
                </div>
              )}
              {result?.errors.map((err, i) => (
                <div
                  key={`e${i}`}
                  className="flex items-stretch border-l-2 border-[#6B4226] bg-[#E8D7C5]/35 hover:bg-[#E8D7C5]/60 transition-colors"
                  role="alert"
                >
                  <div className="flex-1 min-w-0 pl-3 pr-4 py-1.5 text-xs font-mono leading-relaxed text-[#3D2E22] break-words">
                    {linkifyMessage(err, "underline decoration-[#6B4226]/40 underline-offset-2 hover:decoration-[#6B4226] transition-colors")}
                  </div>
                </div>
              ))}
              {result?.warnings.map((w, i) => (
                <div
                  key={`w${i}`}
                  className="flex items-stretch border-l-2 border-[#8B6914] bg-[#EBE0B8]/35 hover:bg-[#EBE0B8]/60 transition-colors"
                  role="status"
                >
                  <div className="flex-1 min-w-0 pl-3 pr-4 py-1.5 text-xs font-mono leading-relaxed text-[#5C4509] break-words">
                    {w}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
