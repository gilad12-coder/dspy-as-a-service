"use client";

import * as React from "react";
import { Check, CircleNotch, Microphone, Square, X } from "@/shared/ui/icons";
import { msg } from "@/shared/lib/messages";

import { useLocale } from "@/shared/providers";
import { useUserPrefs } from "@/features/settings";
import { Button } from "@/shared/ui/primitives/button";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { cn } from "@/shared/lib/utils";

import { autoResizeTextarea } from "./auto-resize";
import { formatRecSeconds, useDictation } from "./use-dictation";

interface ComposerProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop?: () => void;
  placeholder?: string;
  disabled?: boolean;
  streaming?: boolean;
  sendAriaLabel?: string;
  stopAriaLabel?: string;
  /** Optional model selector chip rendered in the control row beside the mic
   *  and send controls (the Codex-style composer — see ``ComposerModelMenu``). */
  modelMenu?: React.ReactNode;
  /** Optional controls docked at the inline-start of the control row (the
   *  Codex layout: attach + permissions on the left, model/mic/send on the
   *  right). */
  leadingControls?: React.ReactNode;
  /** Keep controls below the draft by default; model-free playgrounds can opt
   *  into a single row. */
  layout?: "stacked" | "inline";
  className?: string;
}

/**
 * The shared chat composer, laid out Codex-style: one bordered box holding a
 * borderless textarea with a control row docked under it — model chip, then
 * dictation mic, then the circular send/stop button at the inline end. While
 * dictating, the textarea swaps for a recording strip (pulsing dot, timer,
 * cancel) and the mic becomes the finish control; the transcript is appended
 * to the draft for review, never auto-sent.
 */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  placeholder,
  disabled,
  streaming,
  sendAriaLabel = msg("auto.shared.ui.agent.composer.literal.1"),
  stopAriaLabel = msg("auto.shared.ui.agent.composer.literal.2"),
  modelMenu,
  leadingControls,
  layout = "stacked",
  className,
}: ComposerProps) {
  const inline = layout === "inline";
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);
  const { locale } = useLocale();
  const { prefs } = useUserPrefs();

  const valueRef = React.useRef(value);
  React.useEffect(() => {
    valueRef.current = value;
  }, [value]);
  const dictation = useDictation({
    language: locale,
    onText: (text) => {
      const prev = valueRef.current;
      onChange(prev.trim() ? `${prev.replace(/\s+$/, "")} ${text}` : text);
      // The textarea remounts from the recording strip on this same commit —
      // focus and grow it once it's back in the tree.
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (el) {
          el.focus();
          autoResizeTextarea(el);
        }
      });
    },
  });
  const dictating = dictation.state.kind !== "idle";
  // A recording in flight keeps its controls even if the pref flips off in
  // the settings modal mid-take.
  const showMic = prefs.dictationEnabled || dictating;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (disabled || streaming || dictating || !value.trim()) return;
    onSubmit();
    if (textareaRef.current) textareaRef.current.style.height = inline ? "36px" : "42px";
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={cn("border-t border-border/40 px-3 py-3 shrink-0", className)}
    >
      <div
        className={cn(
          "rounded-2xl border border-[#DDD4C8] bg-muted/20 transition-colors",
          "focus-within:border-[#C8A882]",
          inline && "flex min-h-11 items-end gap-1 p-1",
        )}
      >
        {!dictating ? (
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              onChange(e.target.value);
              autoResizeTextarea(e.target);
            }}
            onKeyDown={handleKeyDown}
            disabled={disabled || streaming}
            rows={1}
            placeholder={placeholder}
            className={cn(
              "bg-transparent text-sm leading-[20px] resize-none overflow-hidden",
              "max-h-[120px] outline-none ring-0 border-0 shadow-none",
              "focus:outline-none focus-visible:outline-none focus-visible:ring-0",
              "placeholder:text-muted-foreground/40",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              inline ? "min-w-0 flex-1 h-9 px-3 py-2" : "block w-full h-[42px] px-4 py-[11px]",
            )}
          />
        ) : (
          <div
            className={cn(
              "flex items-center gap-2 text-sm",
              inline ? "min-w-0 flex-1 h-9 px-3" : "h-[42px] px-4",
            )}
            role="status"
            aria-live="polite"
          >
            {dictation.state.kind === "rec" && (
              <>
                <span className="size-2 shrink-0 animate-pulse rounded-full bg-red-500" />
                <span className="tabular-nums text-muted-foreground" dir="ltr">
                  {formatRecSeconds(dictation.seconds)}
                </span>
                <span className="truncate text-muted-foreground">
                  {msg("agent.composer.recording")}
                </span>
                <button
                  type="button"
                  onClick={dictation.cancel}
                  aria-label={msg("agent.composer.record_cancel")}
                  className={cn(
                    "ms-auto inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full",
                    "text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                    "focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
                  )}
                >
                  <X className="size-4" />
                </button>
              </>
            )}
            {dictation.state.kind === "busy" && (
              <>
                <CircleNotch className="size-4 shrink-0 animate-spin text-muted-foreground" />
                <span className="truncate text-muted-foreground">
                  {msg("agent.composer.transcribing")}
                </span>
              </>
            )}
            {dictation.state.kind === "err" && (
              <span className="truncate text-destructive">{dictation.state.message}</span>
            )}
          </div>
        )}

        <div className={cn("flex items-center gap-1.5", inline ? "shrink-0" : "px-2 pb-2 pt-0.5")}>
          {leadingControls && <div className="flex items-center gap-1">{leadingControls}</div>}
          <div className="ms-auto flex items-center gap-1.5">
            {modelMenu}
            {showMic &&
              (dictation.state.kind === "rec" ? (
                <TooltipButton tooltip={msg("agent.composer.record_finish")} side="top">
                  <Button
                    type="button"
                    size="icon"
                    onClick={dictation.finish}
                    className="shrink-0 rounded-full !size-9"
                    aria-label={msg("agent.composer.record_finish")}
                  >
                    <Check className="size-4" />
                  </Button>
                </TooltipButton>
              ) : (
                <TooltipButton tooltip={msg("agent.composer.record")} side="top">
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    onClick={() => void dictation.start()}
                    disabled={disabled || streaming || dictation.state.kind === "busy"}
                    className="shrink-0 rounded-full !size-9 text-muted-foreground hover:text-foreground"
                    aria-label={msg("agent.composer.record")}
                  >
                    <Microphone className="size-4" />
                  </Button>
                </TooltipButton>
              ))}
            {streaming && onStop ? (
              <TooltipButton tooltip={stopAriaLabel} side="top">
                <Button
                  type="button"
                  size="icon"
                  onClick={onStop}
                  className="shrink-0 rounded-full !size-9"
                  aria-label={stopAriaLabel}
                >
                  <Square className="size-3 fill-current" />
                </Button>
              </TooltipButton>
            ) : (
              <Button
                type="submit"
                size="icon"
                className="shrink-0 rounded-full !size-9"
                disabled={disabled || dictating || !value.trim()}
                aria-label={sendAriaLabel}
              >
                <svg viewBox="0 0 24 24" fill="none" className="size-4">
                  <path
                    d="M12 2L12 22M12 2L5 9M12 2L19 9"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </Button>
            )}
          </div>
        </div>
      </div>
    </form>
  );
}
