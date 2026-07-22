"use client";

import * as React from "react";
import { Square } from "lucide-react";
import { msg } from "@/shared/lib/messages";

import { Button } from "@/shared/ui/primitives/button";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { cn } from "@/shared/lib/utils";

import { autoResizeTextarea } from "./auto-resize";

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
  /** Optional model selector rendered in a footer row under the textarea
   *  (the ChatGPT-style composer pill — see ``ComposerModelMenu``). */
  modelMenu?: React.ReactNode;
}

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
}: ComposerProps) {
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (disabled || streaming || !value.trim()) return;
    onSubmit();
    if (textareaRef.current) textareaRef.current.style.height = "42px";
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-border/40 px-3 py-3 shrink-0">
      <div className="flex items-start gap-2">
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
            "block flex-1 bg-muted/20 rounded-2xl border border-[#DDD4C8] px-4 py-[11px] text-sm leading-[20px] resize-none overflow-hidden",
            "h-[42px] max-h-[120px] outline-none ring-0 shadow-none",
            "focus:outline-none focus-visible:outline-none focus-visible:ring-0 focus:border-[#C8A882] transition-colors",
            "placeholder:text-muted-foreground/40",
            "disabled:opacity-50 disabled:cursor-not-allowed",
          )}
        />
        {streaming && onStop ? (
          <TooltipButton tooltip={stopAriaLabel} side="top">
            <Button
              type="button"
              size="icon"
              onClick={onStop}
              className="shrink-0 rounded-full !size-[42px]"
              aria-label={stopAriaLabel}
            >
              <Square className="size-3 fill-current" />
            </Button>
          </TooltipButton>
        ) : (
          <Button
            type="submit"
            size="icon"
            className="shrink-0 rounded-full !size-[42px]"
            disabled={disabled || !value.trim()}
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
      {modelMenu && <div className="mt-1.5 flex items-center">{modelMenu}</div>}
    </form>
  );
}
