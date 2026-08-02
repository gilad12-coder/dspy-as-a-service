"use client";

import * as React from "react";
import { Check, Clipboard, PencilSimple } from "@/shared/ui/icons";

import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import { TooltipButton } from "@/shared/ui/tooltip-button";

import { autoResizeTextarea } from "./auto-resize";

interface UserBubbleProps {
  content: string;
  onEdit?: () => void;
  editable?: boolean;
}

export function UserBubble({ content, onEdit, editable = true }: UserBubbleProps) {
  const [copied, setCopied] = React.useState(false);
  const handleCopy = React.useCallback(() => {
    if (!content) return;
    void navigator.clipboard.writeText(content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }, [content]);

  return (
    <div className="flex justify-start group/user">
      <div
        className="max-w-[80%] rounded-[22px] rounded-es-[4px] bg-[#3D2E22] text-[#FAF8F5] px-4 py-2.5 text-sm leading-[1.45] shadow-sm"
        dir="auto"
      >
        {content}
      </div>
      {/* Hover-revealed actions beside the bubble: copy (always) + edit (when
          editable), mirroring the assistant reply's copy affordance. */}
      <div
        className={cn(
          "self-center ms-1.5 flex items-center gap-0.5",
          "opacity-0 group-hover/user:opacity-100 transition-opacity",
        )}
      >
        <TooltipButton
          tooltip={msg(copied ? "shared.agent.copied" : "shared.agent.copy")}
          side="top"
        >
          <button
            type="button"
            onClick={handleCopy}
            className="p-1.5 rounded-lg hover:bg-muted/60 cursor-pointer"
            aria-label={msg(copied ? "shared.agent.copied" : "shared.agent.copy")}
          >
            {copied ? (
              <Check className="size-3 text-muted-foreground" />
            ) : (
              <Clipboard className="size-3 text-muted-foreground" />
            )}
          </button>
        </TooltipButton>
        {editable && onEdit && (
          <TooltipButton tooltip={msg("shared.agent.edit_and_resend")} side="top">
            <button
              type="button"
              onClick={onEdit}
              className="p-1.5 rounded-lg hover:bg-muted/60 cursor-pointer"
              aria-label={msg("shared.agent.edit_and_resend")}
            >
              <PencilSimple className="size-3 text-muted-foreground" />
            </button>
          </TooltipButton>
        )}
        <span className="sr-only" role="status" aria-live="polite">
          {copied ? msg("shared.agent.copied") : ""}
        </span>
      </div>
    </div>
  );
}

interface UserBubbleEditorProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  disabled?: boolean;
}

export function UserBubbleEditor({
  value,
  onChange,
  onSubmit,
  onCancel,
  disabled,
}: UserBubbleEditorProps) {
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);

  React.useEffect(() => {
    autoResizeTextarea(textareaRef.current);
  }, [value]);

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] w-full space-y-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            autoResizeTextarea(e.target);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              onCancel();
            }
          }}
          className="w-full bg-white border border-[#DDD4C8] rounded-xl px-3 py-2 text-sm resize-none outline-none focus:border-[#C8A882] transition-colors min-h-[40px] max-h-[120px]"
          rows={1}
          autoFocus
          dir="auto"
        />
        <div className="flex justify-start gap-1.5">
          <button
            type="button"
            onClick={onCancel}
            className="text-[0.75rem] font-medium text-foreground/80 hover:text-foreground bg-white border border-[#DDD4C8] hover:bg-muted px-3.5 py-1.5 rounded-full transition-all duration-150 cursor-pointer active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/25 focus-visible:ring-offset-1"
          >
            {msg("shared.agent.cancel")}
          </button>
          <button
            type="button"
            onClick={onSubmit}
            disabled={!value.trim() || disabled}
            className="text-[0.75rem] font-medium text-white bg-[#3D2E22] hover:bg-[#3D2E22]/90 px-4 py-1.5 rounded-full transition-all duration-150 cursor-pointer active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40 focus-visible:ring-offset-1 disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100"
          >
            {msg("shared.agent.send")}
          </button>
        </div>
      </div>
    </div>
  );
}
