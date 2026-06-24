"use client";

import * as React from "react";
import { Check, Clipboard, Cpu, RefreshCw } from "lucide-react";

import { Badge } from "@/shared/ui/primitives/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { getActiveDir } from "@/shared/lib/runtime-locale";

interface MessageActionsProps {
  text: string;
  model?: string | null;
  onRegenerate?: () => void;
  className?: string;
}

interface ActionButtonProps {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}

function ActionButton({ label, onClick, children }: ActionButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={label}
          className={cn(
            "inline-flex items-center justify-center rounded-md p-1.5 cursor-pointer outline-none",
            "text-foreground/40 hover:text-foreground hover:bg-accent/70",
            "transition-colors",
            "focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
          )}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" dir={getActiveDir()}>
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

export function MessageActions({ text, model, onRegenerate, className }: MessageActionsProps) {
  const [copied, setCopied] = React.useState(false);
  const handleCopy = React.useCallback(() => {
    if (!text) return;
    void navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }, [text]);

  const shortModel = model ? (model.split("/").pop() ?? model) : null;

  return (
    <div className={cn("flex items-center gap-1 -ms-1.5", className)}>
      {text.length > 0 && (
        <ActionButton
          label={msg(copied ? "shared.agent.copied" : "shared.agent.copy")}
          onClick={handleCopy}
        >
          {copied ? (
            <Check className="size-3.5 text-foreground" />
          ) : (
            <Clipboard className="size-3.5" />
          )}
        </ActionButton>
      )}
      {onRegenerate && (
        <ActionButton label={msg("shared.agent.regenerate")} onClick={onRegenerate}>
          <RefreshCw className="size-3.5" />
        </ActionButton>
      )}
      <span className="sr-only" role="status" aria-live="polite">
        {copied ? msg("shared.agent.copied") : ""}
      </span>
      {model && shortModel && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge
              variant="ghost"
              size="sm"
              dir="ltr"
              className={cn(
                "ms-1.5 h-[26px] rounded-md px-2 font-mono cursor-default",
                "shadow-none text-muted-foreground/80",
              )}
            >
              <Cpu aria-hidden="true" />
              <span className="truncate max-w-[140px]">{shortModel}</span>
            </Badge>
          </TooltipTrigger>
          <TooltipContent side="top" dir="ltr" className="font-mono">
            {model}
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}
