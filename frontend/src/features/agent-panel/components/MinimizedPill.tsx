"use client";

import * as React from "react";
import { Sparkles } from "lucide-react";
import { msg } from "@/shared/lib/messages";
import { formatShortcut, useUserPrefs } from "@/features/settings";

import { cn } from "@/shared/lib/utils";

interface MinimizedPillProps {
  onOpen: () => void;
  active: boolean;
  statusLabel?: string;
  hue?: string;
  className?: string;
}

export function MinimizedPill({
  onOpen,
  active,
  statusLabel,
  hue = "#3D2E22",
  className,
}: MinimizedPillProps) {
  const { prefs } = useUserPrefs();
  const shortcutLabel = formatShortcut(prefs.agentShortcut);
  const ariaLabel = `${msg("auto.features.agent.panel.components.minimizedpill.literal.1")} (${shortcutLabel})`;
  const showLabel = active && Boolean(statusLabel);

  return (
    <button
      type="button"
      onClick={onOpen}
      data-tutorial="agent-pill"
      aria-label={ariaLabel}
      title={ariaLabel}
      className={cn(
        "fixed bottom-4 end-4 z-40 inline-flex items-center gap-2 rounded-full",
        "border border-border/60 bg-background/90 backdrop-blur-md",
        "px-3.5 py-2 text-[0.75rem] text-foreground shadow-[0_6px_18px_rgba(61,46,34,0.08)]",
        "transition-all duration-200 hover:bg-background hover:shadow-[0_10px_24px_rgba(61,46,34,0.12)]",
        "active:scale-[0.98] cursor-pointer",
        className,
      )}
    >
      <span className="relative inline-flex size-5 items-center justify-center">
        {active && (
          <span
            className="absolute inset-0 rounded-full animate-ping motion-reduce:animate-none"
            style={{ backgroundColor: `${hue}33` }}
          />
        )}
        <span
          className="relative inline-flex size-5 items-center justify-center rounded-full text-[#FAF8F5]"
          style={{ backgroundColor: hue }}
        >
          <Sparkles className="size-3" aria-hidden="true" />
        </span>
      </span>
      <span className="truncate max-w-[18ch] leading-none">
        {showLabel
          ? statusLabel
          : msg("auto.features.agent.panel.components.minimizedpill.literal.3")}
      </span>
      <span className="text-muted-foreground/70 font-mono text-[0.625rem] tracking-tight">
        {shortcutLabel}
      </span>
    </button>
  );
}
