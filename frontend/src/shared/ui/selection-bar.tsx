"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Trash, X } from "@/shared/ui/icons";

import { formatMsg, msg } from "@/shared/lib/messages";
import { TooltipButton } from "@/shared/ui/tooltip-button";

interface SelectionBarProps {
  count: number;
  onClear: () => void;
  onDelete: () => void;
}

/**
 * Floating bulk-action pill for multi-select card lists (labeling sessions,
 * datasets) — the dashboard jobs table's bottom-docked bar reduced to
 * count + clear + delete. Renders nothing until something is selected.
 */
export function SelectionBar({ count, onClear, onDelete }: SelectionBarProps) {
  return (
    <AnimatePresence>
      {count > 0 && (
        <motion.div
          initial={{ y: 24, opacity: 0, scale: 0.96 }}
          animate={{ y: 0, opacity: 1, scale: 1 }}
          exit={{ y: 24, opacity: 0, scale: 0.96 }}
          transition={{ type: "spring", stiffness: 380, damping: 30, mass: 0.8 }}
          className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50"
        >
          <div className="flex max-w-[92vw] items-center gap-1 rounded-full border border-border/60 bg-background/95 px-3 py-1.5 shadow-[0_12px_32px_rgba(0,0,0,0.18)] backdrop-blur-xl">
            <span className="min-w-0 px-1 text-sm tabular-nums text-foreground">
              {formatMsg("shared.selection.count", { count })}
            </span>
            <div className="mx-1 h-5 w-px bg-border/60" />
            <TooltipButton tooltip={msg("shared.selection.clear")} side="top" delayDuration={150}>
              <button
                type="button"
                onClick={onClear}
                className="close-button"
                style={
                  {
                    "--close-btn-size": "32px",
                    "--close-btn-radius": "9999px",
                    "--close-btn-icon": "16px",
                  } as React.CSSProperties
                }
                aria-label={msg("shared.selection.clear")}
              >
                <X />
              </button>
            </TooltipButton>
            <TooltipButton tooltip={msg("shared.selection.delete")} side="top" delayDuration={150}>
              <button
                type="button"
                onClick={onDelete}
                className="flex size-8 items-center justify-center rounded-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive active:scale-95 transition-all cursor-pointer"
                aria-label={msg("shared.selection.delete")}
              >
                <Trash className="size-4" />
              </button>
            </TooltipButton>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
