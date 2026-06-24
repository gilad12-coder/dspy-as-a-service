import { AnimatePresence, motion } from "framer-motion";
import * as React from "react";
import { ArrowLeftRight, Trash2, X } from "lucide-react";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

type BulkActionBarProps = {
  canDelete: boolean;
  selectedCount: number;
  compareEligibleCount: number;
  canCompare: boolean;
  onClear: () => void;
  onCompare: () => void;
  onRequestBulkDelete: () => void;
};

export function BulkActionBar({
  canDelete,
  selectedCount,
  compareEligibleCount,
  canCompare,
  onClear,
  onCompare,
  onRequestBulkDelete,
}: BulkActionBarProps) {
  const skipped = selectedCount - compareEligibleCount;
  return (
    <AnimatePresence>
      {selectedCount > 0 && (
        <motion.div
          initial={{ y: 24, opacity: 0, scale: 0.96 }}
          animate={{ y: 0, opacity: 1, scale: 1 }}
          exit={{ y: 24, opacity: 0, scale: 0.96 }}
          transition={{
            type: "spring",
            stiffness: 380,
            damping: 30,
            mass: 0.8,
          }}
          className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50"
          data-tutorial="bulk-action-bar"
        >
          <div className="flex max-w-[92vw] flex-wrap items-center justify-center gap-1 rounded-full border border-border/60 bg-background/95 backdrop-blur-xl px-3 py-1.5 shadow-[0_12px_32px_rgba(0,0,0,0.18)]">
            <span className="min-w-0 px-1 text-sm text-foreground">
              {selectedCount === 1 ? (
                <>
                  {msg("auto.features.dashboard.components.bulkactionbar.1")}
                  {TERMS.optimization}
                  {msg("auto.features.dashboard.components.bulkactionbar.2")}
                </>
              ) : (
                <>
                  {msg("auto.features.dashboard.components.bulkactionbar.3")}
                  <span className="font-semibold tabular-nums">{selectedCount}</span>{" "}
                  {TERMS.optimizationPlural}
                </>
              )}
            </span>
            <div className="mx-1 h-5 w-px bg-border/60" />
            <TooltipButton
              tooltip={msg("auto.features.dashboard.components.bulkactionbar.4")}
              side="top"
              delayDuration={150}
            >
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
                aria-label={msg("auto.features.dashboard.components.bulkactionbar.literal.1")}
              >
                <X />
              </button>
            </TooltipButton>
            <TooltipButton
              tooltip={
                canCompare
                  ? skipped > 0
                    ? formatMsg("auto.features.dashboard.components.bulkactionbar.template.1", {
                        p1: skipped,
                      })
                    : msg("auto.features.dashboard.components.bulkactionbar.literal.3")
                  : msg("auto.features.dashboard.components.bulkactionbar.literal.4")
              }
              side="top"
              delayDuration={150}
            >
              <button
                type="button"
                onClick={onCompare}
                disabled={!canCompare}
                className="flex size-8 items-center justify-center rounded-full text-muted-foreground hover:bg-primary/10 hover:text-primary active:scale-95 transition-all cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-muted-foreground"
                aria-label={msg("auto.features.dashboard.components.bulkactionbar.literal.2")}
                data-tutorial="compare-button"
              >
                <ArrowLeftRight className="size-4" />
              </button>
            </TooltipButton>
            {canDelete && (
              <TooltipButton
                tooltip={msg("auto.features.dashboard.components.bulkactionbar.5")}
                side="top"
                delayDuration={150}
              >
                <button
                  type="button"
                  onClick={onRequestBulkDelete}
                  className="flex size-8 items-center justify-center rounded-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive active:scale-95 transition-all cursor-pointer"
                  aria-label={msg("auto.features.dashboard.components.bulkactionbar.literal.5")}
                >
                  <Trash2 className="size-4" />
                </button>
              </TooltipButton>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
