"use client";

import { useState } from "react";
import { toast } from "react-toastify";
import {
  ArrowLeft,
  ArrowRight,
  CaretLeft,
  CaretRight,
  CircleNotch,
  Copy,
  Crown,
  Trash,
  XCircle,
} from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Dialog, DialogContent, DialogFooter } from "@/shared/ui/primitives/dialog";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { FadeIn } from "@/shared/ui/motion";
import { ReasoningPill } from "./ui-primitives";
import { pairLabel } from "./grid-overview-helpers";
import { deleteGridPair } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { useIsPhone } from "@/shared/hooks/use-device-class";
import type { OptimizationStatusResponse, PairResult } from "@/shared/types/api";

export interface PairSelectionStripProps {
  job: OptimizationStatusResponse;
  activePair: PairResult;
  activePairIndex: number;
  pairCount: number;
  isBest: boolean;
  jobActive: boolean;
  jobTerminal: boolean;
  onBack: () => void;
  onPrev: () => void;
  onNext: () => void;
  onClone: () => void;
  onCancel: () => void;
  onDeleted: () => void;
}

export function PairSelectionStrip({
  job,
  activePair,
  activePairIndex,
  pairCount,
  isBest,
  jobActive,
  jobTerminal,
  onBack,
  onPrev,
  onNext,
  onClone,
  onCancel,
  onDeleted,
}: PairSelectionStripProps) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  // Clone opens the (desktop-only) submit wizard and delete is desk work;
  // phones keep only cancel + pair navigation.
  const isPhone = useIsPhone();

  const rtl = getActiveDir() === "rtl";
  const BackIcon = rtl ? CaretRight : CaretLeft;
  const PrevArrow = rtl ? ArrowRight : ArrowLeft;
  const NextArrow = rtl ? ArrowLeft : ArrowRight;

  const handleDeletePair = async () => {
    setDeleting(true);
    try {
      await deleteGridPair(job.optimization_id, activePair.pair_index);
      setDeleteOpen(false);
      window.dispatchEvent(new Event("optimizations-changed"));
      onDeleted();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.delete.failed"));
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <FadeIn>
        <div
          data-tutorial="pair-detail-summary"
          className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-[#C8A882]/30 bg-gradient-to-l from-[#FAF8F5] to-[#F5F1EC] p-3"
        >
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={onBack}
              className="inline-flex min-h-[44px] items-center gap-1.5 text-sm font-medium text-[#3D2E22] hover:text-[#3D2E22]/80 transition-colors cursor-pointer sm:min-h-0 [@media(hover:none)_and_(pointer:coarse)]:min-h-[44px]"
            >
              <BackIcon className="size-4" />
              <span>{msg("auto.features.optimizations.components.pairdetailview.1")}</span>
            </button>
            <span className="text-[0.6875rem] text-muted-foreground/60">|</span>
            <div className="flex items-center gap-1.5 flex-wrap min-w-0">
              {isBest && <Crown className="size-3.5 text-[#C8A882]" />}
              <span className="text-sm font-semibold text-foreground truncate">
                {activePair.generation_model.split("/").pop()}
              </span>
              <ReasoningPill value={activePair.generation_reasoning_effort} size="sm" />
              <span className="text-[0.6875rem] text-muted-foreground/50">×</span>
              <span className="text-sm font-semibold text-foreground truncate">
                {activePair.reflection_model.split("/").pop()}
              </span>
              <ReasoningPill value={activePair.reflection_reasoning_effort} size="sm" />
            </div>
          </div>
          <div className="flex w-full items-center justify-end gap-1 sm:w-auto sm:shrink-0">
            {!isPhone && (
              <TooltipButton tooltip={msg("auto.app.optimizations.id.page.4")}>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-[44px] sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                  onClick={onClone}
                  aria-label={msg("auto.app.optimizations.id.page.literal.4")}
                >
                  <Copy className="size-4" />
                </Button>
              </TooltipButton>
            )}
            {jobActive && (
              <TooltipButton tooltip={msg("auto.app.optimizations.id.page.5")}>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-[44px] text-destructive hover:bg-destructive/10 hover:text-destructive focus-visible:ring-0 focus-visible:border-0 sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                  onClick={onCancel}
                  aria-label={msg("auto.app.optimizations.id.page.literal.5")}
                >
                  <XCircle className="size-4" />
                </Button>
              </TooltipButton>
            )}
            {jobTerminal && !isPhone && (
              <TooltipButton
                tooltip={msg("auto.features.optimizations.components.gridoverview.18")}
              >
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-[44px] text-muted-foreground hover:text-red-600 sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                  onClick={() => setDeleteOpen(true)}
                  aria-label={msg("auto.features.optimizations.components.gridoverview.literal.29")}
                >
                  <Trash className="size-4" />
                </Button>
              </TooltipButton>
            )}
            <span className="mx-1 h-5 w-px bg-[#C8A882]/30" />
            <button
              type="button"
              disabled={activePairIndex <= 0}
              onClick={onPrev}
              className="inline-flex size-[44px] items-center justify-center rounded-lg hover:bg-[#3D2E22]/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
              aria-label={msg("auto.features.optimizations.components.pairdetailview.literal.1")}
              title={msg("auto.features.optimizations.components.pairdetailview.literal.1")}
            >
              <PrevArrow className="size-4 text-[#3D2E22]" />
            </button>
            <span className="text-[0.6875rem] text-muted-foreground tabular-nums font-mono">
              {activePairIndex + 1}/{pairCount}
            </span>
            <button
              type="button"
              disabled={activePairIndex >= pairCount - 1}
              onClick={onNext}
              className="inline-flex size-[44px] items-center justify-center rounded-lg hover:bg-[#3D2E22]/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
              aria-label={msg("auto.features.optimizations.components.pairdetailview.literal.2")}
              title={msg("auto.features.optimizations.components.pairdetailview.literal.2")}
            >
              <NextArrow className="size-4 text-[#3D2E22]" />
            </button>
          </div>
        </div>
      </FadeIn>

      <Dialog open={deleteOpen} onOpenChange={(open) => !open && setDeleteOpen(false)}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogTitleRow
            title={msg("auto.features.optimizations.components.gridoverview.19")}
            description={
              <>
                {msg("auto.features.optimizations.components.gridoverview.20")}{" "}
                <span className="font-mono font-semibold text-foreground break-all">
                  {pairLabel(activePair)}
                </span>
                {msg("auto.features.optimizations.components.gridoverview.21")}
              </>
            }
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteOpen(false)}
              disabled={deleting}
              className="w-full justify-center"
            >
              {msg("auto.features.optimizations.components.gridoverview.22")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeletePair}
              disabled={deleting}
              className="w-full justify-center"
            >
              {deleting ? (
                <CircleNotch className="size-4 animate-spin" />
              ) : (
                msg("auto.features.optimizations.components.gridoverview.literal.30")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
