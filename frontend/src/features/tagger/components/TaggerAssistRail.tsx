"use client";

import { useEffect, useMemo } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Check, CircleNotch, Sparkle } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Badge } from "@/shared/ui/primitives/badge";
import { formatMsg, msg } from "@/shared/lib/messages";
import { formatTaggerLabel } from "../lib/labels";
import type {
  Annotation,
  AssistPrediction,
  AssistState,
  DataRow,
  ReviewRound,
  TaggerConfig,
} from "../lib/types";
import { agreementGate, agreementOver } from "../lib/assist";

interface Props {
  config: TaggerConfig;
  assist: AssistState;
  annotations: Record<string, Annotation>;
  frameData: DataRow[];
  currentIndex: number;
  openRound: ReviewRound | null;
  /** The open round's predictions are still streaming in chunk by chunk. */
  roundPredicting?: boolean;
  onAccept: (id: string) => void;
  onGoTo: (idx: number) => void;
  onFinishRound: () => void;
  predictError: boolean;
}

/**
 * The co-pilot companion rail beside the annotation surface.
 *
 * Review is AI-first: the suggestion is the object under audit, confirmation
 * and correction both cost one keystroke. The rail is display-only chrome —
 * it never takes keyboard focus away from the annotator.
 */
export function TaggerAssistRail({
  config,
  assist,
  annotations,
  frameData,
  currentIndex,
  openRound,
  roundPredicting = false,
  onAccept,
  onGoTo,
  onFinishRound,
  predictError,
}: Props) {
  const row = frameData[currentIndex];
  const rowId = row ? String(row.id) : "";
  const prediction = rowId ? assist.predictions[rowId] : undefined;
  const gate = agreementGate(config.mode);

  const frameIds = useMemo(() => frameData.map((r) => String(r.id)), [frameData]);

  const decidedCount = openRound
    ? openRound.rowIds.filter((id) => openRound.decided[id] !== undefined).length
    : 0;
  const predictedCount = openRound
    ? openRound.rowIds.filter((id) => assist.predictions[id] !== undefined).length
    : 0;

  // A percentage over a handful of rows swings wildly and reads as a verdict
  // before the evidence is in — the meter stays "—" until the whole pass is
  // reviewed (30/30, not 3/30).
  const passComplete = openRound !== null && decidedCount === openRound.rowIds.length;
  const agreement = passComplete
    ? agreementOver(config.mode, frameIds, annotations, assist.predictions)
    : null;

  // Review: Enter confirms the AI's suggestion for the current row and moves
  // to the next unaudited one. Registered at the window so the annotator's own
  // shortcuts keep working untouched; inputs and textareas keep their Enter
  // (which is what makes this safe for freetext rows too).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "TEXTAREA" || tag === "INPUT") return;
      if (!rowId || !assist.predictions[rowId]) return;
      if (openRound?.decided[rowId] !== undefined) return;
      e.preventDefault();
      onAccept(rowId);
      const nextIdx = frameData.findIndex(
        (r, i) => i > currentIndex && openRound?.decided[String(r.id)] === undefined,
      );
      if (nextIdx >= 0) onGoTo(nextIdx);
      else if (currentIndex < frameData.length - 1) onGoTo(currentIndex + 1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [rowId, assist.predictions, openRound, frameData, currentIndex, onAccept, onGoTo]);

  // "Finish round" is gated behind a full pass, so until then the same slot
  // navigates: jump to the next row (wrapping) that still needs a decision.
  const goToNextUnreviewed = () => {
    if (!openRound) return;
    const undecided = (i: number) =>
      openRound.decided[String(frameData[i]!.id)] === undefined;
    for (let step = 1; step <= frameData.length; step++) {
      const idx = (currentIndex + step) % frameData.length;
      if (undecided(idx)) {
        onGoTo(idx);
        return;
      }
    }
  };

  return (
    <aside
      className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto lg:w-[300px]"
      aria-label={msg("tagger.assist.rail.title")}
    >
      <div className="rounded-xl border border-border/60 bg-card p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-sm font-medium text-foreground">
            <Sparkle className="size-3.5 text-primary/70" />
            {msg("tagger.assist.rail.title")}
          </span>
          <span className="text-xs text-muted-foreground tabular-nums">
            {formatMsg("tagger.assist.rail.reviewed", {
              done: decidedCount,
              total: openRound?.rowIds.length ?? 0,
            })}
          </span>
        </div>

        {/* Agreement meter — the one gold element in this view. */}
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-xs text-muted-foreground">
            {msg("tagger.assist.rail.agreement")}
          </span>
          <span className="text-sm font-semibold tabular-nums" style={{ color: "#a68b6b" }}>
            {agreement === null ? "—" : `${Math.round(agreement * 100)}%`}
          </span>
        </div>
        <div className="relative h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${Math.round((agreement ?? 0) * 100)}%`,
              background: "var(--gradient-progress)",
            }}
          />
          <div
            className="absolute top-0 h-full w-px bg-foreground/30"
            style={{ insetInlineStart: `${gate * 100}%` }}
          />
        </div>
        <p className="mt-1 text-[11px] text-muted-foreground">
          {formatMsg("tagger.assist.rail.gate", { gate: Math.round(gate * 100) })}
        </p>

        {/* Live pulse of the streaming batch: predictions land chunk by
            chunk, and this strip fills as they do, then disappears. */}
        {roundPredicting && openRound && (
          <div className="mt-3 border-t border-border/40 pt-2.5 motion-safe:animate-in motion-safe:fade-in-0">
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Sparkle
                  className="size-3 text-primary/60 motion-safe:animate-pulse"
                  aria-hidden="true"
                />
                {msg("tagger.assist.rail.tagging")}
              </span>
              <span className="text-xs text-muted-foreground tabular-nums" dir="ltr">
                {predictedCount}/{openRound.rowIds.length}
              </span>
            </div>
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary/40 transition-all duration-500"
                style={{
                  width: `${Math.round((predictedCount / Math.max(1, openRound.rowIds.length)) * 100)}%`,
                }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-border/60 bg-card p-4">
        {predictError ? (
          <p className="text-sm text-muted-foreground">{msg("tagger.assist.rail.predict_error")}</p>
        ) : (
          <ReviewPanel
            config={config}
            prediction={prediction}
            decided={openRound?.decided[rowId]}
            allDecided={openRound !== null && decidedCount === openRound.rowIds.length}
            remaining={openRound ? openRound.rowIds.length - decidedCount : 0}
            isFreetext={config.mode === "freetext"}
            isFlaggedPass={openRound?.flaggedPass === true}
            onConfirm={() => onAccept(rowId)}
            onFinishRound={onFinishRound}
            onNextUnreviewed={goToNextUnreviewed}
          />
        )}
      </div>
    </aside>
  );
}

function ReviewPanel({
  config,
  prediction,
  decided,
  allDecided,
  remaining,
  isFreetext,
  isFlaggedPass,
  onConfirm,
  onFinishRound,
  onNextUnreviewed,
}: {
  config: TaggerConfig;
  prediction: AssistPrediction | undefined;
  decided: "confirmed" | "corrected" | undefined;
  allDecided: boolean;
  remaining: number;
  isFreetext: boolean;
  isFlaggedPass: boolean;
  onConfirm: () => void;
  onFinishRound: () => void;
  onNextUnreviewed: () => void;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-2.5">
      {prediction ? (
        <Suggestion config={config} prediction={prediction} />
      ) : (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <CircleNotch className="size-3.5 animate-spin" />
          {msg("tagger.assist.rail.predicting")}
        </p>
      )}
      <div className="min-h-8 w-full">
        <AnimatePresence mode="wait" initial={false}>
          {decided === undefined && prediction ? (
            <motion.div
              key="confirm"
              exit={reduceMotion ? undefined : { opacity: 0, scale: 0.97 }}
              transition={{ duration: reduceMotion ? 0 : 0.09 }}
              className="w-full"
            >
              <Button
                variant="secondary"
                size="sm"
                onClick={onConfirm}
                className="w-full justify-center text-center"
              >
                {msg("tagger.assist.rail.confirm")}
              </Button>
            </motion.div>
          ) : decided !== undefined ? (
            <motion.div
              key={decided}
              role="status"
              initial={reduceMotion ? false : { opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={
                reduceMotion ? { duration: 0 } : { duration: 0.22, ease: [0.16, 1, 0.3, 1] }
              }
              className="flex min-h-8 w-full items-center justify-center gap-1.5 rounded-md bg-[var(--success-dim)] px-3 text-center text-xs font-medium text-[var(--success)]"
            >
              <motion.span
                initial={reduceMotion ? false : { opacity: 0, scale: 0.55, rotate: -12 }}
                animate={{ opacity: 1, scale: 1, rotate: 0 }}
                transition={
                  reduceMotion ? { duration: 0 } : { duration: 0.24, ease: [0.16, 1, 0.3, 1] }
                }
                className="inline-flex"
                aria-hidden="true"
              >
                <Check className="size-3.5" />
              </motion.span>
              <span>
                {decided === "confirmed"
                  ? msg("tagger.assist.rail.decided_confirmed")
                  : msg("tagger.assist.rail.decided_corrected")}
              </span>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
      {/* The round only closes after a full pass: until every row carries an
          explicit decision, this slot navigates to what's left instead. */}
      {(isFreetext || isFlaggedPass) &&
        (allDecided ? (
          <Button variant="secondary" size="sm" onClick={onFinishRound} className="w-full">
            {msg("tagger.assist.rail.finish_round")}
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={onNextUnreviewed} className="w-full">
            {formatMsg("tagger.assist.rail.next_unreviewed", { count: remaining })}
          </Button>
        ))}
    </div>
  );
}

function Suggestion({
  config,
  prediction,
}: {
  config: TaggerConfig;
  prediction: AssistPrediction;
}) {
  return (
    <div className="rounded-lg border border-border/50 bg-muted/30 px-3 py-2.5 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-foreground" dir="auto">
          {formatTaggerLabel(config, prediction.value)}
        </span>
        <Badge variant="ghost" size="sm" className="shrink-0 font-mono tabular-nums opacity-60">
          {Math.round(prediction.confidence * 100)}%
        </Badge>
      </div>
      {prediction.reason && (
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground" dir="auto">
          {prediction.reason}
        </p>
      )}
    </div>
  );
}
