"use client";

import { motion } from "framer-motion";
import { GitBranch } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OptimizationStatusResponse } from "@/shared/types/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { TERMS } from "@/shared/lib/terms";
import { cn } from "@/shared/lib/utils";
import { useLiteMode } from "@/features/settings";
import {
  extractCandidates,
  extractMinibatch,
  extractRejected,
  extractValset,
  extractValsetOutputs,
} from "../lib/extract-events";
import { layoutTrajectory } from "../lib/layout";
import { TrajectoryTree } from "./TrajectoryTree";
import { TrajectoryOutline } from "./TrajectoryOutline";
import { TrajectoryDrawer, type DrawerSelection } from "./TrajectoryDrawer";

const NEWEST_HIGHLIGHT_MS = 2200;

type Selected = { kind: "candidate" | "rejected"; id: string };

function isLive(job: OptimizationStatusResponse): boolean {
  return job.status === "running" || job.status === "validating" || job.status === "pending";
}

export interface TrajectoryPanelProps {
  job: OptimizationStatusResponse;
  // When set, only candidate events tagged with this pair_index are kept —
  // grid-search pair views need to scope the tree to a single pair.
  pairIndex?: number;
  // Forwarded to TrajectoryTree — see its prop docs. The tutorial demo uses
  // it to open the tree at the eventual extent before any node streams in.
  previewLayout?: { width: number; height: number };
  // Tool name → approval severity from the run's persisted react_overlay,
  // forwarded to the drawer so its tool cards match the Code tab.
  toolSeverities?: Record<string, string>;
}

export function TrajectoryPanel({
  job,
  pairIndex,
  previewLayout,
  toolSeverities,
}: TrajectoryPanelProps) {
  const live = isLive(job);
  const lite = useLiteMode();
  const { candidates, rejected, valsetRows, minibatch, valsetOutputs } = useMemo(() => {
    const events = job.progress_events ?? [];
    const scoped =
      pairIndex === undefined
        ? events
        : events.filter((e) => e.metrics?.pair_index === pairIndex);
    return {
      candidates: extractCandidates(scoped),
      rejected: extractRejected(scoped),
      valsetRows: extractValset(events),
      minibatch: extractMinibatch(scoped),
      valsetOutputs: extractValsetOutputs(scoped),
    };
  }, [job.progress_events, pairIndex]);
  const maxGeneration = useMemo(() => {
    let m = 0;
    for (const c of candidates) if (c.generation > m) m = c.generation;
    return m;
  }, [candidates]);
  const [generationFilter, setGenerationFilter] = useState<number | null>(null);
  const visibleCandidates = useMemo(() => {
    if (generationFilter === null) return candidates;
    return candidates.filter((c) => c.generation <= generationFilter);
  }, [candidates, generationFilter]);
  const visibleRejected = useMemo(() => {
    if (generationFilter === null) return rejected;
    const visibleIds = new Set(visibleCandidates.map((c) => c.candidate_id));
    return rejected.filter((r) => visibleIds.has(r.parent_id));
  }, [rejected, generationFilter, visibleCandidates]);
  const layout = useMemo(
    () => layoutTrajectory(visibleCandidates, visibleRejected),
    [visibleCandidates, visibleRejected],
  );
  const [selected, setSelected] = useState<Selected | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [newestId, setNewestId] = useState<string | null>(null);
  const newestTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevCountRef = useRef(0);
  const liveRegionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (candidates.length === 0) {
      setSelected(null);
      return;
    }
    if (
      selected !== null &&
      selected.kind === "candidate" &&
      candidates.some((c) => c.candidate_id === selected.id)
    ) {
      return;
    }
    if (selected !== null && selected.kind === "rejected") {
      return;
    }
    if (layout.winnerId !== null) {
      setSelected({ kind: "candidate", id: layout.winnerId });
      return;
    }
    const first = candidates[0];
    if (first !== undefined) setSelected({ kind: "candidate", id: first.candidate_id });
  }, [candidates, layout.winnerId, selected]);

  useEffect(() => {
    if (candidates.length <= prevCountRef.current) {
      prevCountRef.current = candidates.length;
      return;
    }
    const newest = candidates[candidates.length - 1];
    prevCountRef.current = candidates.length;
    if (newest === undefined) return;
    setNewestId(newest.candidate_id);
    if (liveRegionRef.current !== null) {
      liveRegionRef.current.textContent = msg("trajectory.live.new_candidate");
    }
    if (newestTimerRef.current !== null) clearTimeout(newestTimerRef.current);
    newestTimerRef.current = setTimeout(() => setNewestId(null), NEWEST_HIGHLIGHT_MS);
    return () => {
      if (newestTimerRef.current !== null) clearTimeout(newestTimerRef.current);
    };
    // Tracking length only (a primitive) avoids re-running on every parent
    // render when ``candidates`` is reallocated by the upstream useMemo,
    // and keeps the deps shape stable across HMR-driven refreshes that
    // otherwise compared dep arrays of different lengths.
  }, [candidates.length]);

  const drawerSelection: DrawerSelection = useMemo(() => {
    if (selected === null) return null;
    if (selected.kind === "candidate") {
      const node = layout.nodes.find((n) => n.candidate_id === selected.id);
      if (node === undefined) return null;
      const parent =
        node.parent_id === null
          ? null
          : (layout.nodes.find((n) => n.candidate_id === node.parent_id) ?? null);
      return { kind: "candidate", node, parent };
    }
    const ghost = layout.ghosts.find((g) => g.rejection_id === selected.id);
    if (ghost === undefined) return null;
    const parent = layout.nodes.find((n) => n.candidate_id === ghost.parent_id) ?? null;
    return { kind: "rejected", ghost, parent };
  }, [layout.nodes, layout.ghosts, selected]);

  const selectedTreeId =
    selected !== null && selected.kind === "candidate" ? selected.id : null;

  const handleSelectCandidate = useCallback((id: string) => {
    setSelected({ kind: "candidate", id });
    setDrawerOpen(true);
  }, []);

  const handleSelectRejected = useCallback((id: string) => {
    setSelected({ kind: "rejected", id });
    setDrawerOpen(true);
  }, []);

  if (candidates.length === 0) return null;

  return (
    <FadeIn delay={0.12}>
      <Card
        className="relative overflow-hidden shadow-[0_1px_3px_rgba(28,22,18,0.04),inset_0_1px_0_rgba(255,255,255,0.5)]"
        data-tutorial="trajectory-panel"
      >
        <div
          className="absolute inset-x-0 top-0 h-px bg-gradient-to-l from-transparent via-[#C8A882]/40 to-transparent"
          aria-hidden="true"
        />
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <CardTitle className="text-base flex min-w-0 items-center gap-2">
              <GitBranch className="size-4 text-[#7C6350]" aria-hidden="true" />
              <HelpTip text={msg("trajectory.explainer.trajectory")}>
                <span className="font-bold tracking-tight">{msg("trajectory.panel.title")}</span>
              </HelpTip>
            </CardTitle>
          </div>
          {live ? (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex shrink-0 items-center gap-1.5 rounded-full border border-border/40 bg-background/80 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
            >
              <span className="relative inline-flex size-2" aria-hidden="true">
                <motion.span
                  className="absolute inset-0 rounded-full bg-[var(--warning)]/40"
                  animate={{ scale: [1, 2, 1], opacity: [0.6, 0, 0.6] }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
                />
                <span className="relative inline-block size-2 rounded-full bg-[var(--warning)]" />
              </span>
              <span className="tabular-nums">{candidates.length}</span>
              <span>{TERMS.candidatePlural}</span>
            </motion.div>
          ) : (
            <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              <span className="tabular-nums">{candidates.length}</span> {TERMS.candidatePlural}
            </span>
          )}
        </CardHeader>
        <CardContent className="space-y-3">
          {maxGeneration > 0 ? (
            <GenerationTimeline
              maxGeneration={maxGeneration}
              value={generationFilter}
              onChange={setGenerationFilter}
              isLive={live}
            />
          ) : null}
          {lite ? (
            <TrajectoryOutline
              candidates={visibleCandidates}
              rejected={visibleRejected}
              winnerId={layout.winnerId}
              selectedId={selectedTreeId}
              newestId={newestId}
              onSelectCandidate={handleSelectCandidate}
              onSelectRejected={handleSelectRejected}
            />
          ) : (
            <TrajectoryTree
              layout={layout}
              selectedId={selectedTreeId}
              newestId={newestId}
              onSelectCandidate={handleSelectCandidate}
              onSelectRejected={handleSelectRejected}
              previewLayout={previewLayout}
            />
          )}
          <TrajectoryDrawer
            selection={drawerSelection}
            open={drawerOpen}
            onOpenChange={setDrawerOpen}
            valsetRows={valsetRows}
            minibatch={minibatch}
            valsetOutputs={valsetOutputs}
            toolSeverities={toolSeverities}
          />
          <div
            ref={liveRegionRef}
            role="status"
            aria-live="polite"
            className="sr-only"
          />
        </CardContent>
      </Card>
    </FadeIn>
  );
}

function GenerationTimeline({
  maxGeneration,
  value,
  onChange,
  isLive,
}: {
  maxGeneration: number;
  value: number | null;
  onChange: (next: number | null) => void;
  isLive: boolean;
}) {
  const current = value ?? maxGeneration;
  const isAtLive = value === null;
  const isRtl = getActiveDir() === "rtl";
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const genFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (el === null) return current;
      const rect = el.getBoundingClientRect();
      // Generation 0 sits at the inline-start edge — the track's right in RTL,
      // its left in LTR — so measure the drag from that edge in either direction.
      const offset = isRtl ? rect.right - clientX : clientX - rect.left;
      const pct = Math.max(0, Math.min(1, offset / rect.width));
      return Math.round(pct * maxGeneration);
    },
    [current, maxGeneration, isRtl],
  );

  const applyValue = useCallback(
    (next: number) => {
      onChange(next >= maxGeneration ? null : next);
    },
    [onChange, maxGeneration],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      setDragging(true);
      applyValue(genFromClientX(e.clientX));
    },
    [applyValue, genFromClientX],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return;
      applyValue(genFromClientX(e.clientX));
    },
    [dragging, applyValue, genFromClientX],
  );

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
    setDragging(false);
  }, []);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      // Later generations sit toward the inline-end edge — visually left in RTL,
      // right in LTR — so the arrow pointing that way means "forward in time".
      const forwardKey = isRtl ? "ArrowLeft" : "ArrowRight";
      const backKey = isRtl ? "ArrowRight" : "ArrowLeft";
      if (e.key === forwardKey) {
        e.preventDefault();
        applyValue(Math.min(maxGeneration, current + 1));
      } else if (e.key === backKey) {
        e.preventDefault();
        applyValue(Math.max(0, current - 1));
      } else if (e.key === "Home") {
        e.preventDefault();
        applyValue(0);
      } else if (e.key === "End") {
        e.preventDefault();
        applyValue(maxGeneration);
      }
    },
    [applyValue, current, maxGeneration, isRtl],
  );

  const filledPct = maxGeneration === 0 ? 100 : (current / maxGeneration) * 100;
  const steps = useMemo(
    () => Array.from({ length: maxGeneration + 1 }, (_, i) => i),
    [maxGeneration],
  );

  return (
    <div
      className="rounded-xl border border-border/40 bg-background/70 px-4 pt-3 pb-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]"
      dir={isRtl ? "rtl" : "ltr"}
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {msg("trajectory.scrubber.label")}
        </span>
      </div>

      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label={msg("trajectory.scrubber.label")}
        aria-valuemin={0}
        aria-valuemax={maxGeneration}
        aria-valuenow={current}
        aria-valuetext={
          isAtLive
            ? msg("trajectory.scrubber.live")
            : formatMsg("trajectory.scrubber.generation_value", { gen: current })
        }
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onKeyDown={onKeyDown}
        className="relative h-9 cursor-pointer touch-none select-none rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/60"
      >
        <div
          className="absolute inset-x-0 top-1/2 h-[2px] -translate-y-1/2 rounded-full"
          style={{ background: "rgba(28, 22, 18, 0.10)" }}
        />
        <div
          className="absolute top-1/2 h-[2px] -translate-y-1/2 rounded-full bg-[#7C6350]"
          style={{ ...(isRtl ? { right: 0 } : { left: 0 }), width: `${filledPct}%` }}
        />

        {steps.map((gen) => {
          const pct = maxGeneration === 0 ? 0 : (gen / maxGeneration) * 100;
          const isPast = gen <= current;
          const isActive = gen === current;
          if (isActive) return null;
          return (
            <span
              key={`tick-${gen}`}
              className="pointer-events-none absolute top-1/2 inline-flex items-center justify-center rounded-sm bg-background/95 px-1 text-[10px] tabular-nums font-semibold leading-none"
              style={{
                ...(isRtl ? { right: `${pct}%` } : { left: `${pct}%` }),
                transform: isRtl ? "translate(50%, -50%)" : "translate(-50%, -50%)",
                color: isPast ? "#7C6350" : "rgba(28, 22, 18, 0.42)",
              }}
              aria-hidden="true"
            >
              {gen}
            </span>
          );
        })}

        <div
          className="pointer-events-none absolute top-1/2 z-10"
          style={{
            ...(isRtl ? { right: `${filledPct}%` } : { left: `${filledPct}%` }),
            transform: isRtl ? "translate(50%, -50%)" : "translate(-50%, -50%)",
          }}
        >
          {isAtLive && isLive ? (
            <motion.span
              className="absolute left-1/2 top-1/2 block h-8 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#7C8B5A]/30"
              animate={{ scale: [1, 1.7, 1], opacity: [0.55, 0, 0.55] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
              aria-hidden="true"
            />
          ) : null}
          <div
            className={cn(
              "relative block h-4 w-4 rounded-full border-[1.5px] shadow-[0_1px_2px_rgba(28,22,18,0.18)] transition-transform",
              isAtLive
                ? "border-[#1c1612] bg-[#1c1612]"
                : "border-[#1c1612] bg-[#fbf8f3]",
              dragging && "scale-110",
            )}
          />
        </div>
      </div>
    </div>
  );
}
