"use client";

import { memo } from "react";
import { Trophy } from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { CandidateMetrics, RejectedMetrics } from "../lib/types";

interface TrajectoryOutlineProps {
  candidates: CandidateMetrics[];
  rejected: RejectedMetrics[];
  winnerId: string | null;
  selectedId: string | null;
  newestId: string | null;
  onSelectCandidate: (id: string) => void;
  onSelectRejected: (id: string) => void;
}

/**
 * Lite-mode stand-in for the SVG TrajectoryTree: a flat text outline of
 * candidates (grouped by generation) and rejected proposals. No pan/zoom layer
 * and no continuous SVG redraw, so a live optimization doesn't tax a weak
 * machine. Rows open the same detail drawer the tree does, so the drill-down
 * interaction is preserved.
 */
function TrajectoryOutlineImpl({
  candidates,
  rejected,
  winnerId,
  selectedId,
  newestId,
  onSelectCandidate,
  onSelectRejected,
}: TrajectoryOutlineProps) {
  const byGeneration = new Map<number, CandidateMetrics[]>();
  for (const c of candidates) {
    const list = byGeneration.get(c.generation);
    if (list) list.push(c);
    else byGeneration.set(c.generation, [c]);
  }
  const generations = [...byGeneration.entries()].sort((a, b) => a[0] - b[0]);

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        {generations.map(([gen, items]) => (
          <div key={gen} className="space-y-1">
            <p className="px-1 text-[0.7rem] font-semibold uppercase tracking-wide text-muted-foreground">
              {formatMsg("trajectory.scrubber.generation_value", { gen })}
            </p>
            <ul className="space-y-1">
              {items.map((c) => {
                const isWinner = c.candidate_id === winnerId;
                const isSelected = c.candidate_id === selectedId;
                const isNewest = c.candidate_id === newestId;
                return (
                  <li key={c.candidate_id}>
                    <button
                      type="button"
                      onClick={() => onSelectCandidate(c.candidate_id)}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-lg border px-3 py-1.5 text-start",
                        isSelected
                          ? "border-[#B04030]/40 bg-[#B04030]/[0.05]"
                          : "border-border bg-card hover:bg-accent/60",
                      )}
                    >
                      {isWinner && (
                        <Trophy
                          className="size-3.5 shrink-0 text-[#B07A30]"
                          aria-label={msg("trajectory.outline.best")}
                        />
                      )}
                      <span className="min-w-0 flex-1 truncate text-sm text-foreground">
                        {formatMsg("trajectory.node.header.accepted_title", { id: c.candidate_id })}
                        {isNewest && (
                          <span
                            className="ms-2 inline-block size-1.5 rounded-full bg-[#B04030] align-middle"
                            aria-hidden="true"
                          />
                        )}
                      </span>
                      <span
                        className="font-mono text-sm font-semibold tabular-nums text-foreground"
                        dir="ltr"
                      >
                        {c.score.toFixed(2)}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      {rejected.length > 0 && (
        <details className="rounded-lg border border-border bg-card">
          <summary className="cursor-pointer px-3 py-2 text-[0.7rem] font-semibold uppercase tracking-wide text-muted-foreground">
            {msg("trajectory.ghost.legend")} ({rejected.length})
          </summary>
          <ul className="space-y-1 px-2 pb-2">
            {rejected.map((r) => (
              <li key={r.rejection_id}>
                <button
                  type="button"
                  onClick={() => onSelectRejected(r.rejection_id)}
                  className="flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-start hover:bg-accent/60"
                >
                  <span className="min-w-0 flex-1 truncate text-sm text-muted-foreground">
                    {formatMsg("trajectory.outline.rejected_row", { id: r.rejection_id })}
                  </span>
                  <span
                    className="font-mono text-sm tabular-nums text-muted-foreground"
                    dir="ltr"
                  >
                    {r.proposal_score.toFixed(2)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export const TrajectoryOutline = memo(TrajectoryOutlineImpl);
