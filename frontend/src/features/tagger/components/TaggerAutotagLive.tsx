"use client";

import { useMemo, useState } from "react";
import { CaretLeft, CaretRight, CircleNotch } from "@/shared/ui/icons";
import { AgentPillDock } from "@/features/agent-panel";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardTitle } from "@/shared/ui/primitives/card";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { FieldsView } from "./TaggerAnnotation";
import type { Annotation, DataRow, TaggerConfig } from "../lib/types";
import { isBinaryNo, isBinaryYes } from "../lib/types";

interface Props {
  config: TaggerConfig;
  data: DataRow[];
  annotations: Record<string, Annotation>;
  status: { status: string; total: number; done: number; live: boolean } | null;
}

/** True when this row's label has been settled (by the AI or a human). */
function hasLabel(ann: Annotation, mode: TaggerConfig["mode"]): boolean {
  if (mode === "multiclass") return Array.isArray(ann) && ann.length > 0;
  return typeof ann === "string" && ann !== "";
}

/**
 * The bulk auto-tag run as a live walkthrough: a slim tqdm-style header
 * (spinner, progress bar, counts, cancel) above the same row-and-answer
 * surface the manual tagger uses — read-only, following the newest row the
 * AI labeled as batches land. Browsing back detaches from that frontier;
 * stepping forward to it (or the jump chip) re-engages the follow.
 */
export function TaggerAutotagLive({ config, data, annotations, status }: Props) {
  const rtl = getActiveDir() === "rtl";
  const PrevIcon = rtl ? CaretRight : CaretLeft;
  const NextIcon = rtl ? CaretLeft : CaretRight;
  const total = status?.total ?? data.length;
  const done = status?.done ?? 0;
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;

  // null = follow the newest labeled row; a number = the user browsed away.
  const [cursor, setCursor] = useState<number | null>(null);

  // Rows already labeled when the run began — the calibration set, which
  // stride-sampling scattered across the whole dataset (so it can reach the
  // last row). Captured once at mount so the walkthrough can ignore them and
  // follow only the rows THIS run tags, instead of snapping to whatever high
  // index calibration happened to hit.
  const [preLabeled] = useState<Set<string>>(() => {
    const seen = new Set<string>();
    for (const row of data) {
      if (hasLabel(annotations[String(row.id)], config.mode)) seen.add(String(row.id));
    }
    return seen;
  });

  // Dataset indices of the rows this run has to tag, in display order. The
  // worker labels them top-to-bottom, so the follow-cursor walks this list as
  // a contiguous prefix fills in — strictly sequential viewing, even though
  // the tagging itself runs in fast concurrent batches.
  const pendingIdx = useMemo(
    () =>
      data.reduce<number[]>((acc, row, i) => {
        if (!preLabeled.has(String(row.id))) acc.push(i);
        return acc;
      }, []),
    [data, preLabeled],
  );

  // Newest sequentially-tagged row: advance through the pending list while
  // each is labeled, stopping at the first that isn't yet. Sits on the first
  // pending row until the run's first result lands. Every dataset index up to
  // here is labeled (pending prefix + the interleaved calibration rows), so
  // browsing back never lands on a "waiting" row.
  let frontier = pendingIdx.length > 0 ? pendingIdx[0]! : 0;
  for (const i of pendingIdx) {
    if (!hasLabel(annotations[String(data[i]!.id)], config.mode)) break;
    frontier = i;
  }
  const shown = Math.min(cursor ?? frontier, Math.max(0, data.length - 1));
  const item = data[shown];
  if (!item) return null;
  const ann = annotations[String(item.id)];
  const labeled = hasLabel(ann, config.mode);
  const navigate = (dir: 1 | -1) => {
    const next = Math.max(0, Math.min(shown + dir, frontier));
    setCursor(next >= frontier ? null : next);
  };

  return (
    <div className="flex h-[calc(100dvh-var(--header-height,53px)-3rem)] flex-col overflow-hidden md:h-[calc(100dvh-var(--header-height,53px)-4rem)]">
      <div className="flex items-center gap-2 px-3 pb-1.5 pt-3 sm:gap-3 sm:px-5">
        <CircleNotch className="size-3.5 shrink-0 animate-spin text-primary" />
        <span className="shrink-0 text-xs font-medium text-foreground">
          {msg("tagger.assist.autotag.running_title")}
        </span>
        <span className="hidden min-w-0 truncate text-xs text-muted-foreground xl:block">
          {msg("tagger.assist.autotag.running_subtitle")}
        </span>
        <div className="h-1 min-w-16 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: "var(--gradient-progress)" }}
          />
        </div>
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          <span className="font-semibold text-primary">{done}</span>/{total}
        </span>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-hidden p-3 pt-2 sm:p-5 sm:pt-2 max-lg:landscape:grid max-lg:landscape:grid-cols-2">
        <Card className="flex min-h-0 flex-1 flex-col">
          <CardContent
            className="flex-1 overflow-y-auto px-4 py-4 text-base leading-relaxed text-foreground sm:px-6 sm:py-5"
            dir="auto"
          >
            {item.fields && item.fields.length > 1 ? (
              <FieldsView fields={item.fields} />
            ) : (
              <div className="whitespace-pre-wrap">{item.text}</div>
            )}
          </CardContent>
        </Card>

        <Card className="flex min-h-0 flex-1 flex-col p-3 sm:p-5">
          <CardTitle className="mb-3 text-center text-sm font-medium text-muted-foreground">
            {config.mode === "binary" &&
              (config.question ??
                msg("auto.features.tagger.components.taggerannotation.literal.1"))}
            {config.mode === "multiclass" &&
              msg("auto.features.tagger.components.taggerannotation.literal.2")}
            {config.mode === "freetext" &&
              (config.prompt ?? msg("auto.features.tagger.components.taggerannotation.literal.3"))}
          </CardTitle>

          {!labeled ? (
            <div
              role="status"
              className="flex min-h-0 flex-1 items-center justify-center gap-2 rounded-xl border border-dashed border-border/60 text-sm text-muted-foreground"
            >
              <CircleNotch className="size-4 animate-spin" />
              {msg("tagger.assist.autotag.live_waiting")}
            </div>
          ) : (
            <>
              {config.mode === "binary" && (
                <div className="flex min-h-0 flex-1 flex-col gap-2">
                  <div
                    className={cn(
                      "flex flex-1 items-center justify-center rounded-xl border text-base font-medium",
                      isBinaryYes(ann)
                        ? "border-emerald-600/40 bg-emerald-600/15 text-emerald-700"
                        : "border-border/50 text-muted-foreground/50",
                    )}
                  >
                    {msg("auto.features.tagger.components.taggerannotation.5")}
                  </div>
                  <div
                    className={cn(
                      "flex flex-1 items-center justify-center rounded-xl border text-base font-medium",
                      isBinaryNo(ann)
                        ? "border-red-500/40 bg-red-500/15 text-red-600"
                        : "border-border/50 text-muted-foreground/50",
                    )}
                  >
                    {msg("auto.features.tagger.components.taggerannotation.7")}
                  </div>
                </div>
              )}

              {config.mode === "multiclass" && (
                <div
                  className={cn(
                    "flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto",
                    (config.categories?.length ?? 0) >= 7 && "gap-1",
                  )}
                >
                  {(config.categories ?? []).map((cat) => {
                    const selected = Array.isArray(ann) && ann.includes(cat.id);
                    return (
                      <div
                        key={cat.id}
                        className={cn(
                          "flex min-h-0 flex-1 items-center justify-center rounded-xl border px-3 text-center",
                          (config.categories?.length ?? 0) >= 7 ? "text-sm" : "text-base",
                          selected
                            ? "border-primary/40 bg-primary/10 font-medium text-primary"
                            : "border-border/50 text-muted-foreground/50",
                        )}
                      >
                        <span className="min-w-0 break-words">{cat.label}</span>
                      </div>
                    );
                  })}
                </div>
              )}

              {config.mode === "freetext" && (
                <div
                  dir="auto"
                  className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap rounded-xl border border-input/60 bg-background/60 px-4 py-3 text-sm leading-relaxed"
                >
                  {typeof ann === "string" ? ann : ""}
                </div>
              )}
            </>
          )}
        </Card>

        <div className="grid grid-cols-2 gap-2 sm:flex sm:items-center sm:justify-between max-lg:landscape:col-span-2 max-lg:landscape:flex max-lg:landscape:items-center max-lg:landscape:justify-between">
          <Button
            variant="outline"
            onClick={() => navigate(-1)}
            disabled={shown === 0}
            className="min-h-[44px] w-full gap-2 sm:w-auto max-lg:landscape:w-auto lg:min-h-0"
          >
            <PrevIcon className="size-4" />
            {msg("auto.features.tagger.components.taggerannotation.8")}
          </Button>

          <div className="col-span-2 row-start-2 flex min-w-0 flex-wrap items-center justify-center gap-2 sm:col-auto sm:row-auto sm:gap-3">
            <span className="text-xs tabular-nums text-muted-foreground">
              {formatMsg("tagger.assist.autotag.live_position", {
                row: shown + 1,
                total: data.length,
              })}
            </span>
            {cursor !== null && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCursor(null)}
                className="text-muted-foreground"
              >
                {msg("tagger.assist.autotag.follow_latest")}
              </Button>
            )}
            {/* The floating agent pill would cover the Next button on this
                viewport-locked surface; docking it here keeps it out of the
                way and reachable. */}
            <AgentPillDock />
          </div>

          <Button
            variant="outline"
            onClick={() => navigate(1)}
            disabled={shown >= frontier}
            className="col-start-2 row-start-1 min-h-[44px] w-full gap-2 sm:col-auto sm:row-auto sm:w-auto max-lg:landscape:w-auto lg:min-h-0"
          >
            {msg("auto.features.tagger.components.taggerannotation.13")}
            <NextIcon className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
