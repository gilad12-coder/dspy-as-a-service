"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, Loader2, OctagonX } from "lucide-react";
import { AgentPillDock } from "@/features/agent-panel";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardTitle } from "@/shared/ui/primitives/card";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { FieldsView } from "./TaggerAnnotation";
import type { Annotation, DataRow, TaggerConfig } from "../lib/types";

interface Props {
  config: TaggerConfig;
  data: DataRow[];
  annotations: Record<string, Annotation>;
  status: { status: string; total: number; done: number; live: boolean } | null;
  onCancel: () => void;
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
export function TaggerAutotagLive({ config, data, annotations, status, onCancel }: Props) {
  const rtl = getActiveDir() === "rtl";
  const PrevIcon = rtl ? ChevronRight : ChevronLeft;
  const NextIcon = rtl ? ChevronLeft : ChevronRight;
  const total = status?.total ?? data.length;
  const done = status?.done ?? 0;
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;

  // null = follow the newest labeled row; a number = the user browsed away.
  const [cursor, setCursor] = useState<number | null>(null);

  // The worker tags in dataset order, so the newest labeled row is the last
  // labeled index. Recomputed per poll-driven render; datasets are small.
  let frontier = 0;
  for (let i = data.length - 1; i >= 0; i--) {
    if (hasLabel(annotations[String(data[i]!.id)], config.mode)) {
      frontier = i;
      break;
    }
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
      <div className="flex items-center gap-3 px-5 pt-3 pb-1.5">
        <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" />
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
        <Button
          variant="ghost"
          size="sm"
          onClick={onCancel}
          className="shrink-0 gap-1.5 text-muted-foreground"
        >
          <OctagonX className="size-3.5" />
          {msg("tagger.assist.autotag.cancel")}
        </Button>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-hidden p-5 pt-2">
        <Card className="flex min-h-0 flex-1 flex-col">
          <CardContent
            className="flex-1 overflow-y-auto px-6 py-5 text-base leading-relaxed text-foreground"
            dir="auto"
          >
            {item.fields && item.fields.length > 1 ? (
              <FieldsView fields={item.fields} />
            ) : (
              <div className="whitespace-pre-wrap">{item.text}</div>
            )}
          </CardContent>
        </Card>

        <Card className="flex min-h-0 flex-1 flex-col p-5">
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
              <Loader2 className="size-4 animate-spin" />
              {msg("tagger.assist.autotag.live_waiting")}
            </div>
          ) : (
            <>
              {config.mode === "binary" && (
                <div className="flex min-h-0 flex-1 flex-col gap-2">
                  <div
                    className={cn(
                      "flex flex-1 items-center justify-center rounded-xl border text-base font-medium",
                      ann === "yes"
                        ? "border-emerald-600/40 bg-emerald-600/15 text-emerald-700"
                        : "border-border/50 text-muted-foreground/50",
                    )}
                  >
                    {msg("auto.features.tagger.components.taggerannotation.5")}
                  </div>
                  <div
                    className={cn(
                      "flex flex-1 items-center justify-center rounded-xl border text-base font-medium",
                      ann === "no"
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

        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={() => navigate(-1)}
            disabled={shown === 0}
            className="gap-2"
          >
            <PrevIcon className="size-4" />
            {msg("auto.features.tagger.components.taggerannotation.8")}
          </Button>

          <div className="flex items-center gap-3">
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
            className="gap-2"
          >
            {msg("auto.features.tagger.components.taggerannotation.13")}
            <NextIcon className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
