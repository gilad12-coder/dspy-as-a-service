"use client";

import { Loader2, OctagonX, RotateCcw } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { formatMsg, msg } from "@/shared/lib/messages";

interface Props {
  status: { status: string; total: number; done: number; live: boolean } | null;
  onCancel: () => void;
  onResume: () => void;
  onBrowse: () => void;
}

/**
 * Calm progress surface for the bulk auto-tag job. The job runs server-side —
 * leaving the page is safe — so this view only narrates: live counters, a
 * cancel that keeps everything tagged so far, and honest recovery paths when
 * the job was interrupted or failed.
 */
export function TaggerAutotagProgress({ status, onCancel, onResume, onBrowse }: Props) {
  const total = status?.total ?? 0;
  const done = status?.done ?? 0;
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  const running = status?.status === "running" && status.live;
  const interrupted = status?.status === "running" && status !== null && !status.live;
  const failed = status?.status === "failed";
  const canceled = status?.status === "canceled";

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-4 pt-10">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            {running && <Loader2 className="size-4 animate-spin text-primary" />}
            {running
              ? msg("tagger.assist.autotag.running_title")
              : interrupted
                ? msg("tagger.assist.autotag.interrupted_title")
                : failed
                  ? msg("tagger.assist.autotag.failed_title")
                  : canceled
                    ? msg("tagger.assist.autotag.canceled_title")
                    : msg("tagger.assist.autotag.starting_title")}
          </CardTitle>
          <CardDescription>
            {running
              ? msg("tagger.assist.autotag.running_subtitle")
              : interrupted || failed || canceled
                ? msg("tagger.assist.autotag.recover_subtitle")
                : msg("tagger.assist.autotag.starting_subtitle")}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div>
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="text-sm text-muted-foreground">
                {msg("tagger.assist.autotag.progress_label")}
              </span>
              <span className="text-sm font-semibold tabular-nums text-primary">
                {formatMsg("tagger.assist.autotag.progress_count", { done, total })}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${pct}%`, background: "var(--gradient-progress)" }}
              />
            </div>
          </div>

          {running && (
            <Button variant="outline" onClick={onCancel} className="w-full gap-2">
              <OctagonX className="size-4" />
              {msg("tagger.assist.autotag.cancel")}
            </Button>
          )}
          {(interrupted || failed || canceled) && (
            <div className="flex flex-col gap-2">
              <Button onClick={onResume} className="w-full gap-2">
                <RotateCcw className="size-4" />
                {msg("tagger.assist.autotag.resume")}
              </Button>
              <Button variant="outline" onClick={onBrowse} className="w-full">
                {msg("tagger.assist.autotag.browse")}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
