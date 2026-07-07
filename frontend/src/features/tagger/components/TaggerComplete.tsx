"use client";

import { CircleAlert, List, Loader2, Rocket, UserRound, Sparkles, BadgeCheck } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { Annotation, AssistState } from "../lib/types";
import { MIN_DEEP_OPTIMIZE_EXAMPLES, flaggedRowIds, provenanceCounts } from "../lib/assist";
import { LiftLine, OpenRunLink } from "./TaggerReviewGate";

interface Props {
  assist: AssistState;
  annotations: Record<string, Annotation>;
  rowCount: number;
  onFlaggedPass: () => void;
  onBrowse: () => void;
  onDeepOptimize: () => void;
}

/**
 * The completion summary: an honest accounting of who labeled what (the
 * provenance breakdown), what it cost, and which auto-tagged rows the model
 * itself wasn't sure about — with a one-click pass over exactly those.
 */
export function TaggerComplete({
  assist,
  annotations,
  rowCount,
  onFlaggedPass,
  onBrowse,
  onDeepOptimize,
}: Props) {
  const counts = provenanceCounts(assist, annotations);
  const flagged = flaggedRowIds(assist);
  const credits = assist.autotag?.credits_spent ?? 0;
  const vettedLabels = counts.human + counts.aiConfirmed;

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-4 pt-10">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{msg("tagger.assist.complete.title")}</CardTitle>
          <CardDescription>
            {formatMsg("tagger.assist.complete.subtitle", { total: rowCount })}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <dl className="flex flex-col gap-2">
            <SummaryRow
              icon={<UserRound className="size-4 text-muted-foreground" />}
              label={msg("tagger.assist.complete.human")}
              value={counts.human}
            />
            <SummaryRow
              icon={<BadgeCheck className="size-4 text-muted-foreground" />}
              label={msg("tagger.assist.complete.ai_confirmed")}
              value={counts.aiConfirmed}
            />
            <SummaryRow
              icon={<Sparkles className="size-4 text-muted-foreground" />}
              label={msg("tagger.assist.complete.ai_auto")}
              value={counts.aiAuto}
            />
          </dl>

          {credits > 0 && (
            <p className="text-xs text-muted-foreground">
              {formatMsg("tagger.assist.complete.credits", { credits })}
            </p>
          )}

          <div className="flex flex-col gap-2">
            {flagged.length > 0 && (
              <Button onClick={onFlaggedPass} className="w-full gap-2">
                <CircleAlert className="size-4" />
                {formatMsg("tagger.assist.complete.flagged_cta", { count: flagged.length })}
              </Button>
            )}
            <Button
              variant={flagged.length > 0 ? "outline" : "default"}
              onClick={onBrowse}
              className="w-full gap-2"
            >
              <List className="size-4" />
              {msg("tagger.assist.complete.browse")}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            {msg("tagger.assist.complete.optimization_title")}
          </CardTitle>
          <CardDescription>
            {msg("tagger.assist.complete.optimization_description")}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {assist.deepOptimize?.status === "running" ? (
            <>
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                {msg("tagger.assist.optimize.running")}
              </p>
              <OpenRunLink jobId={assist.deepOptimize.jobId} />
            </>
          ) : assist.deepOptimize?.status === "success" ? (
            <>
              <LiftLine deepOptimize={assist.deepOptimize} />
              <OpenRunLink jobId={assist.deepOptimize.jobId} />
            </>
          ) : (
            <>
              <Button
                variant="secondary"
                onClick={onDeepOptimize}
                disabled={vettedLabels < MIN_DEEP_OPTIMIZE_EXAMPLES}
                className="w-full gap-2"
              >
                <Rocket className="size-4" />
                {msg("tagger.assist.complete.optimization_cta")}
              </Button>
              {vettedLabels < MIN_DEEP_OPTIMIZE_EXAMPLES && (
                <p className="text-xs text-muted-foreground">
                  {formatMsg("tagger.assist.complete.optimization_too_few", {
                    minimum: MIN_DEEP_OPTIMIZE_EXAMPLES,
                  })}
                </p>
              )}
              {assist.deepOptimize?.status === "failed" && (
                <p className="text-xs text-muted-foreground">
                  {msg("tagger.assist.optimize.failed")}
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function SummaryRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="flex items-center gap-2 text-sm text-muted-foreground">
        {icon}
        {label}
      </dt>
      <dd className="text-sm font-semibold tabular-nums text-foreground">{value}</dd>
    </div>
  );
}
