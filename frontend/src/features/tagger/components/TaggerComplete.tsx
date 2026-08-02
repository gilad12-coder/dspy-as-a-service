"use client";

import { WarningCircle, List, User, Sparkle, SealCheck } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { Annotation, AssistState } from "../lib/types";
import { flaggedRowIds, provenanceCounts } from "../lib/assist";

interface Props {
  assist: AssistState;
  annotations: Record<string, Annotation>;
  rowCount: number;
  onFlaggedPass: () => void;
  onBrowse: () => void;
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
}: Props) {
  const counts = provenanceCounts(assist, annotations);
  const flagged = flaggedRowIds(assist);
  const credits = assist.autotag?.credits_spent ?? 0;

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
              icon={<User className="size-4 text-muted-foreground" />}
              label={msg("tagger.assist.complete.human")}
              value={counts.human}
            />
            <SummaryRow
              icon={<SealCheck className="size-4 text-muted-foreground" />}
              label={msg("tagger.assist.complete.ai_confirmed")}
              value={counts.aiConfirmed}
            />
            <SummaryRow
              icon={<Sparkle className="size-4 text-muted-foreground" />}
              label={msg("tagger.assist.complete.ai_auto")}
              value={counts.aiAuto}
            />
          </dl>

          {flagged.length === 0 && counts.aiAuto > 0 && (
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <SealCheck className="size-3.5 shrink-0 text-emerald-700" />
              {formatMsg("tagger.assist.complete.no_flags", { count: counts.aiAuto })}
            </p>
          )}

          {credits > 0 && (
            <p className="text-xs text-muted-foreground">
              {credits === 1
                ? msg("tagger.assist.complete.credits_one")
                : formatMsg("tagger.assist.complete.credits", { credits })}
            </p>
          )}

          <div className="flex flex-col gap-2">
            {flagged.length > 0 && (
              <Button onClick={onFlaggedPass} className="w-full gap-2">
                <WarningCircle className="size-4" />
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
