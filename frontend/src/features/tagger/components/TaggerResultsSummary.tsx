"use client";

import { SealCheck, Sparkle, User, WarningCircle } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { Annotation, AssistState } from "../lib/types";
import { flaggedRowIds, provenanceCounts } from "../lib/assist";

interface Props {
  assist: AssistState;
  annotations: Record<string, Annotation>;
  /**
   * Opens a review pass over the low-confidence auto-tagged rows. Omitted for
   * read-only viewers, who see the breakdown but can't act on it.
   */
  onFlaggedPass?: () => void;
}

/**
 * The provenance/cost accounting that used to be a standalone completion card,
 * folded into a slim strip above the results table: the same who-labeled-what
 * counts (in the table's own Source vocabulary) and credit line, plus the
 * one-click flagged pass — so finishing a run flows straight into browsing
 * instead of stopping on an interstitial.
 */
export function TaggerResultsSummary({ assist, annotations, onFlaggedPass }: Props) {
  const counts = provenanceCounts(assist, annotations);
  const flagged = flaggedRowIds(assist);
  const credits = assist.autotag?.credits_spent ?? 0;

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-lg border border-border/60 bg-card px-4 py-2.5">
      <Stat
        icon={<User className="size-4 text-muted-foreground" />}
        label={msg("tagger.results.who.human")}
        value={counts.human}
      />
      <Stat
        icon={<SealCheck className="size-4 text-muted-foreground" />}
        label={msg("tagger.results.who.ai_confirmed")}
        value={counts.aiConfirmed}
      />
      <Stat
        icon={<Sparkle className="size-4 text-muted-foreground" />}
        label={msg("tagger.results.who.ai_auto")}
        value={counts.aiAuto}
      />
      {credits > 0 && (
        <span className="text-xs text-muted-foreground">
          {credits === 1
            ? msg("tagger.assist.complete.credits_one")
            : formatMsg("tagger.assist.complete.credits", { credits })}
        </span>
      )}
      {onFlaggedPass && flagged.length > 0 && (
        <Button variant="outline" size="sm" onClick={onFlaggedPass} className="ms-auto gap-1.5">
          <WarningCircle className="size-3.5" />
          {formatMsg("tagger.assist.complete.flagged_cta", { count: flagged.length })}
        </Button>
      )}
    </div>
  );
}

function Stat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
}) {
  return (
    <span className="flex items-center gap-1.5 text-sm">
      {icon}
      <span className="text-muted-foreground">{label}</span>
      <span className="font-semibold tabular-nums text-foreground">{value}</span>
    </span>
  );
}
