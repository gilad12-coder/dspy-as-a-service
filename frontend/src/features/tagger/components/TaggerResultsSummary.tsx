"use client";

import { Coins, SealCheck, Sparkle, User, WarningCircle } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
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

// A dark-brand → brass → light-brass ramp: the human's own work reads as the
// brand tone, and the two AI provenances share the brass accent (deeper for the
// rows you verified, lighter for the ones left on autopilot). The same three
// colours key the icons, the count labels, and the proportion bar's segments.
const SOURCE_COLOR = {
  human: "var(--primary)",
  ai_confirmed: "#a68b6b",
  ai_auto: "#c8a882",
} as const;

/**
 * The provenance/cost accounting that used to be a standalone completion card,
 * folded into a recap band above the results table: a titled who-did-what
 * header, the same counts (in the table's own Source vocabulary) as prominent
 * tabular figures — each spelled out so its colour is unambiguous — the
 * auto-labeling cost as a fourth figure, a proportion bar showing how the
 * labeled set splits between you and the AI, and the one-click flagged pass, so
 * finishing a run flows straight into browsing instead of an interstitial.
 */
export function TaggerResultsSummary({ assist, annotations, onFlaggedPass }: Props) {
  const counts = provenanceCounts(assist, annotations);
  const flagged = flaggedRowIds(assist);
  const credits = assist.autotag?.credits_spent ?? 0;

  const stats = [
    {
      key: "human" as const,
      icon: User,
      label: msg("tagger.results.who.human"),
      hint: msg("tagger.results.recap.human"),
      value: counts.human,
    },
    {
      key: "ai_confirmed" as const,
      icon: SealCheck,
      label: msg("tagger.results.who.ai_confirmed"),
      hint: msg("tagger.results.recap.ai_confirmed"),
      value: counts.aiConfirmed,
    },
    {
      key: "ai_auto" as const,
      icon: Sparkle,
      label: msg("tagger.results.who.ai_auto"),
      hint: msg("tagger.results.recap.ai_auto"),
      value: counts.aiAuto,
    },
  ];
  const total = stats.reduce((sum, s) => sum + s.value, 0);

  return (
    <div className="overflow-hidden rounded-xl border border-border/60 bg-card">
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 py-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-foreground">
            {msg("tagger.results.recap.title")}
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {formatMsg("tagger.results.recap.subtitle", { total })}
          </p>
        </div>
        {onFlaggedPass && flagged.length > 0 && (
          <Button variant="outline" size="sm" onClick={onFlaggedPass} className="shrink-0 gap-1.5">
            <WarningCircle className="size-3.5" />
            {formatMsg("tagger.assist.complete.flagged_cta", { count: flagged.length })}
          </Button>
        )}
      </div>

      <div className="flex flex-wrap items-stretch gap-y-3 border-t border-border/60 px-4 py-3.5">
        {stats.map((s, i) => (
          <StatCell
            key={s.key}
            first={i === 0}
            value={s.value}
            icon={s.icon}
            color={SOURCE_COLOR[s.key]}
            label={s.label}
            hint={s.hint}
          />
        ))}
        {credits > 0 && (
          <StatCell
            value={credits}
            icon={Coins}
            label={msg("tagger.results.recap.credits")}
            hint={msg("tagger.results.recap.credits_hint")}
          />
        )}
      </div>

      {total > 0 && (
        <div className="flex h-1.5 w-full overflow-hidden bg-muted" aria-hidden="true">
          {stats.map(
            (s) =>
              s.value > 0 && (
                <div
                  key={s.key}
                  style={{ width: `${(s.value / total) * 100}%`, background: SOURCE_COLOR[s.key] }}
                />
              ),
          )}
        </div>
      )}
    </div>
  );
}

/**
 * One figure in the recap band: a big tabular count over a colour-keyed
 * icon+label and a spelled-out hint. `color` tints the icon (a provenance's
 * ramp tone, or the muted default for the cost cell); `first` drops the leading
 * divider so only inter-cell borders show.
 */
function StatCell({
  value,
  icon: Icon,
  label,
  hint,
  color = "var(--muted-foreground)",
  first = false,
}: {
  value: number;
  icon: typeof User;
  label: string;
  hint: string;
  color?: string;
  first?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-1.5 px-5",
        first ? "ps-0" : "border-s border-border/60",
      )}
    >
      <span className="text-2xl font-semibold leading-none tracking-tight tabular-nums text-foreground">
        {value}
      </span>
      <div className="flex flex-col gap-0.5">
        <span className="flex items-center gap-1.5 text-xs font-medium text-foreground">
          <Icon className="size-3.5 shrink-0" style={{ color }} />
          {label}
        </span>
        <span className="text-[0.6875rem] leading-tight text-muted-foreground">{hint}</span>
      </div>
    </div>
  );
}
