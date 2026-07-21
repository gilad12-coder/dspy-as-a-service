"use client";

import { useEffect } from "react";
import { ArrowRight, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { AutotagEstimate } from "../hooks/use-tagger";
import type { Annotation, AssistState, TaggerConfig } from "../lib/types";
import { agreementGate, agreementOver, gateUnlocked } from "../lib/assist";

interface Props {
  config: TaggerConfig;
  assist: AssistState;
  annotations: Record<string, Annotation>;
  remainingCount: number;
  estimate: AutotagEstimate | null;
  roundLoading: boolean;
  assistError: string | null;
  onStartRound: () => void;
  onStartAutotag: () => void;
  onFetchEstimate: () => void;
}

/**
 * The between-rounds interstitial of the review phase: shows what the last
 * round proved, whether the agreement gate is open, and — only once it is
 * earned (or in autopilot, where the user chose full autonomy) — the
 * cost-labeled "tag the rest" commitment. Cost before commitment, always.
 */
export function TaggerReviewGate({
  config,
  assist,
  annotations,
  remainingCount,
  estimate,
  roundLoading,
  assistError,
  onStartRound,
  onStartAutotag,
  onFetchEstimate,
}: Props) {
  const gate = agreementGate(config.mode);
  const closedRounds = assist.rounds.filter((r) => !r.flaggedPass && r.agreement !== undefined);
  const lastRound = closedRounds[closedRounds.length - 1];
  const unlocked = assist.mode === "autopilot" || gateUnlocked(config, assist);
  const calibrationAgreement = agreementOver(
    config.mode,
    assist.calibrationIds,
    annotations,
    assist.predictions,
  );

  useEffect(() => {
    if (unlocked) onFetchEstimate();
  }, [unlocked]);

  if (roundLoading) {
    return (
      <Centered>
        <Loader2 className="size-6 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">{msg("tagger.assist.gate.preparing")}</p>
      </Centered>
    );
  }

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-4 pt-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {assist.mode === "autopilot" && closedRounds.length === 0
              ? msg("tagger.assist.gate.autopilot_title")
              : lastRound
                ? msg("tagger.assist.gate.round_title")
                : msg("tagger.assist.gate.calibration_title")}
          </CardTitle>
          <CardDescription>
            {assist.mode === "autopilot" && closedRounds.length === 0
              ? msg("tagger.assist.gate.autopilot_subtitle")
              : lastRound
                ? formatMsg("tagger.assist.gate.round_subtitle", {
                    agreement: Math.round((lastRound.agreement ?? 0) * 100),
                    gate: Math.round(gate * 100),
                  })
                : calibrationAgreement !== null
                  ? formatMsg("tagger.assist.gate.calibration_subtitle", {
                      agreement: Math.round(calibrationAgreement * 100),
                    })
                  : msg("tagger.assist.gate.calibration_subtitle_blind")}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {closedRounds.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              {closedRounds.map((round, idx) => {
                const pct = Math.round((round.agreement ?? 0) * 100);
                const passed = (round.agreement ?? 0) >= gate;
                return (
                  <span
                    key={idx}
                    className={
                      passed
                        ? "rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium tabular-nums text-primary"
                        : "rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium tabular-nums text-muted-foreground"
                    }
                  >
                    {formatMsg("tagger.assist.gate.round_chip", { n: idx + 1, pct })}
                  </span>
                );
              })}
            </div>
          )}

          {assistError && (
            <p className="text-sm text-destructive">{msg("tagger.assist.gate.error")}</p>
          )}

          {unlocked ? (
            <>
              <Button onClick={onStartAutotag} size="lg" className="w-full gap-2">
                <Sparkles className="size-4" />
                {estimate
                  ? formatMsg("tagger.assist.gate.tag_rest_estimate", {
                      rows: remainingCount,
                      low: estimate.credits_low,
                      high: estimate.credits_high,
                    })
                  : formatMsg("tagger.assist.gate.tag_rest", { rows: remainingCount })}
              </Button>
              <Button variant="outline" onClick={onStartRound} className="w-full gap-2">
                {msg("tagger.assist.gate.another_round")}
                <ArrowRight className="size-4 rtl:rotate-180" />
              </Button>
            </>
          ) : (
            <Button onClick={onStartRound} size="lg" className="w-full gap-2">
              {closedRounds.length === 0
                ? msg("tagger.assist.gate.first_round")
                : msg("tagger.assist.gate.next_round")}
              <ArrowRight className="size-4 rtl:rotate-180" />
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[40vh] flex-col items-center justify-center gap-3">{children}</div>
  );
}
