"use client";

import { useEffect } from "react";
import { ArrowRight, CircleNotch, Sparkle } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { AutotagEstimate } from "../hooks/use-tagger";
import type { AssistState, TaggerConfig } from "../lib/types";
import { agreementGate, gateUnlocked } from "../lib/assist";

interface Props {
  config: TaggerConfig;
  assist: AssistState;
  remainingCount: number;
  estimate: AutotagEstimate | null;
  roundLoading: boolean;
  assistError: string | null;
  onStartRound: () => void;
  onStartAutotag: () => void;
  onFetchEstimate: () => void;
}

/**
 * The between-rounds card of the review phase, rendered in the assist rail's
 * slot so the tagging surface never changes geometry: shows what the last
 * round proved, whether the agreement gate is open, and — only once it is
 * earned (or in autopilot, where the user chose full autonomy) — the
 * cost-labeled "tag the rest" commitment. Cost before commitment, always.
 */
export function TaggerReviewGate({
  config,
  assist,
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

  useEffect(() => {
    if (unlocked) onFetchEstimate();
  }, [unlocked]);

  if (roundLoading) {
    return (
      <Centered>
        <CircleNotch className="size-6 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">{msg("tagger.assist.gate.preparing")}</p>
      </Centered>
    );
  }

  return (
    <div className="flex w-full flex-col gap-4 lg:w-[300px]">
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
                : msg("tagger.assist.gate.calibration_subtitle_blind")}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {assistError && (
            <p className="text-sm text-destructive">{msg("tagger.assist.gate.error")}</p>
          )}

          {unlocked ? (
            <>
              <Button onClick={onStartAutotag} size="lg" className="w-full gap-2">
                <Sparkle className="size-4" />
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
