"use client";

import { ArrowRight } from "lucide-react";
import type { TaggerSessionDetail } from "@/shared/lib/api";
import { Button } from "@/shared/ui/primitives/button";
import { msg } from "@/shared/lib/messages";
import { useTagger } from "../hooks/use-tagger";
import { TaggerSetup } from "./TaggerSetup";
import { TaggerAnnotation } from "./TaggerAnnotation";
import { TaggerInterview } from "./TaggerInterview";
import { TaggerAssistRail } from "./TaggerAssistRail";
import { TaggerReviewGate } from "./TaggerReviewGate";
import { TaggerAutotagProgress } from "./TaggerAutotagProgress";
import { TaggerComplete } from "./TaggerComplete";

export function TaggerView({ initialSession }: { initialSession?: TaggerSessionDetail | null }) {
  const tagger = useTagger(initialSession);

  if (tagger.phase === "setup") {
    return <TaggerSetup onStart={tagger.startAnnotating} />;
  }

  if (!tagger.config) return null;

  if (tagger.phase === "interview" && tagger.assist) {
    return (
      <TaggerInterview
        assist={tagger.assist}
        busy={tagger.interviewBusy}
        quickReplies={tagger.quickReplies}
        error={tagger.assistError}
        onSend={(content) => void tagger.sendInterviewMessage(content)}
        onRetry={() => void tagger.sendInterviewMessage(null)}
        onSkip={() => tagger.confirmRubric(tagger.assist?.rubric ?? [])}
        onConfirmRubric={tagger.confirmRubric}
      />
    );
  }

  if (tagger.phase === "autotagging" && tagger.assist) {
    return (
      <TaggerAutotagProgress
        status={tagger.autotagStatus}
        onCancel={() => void tagger.cancelAutotag()}
        onResume={() => void tagger.startAutotag()}
        onBrowse={tagger.browseAll}
      />
    );
  }

  if (tagger.phase === "complete" && tagger.assist) {
    return (
      <TaggerComplete
        assist={tagger.assist}
        annotations={tagger.annotations}
        rowCount={tagger.data.length}
        onFlaggedPass={tagger.startFlaggedPass}
        onBrowse={tagger.browseAll}
        onDeepOptimize={() => void tagger.startDeepOptimize()}
      />
    );
  }

  if (tagger.phase === "review" && tagger.assist && !tagger.openRound) {
    return (
      <TaggerReviewGate
        config={tagger.config}
        assist={tagger.assist}
        annotations={tagger.annotations}
        remainingCount={tagger.data.length - tagger.taggedCount}
        estimate={tagger.estimate}
        roundLoading={tagger.roundLoading}
        optimizeBusy={tagger.optimizeBusy}
        assistError={tagger.assistError}
        onStartRound={() => void tagger.startReviewRound()}
        onStartAutotag={() => void tagger.startAutotag()}
        onOptimize={() => void tagger.runOptimize()}
        onDeepOptimize={() => void tagger.startDeepOptimize()}
        onFetchEstimate={() => void tagger.fetchEstimate()}
      />
    );
  }

  const assistActive =
    tagger.assist !== null &&
    (tagger.phase === "calibration" || tagger.phase === "review");

  const annotation = (
    <TaggerAnnotation
      config={tagger.config}
      data={tagger.frameData}
      columns={tagger.columns}
      annotations={tagger.annotations}
      provenance={tagger.assist?.provenance}
      currentIndex={tagger.currentIndex}
      taggedCount={tagger.frameTaggedCount}
      onNavigate={tagger.navigate}
      onGoTo={tagger.goTo}
      onJumpUntagged={tagger.jumpToUntagged}
      onToggleBinary={assistActive ? tagger.assistToggleBinary : tagger.toggleBinary}
      onToggleCategory={assistActive ? tagger.assistToggleCategory : tagger.toggleCategory}
      onSetFreetext={assistActive ? tagger.assistSetFreetext : tagger.setFreetext}
      onBack={tagger.backToSetup}
    />
  );

  if (!assistActive || !tagger.assist) return annotation;

  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
      <div className="min-w-0 flex-1">
        {tagger.phase === "calibration" && tagger.calibrationDone && (
          <div className="mb-3 flex items-center justify-between gap-3 rounded-xl border border-primary/30 bg-primary/5 px-4 py-2.5">
            <p className="text-sm text-foreground">{msg("tagger.assist.calibration.done")}</p>
            <Button size="sm" onClick={tagger.finishCalibration} className="gap-1.5 shrink-0">
              {msg("tagger.assist.calibration.continue")}
              <ArrowRight className="size-3.5 rtl:rotate-180" />
            </Button>
          </div>
        )}
        {annotation}
      </div>
      <div className="lg:max-h-[calc(100dvh-var(--header-height,53px)-4rem)] lg:sticky lg:top-4">
        <TaggerAssistRail
          phase={tagger.phase as "calibration" | "review"}
          config={tagger.config}
          assist={tagger.assist}
          annotations={tagger.annotations}
          frameData={tagger.frameData}
          currentIndex={tagger.currentIndex}
          openRound={tagger.openRound}
          onAccept={tagger.acceptPrediction}
          onGoTo={tagger.goTo}
          onFinishRound={tagger.finishRound}
          onRubricChange={tagger.setRubric}
          predictError={tagger.assistError === "predict"}
        />
      </div>
    </div>
  );
}
