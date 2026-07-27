"use client";

import { useEffect, useState } from "react";
import { ArrowRight, List } from "lucide-react";
import type { TaggerSessionDetail } from "@/shared/lib/api";
import { Button } from "@/shared/ui/primitives/button";
import { DataHubTabs } from "@/shared/ui/data-hub-tabs";
import { PageContainer } from "@/shared/layout/page-container";
import { msg } from "@/shared/lib/messages";
import { useTagger } from "../hooks/use-tagger";
import { TaggerResultsTable } from "./TaggerResultsTable";
import { TaggerBackLink } from "./TaggerBackLink";
import { TaggerSetup } from "./TaggerSetup";
import { TaggerAnnotation } from "./TaggerAnnotation";
import { TaggerInterview } from "./TaggerInterview";
import { TaggerAssistRail } from "./TaggerAssistRail";
import { TaggerReviewGate } from "./TaggerReviewGate";
import { TaggerAutotagLive } from "./TaggerAutotagLive";
import { TaggerAutotagProgress } from "./TaggerAutotagProgress";
import { TaggerComplete } from "./TaggerComplete";
import { TaggingSessionsPanel } from "./TaggingSessionsPanel";

export function TaggerView({ initialSession }: { initialSession?: TaggerSessionDetail | null }) {
  const [startingNew, setStartingNew] = useState(false);
  const tagger = useTagger(initialSession);
  const allLabeled = tagger.data.length > 0 && tagger.taggedCount >= tagger.data.length;
  const readOnlyViewer = initialSession?.role === "viewer";
  // A session that arrives fully labeled opens on the results table; one still
  // being labeled stays in the row view even as the last label lands (no
  // surprise scene change mid-flip) — the overview is one click away instead.
  // Shared-in viewers always land on the table: browsing is their whole job.
  const [focusRow, setFocusRow] = useState(() => !allLabeled && !readOnlyViewer);

  // "Tag this dataset" deep links (/tagger?dataset=…) skip the session chooser
  // straight into setup, which loads the referenced dataset itself. Applied in
  // an effect (not the state initializer) so SSR and first client paint agree.
  useEffect(() => {
    if (new URLSearchParams(window.location.search).has("dataset")) setStartingNew(true);
  }, []);

  if (!initialSession && !startingNew) {
    // The shell leaves /tagger unwrapped for the annotation surfaces; the
    // chooser is a list page, so it renders the same capped PageContainer the
    // shell gives /datasets — identical geometry, no hop between hub tabs.
    return (
      <PageContainer>
        <DataHubTabs active="sessions" />
        <TaggingSessionsPanel onStartNew={() => setStartingNew(true)} />
      </PageContainer>
    );
  }

  // Sessions saved at /tagger/[id] navigate back as a plain link; a session
  // still living on /tagger (the pre-persist wizard flow) instead resets local
  // state so the panel reappears without a navigation.
  const backBar = (
    <div className="mb-3">
      <TaggerBackLink
        onExit={
          initialSession
            ? undefined
            : () => {
                tagger.backToSetup();
                setStartingNew(false);
              }
        }
      />
    </div>
  );

  if (tagger.phase === "setup") {
    return (
      <PageContainer full>
        {backBar}
        <TaggerSetup onStart={tagger.startAnnotating} />
      </PageContainer>
    );
  }

  if (!tagger.config) return null;

  // Shared-in viewers browse read-only: the results table by default, with a
  // row-by-row view whose answer controls render disabled. The assist surfaces
  // are never mounted, and the hook never buffers autosaves for a viewer (the
  // PUT would be rejected below editor).
  if (readOnlyViewer) {
    if (!focusRow) {
      return (
        <PageContainer full>
          {backBar}
          <TaggerResultsTable
            config={tagger.config}
            data={tagger.data}
            annotations={tagger.annotations}
            assist={tagger.assist}
            onOpenRow={(index) => {
              tagger.goTo(index);
              setFocusRow(true);
            }}
          />
        </PageContainer>
      );
    }
    return (
      <PageContainer full>
        {backBar}
        <div className="mb-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setFocusRow(false)}
            className="gap-1.5"
          >
            <List className="size-3.5" />
            {msg("tagger.results.back")}
          </Button>
        </div>
        <TaggerAnnotation
          config={tagger.config}
          data={tagger.frameData}
          columns={tagger.columns}
          annotations={tagger.annotations}
          provenance={tagger.assist?.provenance}
          currentIndex={tagger.currentIndex}
          taggedCount={tagger.frameTaggedCount}
          readOnly
          onNavigate={tagger.navigate}
          onGoTo={tagger.goTo}
          onJumpUntagged={tagger.jumpToUntagged}
          onToggleBinary={() => undefined}
          onToggleCategory={() => undefined}
          onSetFreetext={() => undefined}
          onBack={tagger.backToSetup}
        />
      </PageContainer>
    );
  }

  if (tagger.phase === "interview" && tagger.assist) {
    return (
      <PageContainer full>
        {backBar}
        <TaggerInterview
          config={tagger.config}
          assist={tagger.assist}
          busy={tagger.interviewBusy}
          streamText={tagger.interviewStreamText}
          thinking={tagger.interviewThinking}
          options={tagger.interviewOptions}
          pending={tagger.interviewPending}
          error={tagger.assistError}
          rowCount={tagger.data.length}
          estimate={tagger.estimate}
          onFetchEstimate={() => void tagger.fetchEstimate()}
          onSetModel={tagger.setAssistModel}
          onSetInterviewModel={tagger.setInterviewModel}
          onSetInterviewEffort={tagger.setInterviewEffort}
          onSend={(content) => void tagger.sendInterviewMessage(content)}
          onEditResend={(index, content) => void tagger.sendInterviewMessage(content, index)}
          onStop={tagger.stopInterview}
          onRetry={() => void tagger.sendInterviewMessage(null)}
          onSkip={tagger.skipInterview}
          onConfirmRubric={tagger.confirmRubric}
        />
      </PageContainer>
    );
  }

  if (tagger.phase === "autotagging" && tagger.assist) {
    // The healthy run is a live walkthrough of rows being tagged; the status
    // card remains for the states that need a decision (interrupted job,
    // failure, cancellation) and their resume/browse recovery paths.
    const autotag = tagger.autotagStatus;
    const needsRecovery =
      autotag !== null &&
      ((autotag.status === "running" && !autotag.live) ||
        autotag.status === "failed" ||
        autotag.status === "canceled");
    return (
      <PageContainer full>
        {backBar}
        {needsRecovery ? (
          <TaggerAutotagProgress
            status={autotag}
            onCancel={() => void tagger.cancelAutotag()}
            onResume={() => void tagger.startAutotag()}
            onBrowse={() => {
              setFocusRow(false);
              tagger.browseAll();
            }}
          />
        ) : (
          <TaggerAutotagLive
            config={tagger.config}
            data={tagger.data}
            annotations={tagger.annotations}
            status={autotag}
          />
        )}
      </PageContainer>
    );
  }

  if (tagger.phase === "complete" && tagger.assist) {
    return (
      <PageContainer full>
        {backBar}
        <TaggerComplete
          assist={tagger.assist}
          annotations={tagger.annotations}
          rowCount={tagger.data.length}
          onFlaggedPass={tagger.startFlaggedPass}
          onBrowse={() => {
            setFocusRow(false);
            tagger.browseAll();
          }}
        />
      </PageContainer>
    );
  }

  if (tagger.phase === "review" && tagger.assist && !tagger.openRound) {
    return (
      <PageContainer full>
        {backBar}
        <TaggerReviewGate
          config={tagger.config}
          assist={tagger.assist}
          annotations={tagger.annotations}
          remainingCount={tagger.data.length - tagger.taggedCount}
          estimate={tagger.estimate}
          roundLoading={tagger.roundLoading || tagger.contractStarting}
          assistError={tagger.assistError}
          onStartRound={() => void tagger.startReviewRound()}
          onStartAutotag={() => void tagger.startAutotag()}
          onFetchEstimate={() => void tagger.fetchEstimate()}
        />
      </PageContainer>
    );
  }

  const assistActive =
    tagger.assist !== null && (tagger.phase === "calibration" || tagger.phase === "review");

  // During a round the header bar tracks the human's audit, not the AI's
  // pre-labels — the same decided/total the rail shows, one number per screen.
  const openRound = tagger.openRound;
  const reviewProgress =
    tagger.phase === "review" && openRound
      ? {
          done: openRound.rowIds.filter((id) => openRound.decided[id] !== undefined).length,
          total: openRound.rowIds.length,
        }
      : undefined;

  const annotation = (
    <TaggerAnnotation
      config={tagger.config}
      data={tagger.frameData}
      columns={tagger.columns}
      annotations={tagger.annotations}
      provenance={tagger.assist?.provenance}
      suggestions={tagger.phase === "review" ? tagger.assist?.predictions : undefined}
      currentIndex={tagger.currentIndex}
      taggedCount={tagger.frameTaggedCount}
      reviewProgress={reviewProgress}
      onNavigate={tagger.navigate}
      onGoTo={tagger.goTo}
      onJumpUntagged={tagger.jumpToUntagged}
      onToggleBinary={assistActive ? tagger.assistToggleBinary : tagger.toggleBinary}
      onToggleCategory={assistActive ? tagger.assistToggleCategory : tagger.toggleCategory}
      onSetFreetext={assistActive ? tagger.assistSetFreetext : tagger.setFreetext}
      onBack={tagger.backToSetup}
    />
  );

  if (!assistActive || !tagger.assist) {
    if (tagger.phase === "annotating" && allLabeled && !focusRow) {
      return (
        <PageContainer full>
          {backBar}
          <TaggerResultsTable
            config={tagger.config}
            data={tagger.data}
            annotations={tagger.annotations}
            assist={tagger.assist}
            onOpenRow={(index) => {
              tagger.goTo(index);
              setFocusRow(true);
            }}
          />
        </PageContainer>
      );
    }
    return (
      <PageContainer full>
        {backBar}
        {tagger.phase === "annotating" && allLabeled && (
          <div className="mb-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setFocusRow(false)}
              className="gap-1.5"
            >
              <List className="size-3.5" />
              {msg("tagger.results.back")}
            </Button>
          </div>
        )}
        {annotation}
      </PageContainer>
    );
  }

  return (
    <PageContainer full>
      {backBar}
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
            predictError={tagger.assistError === "predict"}
          />
        </div>
      </div>
    </PageContainer>
  );
}
