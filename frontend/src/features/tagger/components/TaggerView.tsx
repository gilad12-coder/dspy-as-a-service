"use client";

import { useEffect, useState } from "react";
import { List } from "@/shared/ui/icons";
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
import { TaggerResultsSummary } from "./TaggerResultsSummary";
import { TaggerMoveToDatasets } from "./TaggerMoveToDatasets";
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

  // A finished assist run swaps the row view for the results overview once —
  // the completion accounting rides above that table now, so landing there is
  // the "you're done" moment (no separate summary screen). Detected as a
  // render-time phase change so the switch happens before paint, not after.
  const [seenPhase, setSeenPhase] = useState(tagger.phase);
  if (tagger.phase !== seenPhase) {
    setSeenPhase(tagger.phase);
    if (tagger.phase === "complete") setFocusRow(false);
  }

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
          onRestart={tagger.restartInterview}
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

  const assistActive = tagger.assist !== null && tagger.phase === "review";

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
      suggestionsPending={tagger.phase === "review" && tagger.roundPredicting}
      currentIndex={tagger.currentIndex}
      taggedCount={tagger.frameTaggedCount}
      reviewProgress={reviewProgress}
      celebrateCompletion={!tagger.assist}
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
    // A completed assist run browses the same overview as a fully-labeled
    // manual session — the results table — so the two flows converge here.
    const browsing = tagger.phase === "annotating" || tagger.phase === "complete";
    if (browsing && allLabeled && !focusRow) {
      return (
        <PageContainer full>
          {backBar}
          {/* The finished session's doorway into Datasets: only the owner of a
              persisted session may move it, and moving deletes the session so no
              completed labeling run is left stranded outside the library. */}
          {(!initialSession || initialSession.role === "owner") && tagger.sessionId && (
            <div className="mb-3">
              <TaggerMoveToDatasets
                sessionId={tagger.sessionId}
                config={tagger.config}
                data={tagger.data}
                columns={tagger.columns}
                annotations={tagger.annotations}
                provenance={tagger.assist?.provenance}
              />
            </div>
          )}
          {/* The finished run's accounting — who labeled what, the credit
              cost, and the one-click flagged pass — rides above the table
              instead of on a separate summary screen. */}
          {tagger.assist && (
            <div className="mb-3">
              <TaggerResultsSummary
                assist={tagger.assist}
                annotations={tagger.annotations}
                onFlaggedPass={tagger.startFlaggedPass}
              />
            </div>
          )}
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
        {browsing && allLabeled && (
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
        <div className="min-w-0 flex-1">{annotation}</div>
        <div className="lg:max-h-[calc(100dvh-var(--header-height,53px)-4rem)] lg:sticky lg:top-4">
          {/* Between rounds the gate question takes the rail's slot instead of
              replacing the whole screen — the annotation surface stays mounted
              and the app keeps the exact same viewport geometry. */}
          {tagger.openRound ? (
            <TaggerAssistRail
              config={tagger.config}
              assist={tagger.assist}
              annotations={tagger.annotations}
              frameData={tagger.frameData}
              currentIndex={tagger.currentIndex}
              openRound={tagger.openRound}
              roundPredicting={tagger.roundPredicting}
              onAccept={tagger.acceptPrediction}
              onGoTo={tagger.goTo}
              onFinishRound={tagger.finishRound}
              predictError={tagger.assistError === "predict"}
            />
          ) : (
            <TaggerReviewGate
              config={tagger.config}
              assist={tagger.assist}
              remainingCount={tagger.data.length - tagger.taggedCount}
              estimate={tagger.estimate}
              roundLoading={tagger.roundLoading || tagger.contractStarting}
              assistError={tagger.assistError}
              onStartRound={() => void tagger.startReviewRound()}
              onStartAutotag={() => void tagger.startAutotag()}
              onFetchEstimate={() => void tagger.fetchEstimate()}
            />
          )}
        </div>
      </div>
    </PageContainer>
  );
}
