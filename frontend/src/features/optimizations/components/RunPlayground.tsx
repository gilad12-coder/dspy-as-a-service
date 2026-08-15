"use client";

import { Trash } from "@/shared/ui/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { Button } from "@/shared/ui/primitives/button";
import { Separator } from "@/shared/ui/primitives/separator";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { ServeChat, type ServeChatProps } from "./ServeChat";
import { ServeCodeSnippets } from "./ServeCodeSnippets";
import { CopyButton } from "./ui-primitives";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { msg } from "@/shared/lib/messages";
import { tip } from "@/shared/lib/tooltips";

export interface RunPlaygroundProps extends ServeChatProps {
  optimizationId: string;
  pairIndex?: number;
  onClearHistory: () => void;
  /** Public share view: hide the owner-gated /serve API-URL + code snippets. */
  isShare?: boolean;
}

export function RunPlayground({
  serveInfo,
  runHistory,
  setRunHistory,
  streamingRun,
  serveLoading,
  serveError,
  setServeError,
  textareaRefs,
  chatScrollRef,
  handleServe,
  handleStopServe,
  optimizationId,
  pairIndex,
  onClearHistory,
  isShare = false,
}: RunPlaygroundProps) {
  const apiBase = getRuntimeEnv().apiUrl;
  const servePath =
    pairIndex != null ? `/serve/${optimizationId}/pair/${pairIndex}` : `/serve/${optimizationId}`;
  const serveUrl = `${apiBase}${servePath}`;
  const description =
    pairIndex != null
      ? msg("auto.features.optimizations.components.pairdetailview.16")
      : msg("auto.app.optimizations.id.page.20");
  const clearTooltip =
    pairIndex != null
      ? msg("auto.features.optimizations.components.pairdetailview.17")
      : msg("auto.app.optimizations.id.page.21");
  const clearLabel =
    pairIndex != null
      ? msg("auto.features.optimizations.components.pairdetailview.literal.3")
      : msg("auto.app.optimizations.id.page.literal.6");

  return (
    <>
      <FadeIn>
        <div className="flex items-start justify-between gap-2 border-b border-border/60 pb-3">
          <p className="min-w-0 text-sm text-muted-foreground">{description}</p>
          {runHistory.length > 0 && (
            <TooltipButton tooltip={clearTooltip}>
              <Button
                variant="ghost"
                size="icon"
                className="size-[44px] sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                onClick={onClearHistory}
                aria-label={clearLabel}
              >
                <Trash className="size-4" />
              </Button>
            </TooltipButton>
          )}
        </div>
      </FadeIn>
      <div data-tutorial="serve-playground">
        <ServeChat
          serveInfo={serveInfo}
          runHistory={runHistory}
          setRunHistory={setRunHistory}
          streamingRun={streamingRun}
          serveLoading={serveLoading}
          serveError={serveError}
          setServeError={setServeError}
          textareaRefs={textareaRefs}
          chatScrollRef={chatScrollRef}
          handleServe={handleServe}
          handleStopServe={handleStopServe}
        />
      </div>

      {!isShare && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              <HelpTip text={tip("serve.section_run")}>
                {msg("auto.app.optimizations.id.page.22")}
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1.5">
              <p className="text-[0.625rem] text-muted-foreground uppercase tracking-wider">
                <HelpTip text={tip("serve.api_url_run")}>
                  {msg("auto.app.optimizations.id.page.23")}
                </HelpTip>
              </p>
              <div className="group relative rounded-lg bg-muted/40 p-2.5 pe-11 sm:pe-8" dir="ltr">
                <code className="text-xs font-mono break-all">{serveUrl}</code>
                <CopyButton
                  text={serveUrl}
                  className="absolute end-1 top-1 opacity-100 sm:end-1.5 sm:top-1.5 sm:opacity-0 sm:group-hover:opacity-100"
                />
              </div>
            </div>

            <Separator />

            <div className="space-y-2">
              <p className="text-[0.625rem] text-muted-foreground uppercase tracking-wider">
                <HelpTip text={tip("serve.integration_code")}>
                  {msg("auto.app.optimizations.id.page.26")}
                </HelpTip>
              </p>
              <ServeCodeSnippets
                serveInfo={serveInfo}
                optimizationId={optimizationId}
                pairIndex={pairIndex}
              />
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
}
