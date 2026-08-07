"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { Code, Sparkle, Wrench } from "@/shared/ui/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { Skeleton } from "@/shared/ui/skeleton";
import { Carousel, ToolHeader } from "@/features/agent-panel";
import type {
  OptimizationStatusResponse,
  OptimizedPredictor,
  PairResult,
  ReactOverlay,
} from "@/shared/types/api";
import { tip } from "@/shared/lib/tooltips";
import { msg } from "@/shared/lib/messages";
import { CopyButton } from "./ui-primitives";
import { ExportMenu } from "./ExportMenu";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={180} borderRadius={8} />,
});

// What the run produced, gathered in one place: the export menu (runnable
// ZIP, pickle, prompt JSON, logs CSV) plus the optimized prompt and — for
// react runs — the tuned tool roster. The code tab stays inputs-only
// (signature + metric source); this tab is the outputs.
export function ArtifactTab({
  job,
  activePair,
  optimizedPrompt,
  reactOverlay,
  optimizedModuleSrc,
  optimizedComponentSrcs,
  isShare,
}: {
  job: OptimizationStatusResponse;
  activePair?: PairResult | null;
  optimizedPrompt: OptimizedPredictor | null;
  reactOverlay?: ReactOverlay | null;
  /** GEPA-rewritten Flex module source; shown as a read-only code viewer. */
  optimizedModuleSrc?: string | null;
  /** Per-submodule Flex sources (a workflow's flex nodes), one viewer each. */
  optimizedComponentSrcs?: Record<string, string>;
  isShare?: boolean;
}) {
  // The runnable export reconstructs from state JSON + signature_code, which
  // the /program-export endpoint serves for single-run (non-grid) jobs only —
  // and it is an authed endpoint, so the public share view hides it.
  const hasProgram = !isShare && !!job.result?.program_artifact?.program_state_json;
  const pickleBase64 = activePair?.program_artifact?.program_pickle_base64 ?? null;
  const hasPickle =
    !!pickleBase64 ||
    !!job.result?.program_artifact?.program_pickle_base64 ||
    !!job.grid_result?.best_pair?.program_artifact?.program_pickle_base64;
  const hasExports =
    hasProgram ||
    hasPickle ||
    !!optimizedPrompt ||
    !!optimizedModuleSrc ||
    Object.keys(optimizedComponentSrcs ?? {}).length > 0 ||
    (job.logs?.length ?? 0) > 0;

  return (
    <>
      <FadeIn>
        <p className="text-sm text-muted-foreground">{msg("optimization.artifact.description")}</p>
      </FadeIn>

      {hasExports && (
        <div className="flex items-center gap-3 p-5 rounded-xl border border-primary/30 bg-gradient-to-br from-primary/5 to-primary/10 shadow-[0_0_20px_rgba(var(--primary),0.06)]">
          <p className="flex-1 text-sm font-medium">
            {activePair
              ? msg("auto.features.optimizations.components.pairdetailview.2")
              : msg("auto.app.optimizations.id.page.9")}
          </p>
          <ExportMenu
            job={job}
            optimizedPrompt={optimizedPrompt}
            optimizedModuleSrc={optimizedModuleSrc}
            optimizedComponentSrcs={optimizedComponentSrcs}
            pickleBase64={pickleBase64}
            isShare={isShare}
          />
        </div>
      )}

      {optimizedPrompt && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Sparkle className="size-4" />
              <HelpTip text={tip("prompt.optimized")}>
                {msg("auto.features.optimizations.components.codetab.5")}
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative group">
              <pre
                className="text-sm font-mono bg-muted/50 rounded-lg p-4 pe-10 overflow-x-auto whitespace-pre-wrap leading-relaxed"
                dir="ltr"
              >
                {optimizedPrompt.formatted_prompt}
              </pre>
              <CopyButton
                text={optimizedPrompt.formatted_prompt}
                className="absolute top-2 end-2 opacity-0 group-hover:opacity-100"
              />
            </div>
            {optimizedPrompt.demos && optimizedPrompt.demos.length > 0 && (
              <div className="mt-4 pt-4 border-t border-border">
                <p className="text-xs text-muted-foreground mb-2">
                  {optimizedPrompt.demos.length}{" "}
                  <HelpTip text={tip("prompt.demonstrations")}>
                    {msg("auto.features.optimizations.components.codetab.6")}
                  </HelpTip>
                </p>
                <div className="space-y-2">
                  {optimizedPrompt.demos.map((demo, i) => (
                    <div key={i} className="text-xs font-mono bg-muted/50 rounded-lg p-3" dir="ltr">
                      {Object.entries(demo.inputs).map(([k, v]) => (
                        <div key={k}>
                          <span className="text-muted-foreground">{k}:</span> {String(v)}
                        </div>
                      ))}
                      {Object.entries(demo.outputs).map(([k, v]) => (
                        <div key={k}>
                          <span className="text-stone-600">{k}:</span> {String(v)}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {optimizedModuleSrc && <OptimizedCodeCard source={optimizedModuleSrc} />}

      {Object.entries(optimizedComponentSrcs ?? {}).map(([path, source]) => (
        <OptimizedCodeCard key={path} source={source} path={path} />
      ))}

      {reactOverlay && Object.keys(reactOverlay.tool_descriptions).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Wrench className="size-4" />
              <HelpTip text={tip("react.optimized_tools")}>
                {msg("optimizations.react.optimized_tools")}
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ReactToolsCarousel overlay={reactOverlay} />
          </CardContent>
        </Card>
      )}
    </>
  );
}

// Pages the ReAct overlay's tools one at a time through the shared carousel
// chrome (counter, dots, RTL prev/next, keyboard) instead of a long vertical
// list — the same paging the trajectory drawer uses for tool descriptions.
function ReactToolsCarousel({ overlay }: { overlay: ReactOverlay }) {
  const tools = useMemo(
    () => Object.keys(overlay.tool_descriptions),
    [overlay.tool_descriptions],
  );
  return (
    <Carousel
      items={tools}
      itemKey={(name) => name}
      renderItem={(name) => (
        <ReactToolSlide
          name={name}
          desc={overlay.tool_descriptions[name] ?? ""}
          renamed={overlay.tool_names?.[name]}
          argDescs={overlay.tool_arg_descriptions?.[name]}
          severity={overlay.tool_severities?.[name]}
        />
      )}
      ariaLabel={msg("optimizations.react.optimized_tools")}
      fluid
      className="w-full"
    />
  );
}

// One tool's slide. Wears the shared ToolHeader chrome — severity-tinted icon,
// friendly title, severity label — so it reads identically to the agent tour and
// the trajectory drawer's allowed_tools carousel; any optimized agent's tools get
// the same treatment (uncatalogued ones fall back to a wrench + prettified name).
// Severity comes from the run's own tool metadata (overlay.tool_severities,
// captured from the source MCP's annotations) and is never fabricated. The
// optimized description and per-argument descriptions sit below it, plus the
// GEPA-renamed name when the optimizer changed it.
function ReactToolSlide({
  name,
  desc,
  renamed,
  argDescs,
  severity,
}: {
  name: string;
  desc: string;
  renamed?: string;
  argDescs?: Record<string, string>;
  severity?: string;
}) {
  const optimizedName = renamed && renamed !== name ? renamed : null;
  return (
    <div className="p-3.5">
      <ToolHeader toolKey={name} severity={severity} className="mb-2.5" />
      {optimizedName ? (
        <p className="-mt-1.5 mb-2 font-mono text-[0.625rem] text-muted-foreground/70" dir="ltr">
          {`↳ ${optimizedName}`}
        </p>
      ) : null}
      {desc ? (
        <p className="text-[0.75rem] leading-relaxed text-foreground/75" dir="auto">
          {desc}
        </p>
      ) : null}
      {argDescs && Object.keys(argDescs).length > 0 && (
        <div className="mt-2 space-y-0.5 border-t border-border/40 pt-2">
          {Object.entries(argDescs).map(([arg, argDesc]) => (
            <div key={arg} className="text-[0.6875rem] text-muted-foreground" dir="auto">
              <span className="font-mono text-foreground/70" dir="ltr">
                {arg}
              </span>
              {argDesc ? ` — ${argDesc}` : ""}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// One read-only viewer over GEPA-rewritten code. `path` names the Flex
// submodule it came from and is omitted when the program is itself a Flex.
function OptimizedCodeCard({ source, path }: { source: string; path?: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Code className="size-4" />
          <HelpTip text={tip("flex.optimized_code")}>
            {msg("optimizations.flex.optimized_code")}
          </HelpTip>
          {path && (
            <span className="font-mono text-xs font-normal text-muted-foreground" dir="ltr">
              {path}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <CodeEditor
          value={source}
          onChange={() => {}}
          height={`${Math.min((source.split("\n").length + 1) * 19.6 + 8, 600)}px`}
          readOnly
        />
      </CardContent>
    </Card>
  );
}
