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
  NodeArtifact,
  OptimizationStatusResponse,
  OptimizedPredictor,
  PairResult,
  ReactOverlay,
  WorkflowSignatureNodeSpec,
  WorkflowSpec,
} from "@/shared/types/api";
import { displayName, kindLabel } from "@/features/submit/workflow/nodes";
import { tip } from "@/shared/lib/tooltips";
import { msg } from "@/shared/lib/messages";
import { CopyButton } from "./ui-primitives";
import { ExportMenu } from "./ExportMenu";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={180} borderRadius={8} />,
});

// Reuses the detail page's read-only DAG (its own 480px canvas) to frame the
// optimized workflow; heavy enough to load on demand like the code editor.
const WorkflowGraphView = dynamic(
  () => import("./WorkflowGraphView").then((m) => m.WorkflowGraphView),
  { ssr: false, loading: () => <Skeleton height={480} borderRadius={12} /> },
);

// What the run produced, gathered in one place: the export menu (runnable
// ZIP, pickle, prompt JSON, logs CSV) plus the optimized prompt and — for
// react runs — the tuned tool roster. A workflow run instead surfaces the whole
// optimized graph, one card per tuned node. The code tab stays inputs-only
// (signature + metric source); this tab is the outputs.
export function ArtifactTab({
  job,
  activePair,
  optimizedPrompt,
  reactOverlay,
  optimizedModuleSrc,
  optimizedComponentSrcs,
  optimizedNodes,
  workflowSpec,
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
  /** Per-node optimized surface for a workflow run, keyed "n_<node_id>". */
  optimizedNodes?: Record<string, NodeArtifact> | null;
  /** The run's workflow graph, for ordering and labelling the per-node view. */
  workflowSpec?: WorkflowSpec | null;
  isShare?: boolean;
}) {
  const exportPair = activePair ?? job.grid_result?.best_pair ?? null;
  const hasProgram =
    !isShare &&
    !!(
      job.result?.program_artifact?.program_state_json ||
      exportPair?.program_artifact?.program_state_json
    );
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

  // A workflow run surfaces every node's optimized output (prompt/code/tools)
  // as a graph. The scalar prompt/flex/react fields are the first predictor's
  // projection and would duplicate a node, so the per-node view replaces them
  // whenever it carries data.
  const isWorkflowArtifact = !!optimizedNodes && Object.keys(optimizedNodes).length > 0;

  return (
    <div className="space-y-6" data-tutorial="artifact-output">
      <FadeIn>
        <p className="text-sm text-muted-foreground">{msg("optimization.artifact.description")}</p>
      </FadeIn>

      {hasExports && (
        <div className="flex flex-col items-stretch gap-3 rounded-xl border border-primary/30 bg-gradient-to-br from-primary/5 to-primary/10 p-4 shadow-[0_0_20px_rgba(var(--primary),0.06)] sm:flex-row sm:items-center sm:p-5">
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
            programPairIndex={exportPair?.pair_index}
            isShare={isShare}
          />
        </div>
      )}

      {isWorkflowArtifact ? (
        <WorkflowArtifactView
          workflowSpec={workflowSpec ?? null}
          optimizedNodes={optimizedNodes ?? {}}
        />
      ) : (
        <>
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
                <PromptBody predictor={optimizedPrompt} />
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
      )}
    </div>
  );
}

// The optimized prompt as the model sees it — the formatted string with a hover
// copy, then any few-shot demonstrations. Shared by the scalar prompt card and
// each workflow node's card.
function PromptBody({ predictor }: { predictor: OptimizedPredictor }) {
  return (
    <>
      <div className="relative group">
        <pre
          className="text-sm font-mono bg-muted/50 rounded-lg p-4 pe-10 overflow-x-auto whitespace-pre-wrap leading-relaxed"
          dir="ltr"
        >
          {predictor.formatted_prompt}
        </pre>
        <CopyButton
          text={predictor.formatted_prompt}
          className="absolute end-1.5 top-1.5 opacity-100 sm:end-2 sm:top-2 sm:opacity-0 sm:group-hover:opacity-100"
        />
      </div>
      {predictor.demos && predictor.demos.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground mb-2">
            {predictor.demos.length}{" "}
            <HelpTip text={tip("prompt.demonstrations")}>
              {msg("auto.features.optimizations.components.codetab.6")}
            </HelpTip>
          </p>
          <div className="space-y-2">
            {predictor.demos.map((demo, i) => (
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
    </>
  );
}

// Read-only viewer over GEPA-rewritten code, sized to its content up to a cap.
function CodeBody({ source }: { source: string }) {
  return (
    <CodeEditor
      value={source}
      onChange={() => {}}
      height={`${Math.min((source.split("\n").length + 1) * 19.6 + 8, 600)}px`}
      readOnly
    />
  );
}

// True when a node carries any optimized output worth a card.
function nodeHasContent(node: NodeArtifact): boolean {
  return !!(
    node.optimized_prompt ||
    node.optimized_src ||
    (node.react_overlay && Object.keys(node.react_overlay.tool_descriptions).length > 0)
  );
}

// A workflow's optimized artifact IS its graph: the read-only DAG, then one card
// per node the optimizer tuned, ordered to match the graph. Falls back to the
// raw "n_<id>" keys when the workflow spec is unavailable (e.g. a share view
// reached without the workflow payload).
function WorkflowArtifactView({
  workflowSpec,
  optimizedNodes,
}: {
  workflowSpec: WorkflowSpec | null;
  optimizedNodes: Record<string, NodeArtifact>;
}) {
  const cards = useMemo(() => {
    if (workflowSpec) {
      return workflowSpec.nodes.flatMap((node) => {
        if (node.kind !== "signature") return [];
        const artifact = optimizedNodes[`n_${node.id}`];
        if (!artifact || !nodeHasContent(artifact)) return [];
        const signature: WorkflowSignatureNodeSpec = node;
        return [
          {
            id: node.id,
            title: displayName(signature),
            kind: kindLabel(signature) as string | null,
            artifact,
          },
        ];
      });
    }
    return Object.entries(optimizedNodes).flatMap(([key, artifact]) =>
      nodeHasContent(artifact)
        ? [{ id: key, title: key.replace(/^n_/, ""), kind: null as string | null, artifact }]
        : [],
    );
  }, [workflowSpec, optimizedNodes]);

  return (
    <>
      {workflowSpec && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Sparkle className="size-4" />
              {msg("optimization.artifact.workflow_graph")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <WorkflowGraphView spec={workflowSpec} />
          </CardContent>
        </Card>
      )}
      {cards.map((card) => (
        <NodeArtifactCard
          key={card.id}
          title={card.title}
          kind={card.kind}
          artifact={card.artifact}
        />
      ))}
    </>
  );
}

// One tuned node's optimized surface: its prompt, its rewritten code, or its
// react tools, under a header naming the node and its module kind.
function NodeArtifactCard({
  title,
  kind,
  artifact,
}: {
  title: string;
  kind: string | null;
  artifact: NodeArtifact;
}) {
  const Icon = artifact.optimized_src ? Code : Sparkle;
  const overlay = artifact.react_overlay;
  const hasTools = !!overlay && Object.keys(overlay.tool_descriptions).length > 0;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Icon className="size-4" />
          <span dir="auto">{title}</span>
          {kind && (
            <span className="font-mono text-xs font-normal text-muted-foreground" dir="ltr">
              {kind}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {artifact.optimized_prompt && <PromptBody predictor={artifact.optimized_prompt} />}
        {artifact.optimized_src && <CodeBody source={artifact.optimized_src} />}
        {hasTools && overlay && <ReactToolsCarousel overlay={overlay} />}
      </CardContent>
    </Card>
  );
}

// Pages the ReAct overlay's tools one at a time through the shared carousel
// chrome (counter, dots, RTL prev/next, keyboard) instead of a long vertical
// list — the same paging the trajectory drawer uses for tool descriptions.
function ReactToolsCarousel({ overlay }: { overlay: ReactOverlay }) {
  const tools = useMemo(() => Object.keys(overlay.tool_descriptions), [overlay.tool_descriptions]);
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
        <CodeBody source={source} />
      </CardContent>
    </Card>
  );
}
