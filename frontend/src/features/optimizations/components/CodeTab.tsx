"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Code } from "@/shared/ui/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { Skeleton } from "@/shared/ui/skeleton";
import { tip } from "@/shared/lib/tooltips";
import { msg } from "@/shared/lib/messages";
import type { WorkflowSpec } from "@/shared/types/api";

import { compileWorkflowToCode } from "../lib/workflow-code";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={180} borderRadius={8} />,
});

const WorkflowGraphView = dynamic(
  () => import("./WorkflowGraphView").then((m) => m.WorkflowGraphView),
  { ssr: false, loading: () => <Skeleton height={480} borderRadius={8} /> },
);

const TRIGGER_CLASS =
  "relative z-10 rounded-full px-4 py-2 text-sm font-semibold cursor-pointer border-none bg-transparent text-foreground/65 shadow-none transition-[color,transform] data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:border-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 gap-1.5";

const editorHeight = (code: string): string => `${(code.split("\n").length + 1) * 19.6 + 8}px`;

export function CodeTab({
  signatureCode,
  metricCode,
  workflowSpec = null,
}: {
  signatureCode: string;
  metricCode: string;
  workflowSpec?: WorkflowSpec | null;
}) {
  const workflowCode = useMemo(
    () => (workflowSpec ? compileWorkflowToCode(workflowSpec) : null),
    [workflowSpec],
  );

  const tabs: Array<{ value: string; label: string }> = workflowSpec
    ? [
        { value: "code", label: msg("optimization.code.tab_program") },
        ...(metricCode
          ? [{ value: "metric", label: msg("auto.features.optimizations.components.codetab.4") }]
          : []),
        { value: "workflow", label: msg("optimization.code.tab_workflow") },
      ]
    : [
        ...(signatureCode
          ? [{ value: "signature", label: msg("auto.features.optimizations.components.codetab.3") }]
          : []),
        ...(metricCode
          ? [{ value: "metric", label: msg("auto.features.optimizations.components.codetab.4") }]
          : []),
      ];

  const [activeCodeTab, setActiveCodeTab] = useState<string>(
    workflowSpec ? "code" : signatureCode ? "signature" : "metric",
  );
  const activeIndex = Math.max(0, tabs.findIndex((t) => t.value === activeCodeTab));
  const share = 100 / Math.max(1, tabs.length);

  return (
    <>
      <FadeIn>
        <p className="text-sm text-muted-foreground">
          {workflowSpec
            ? msg("optimization.code.workflow_intro")
            : msg("auto.features.optimizations.components.codetab.1")}
        </p>
      </FadeIn>
      {(signatureCode || metricCode || workflowSpec) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Code className="size-4" />
              <HelpTip text={workflowSpec ? tip("code.workflow") : tip("code.signature_metric")}>
                {msg("auto.features.optimizations.components.codetab.2")}
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeCodeTab} dir="ltr" onValueChange={setActiveCodeTab}>
              <TabsList className="relative inline-flex h-auto w-full gap-1 rounded-full border border-border/60 bg-muted/50 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]">
                {tabs.length > 1 && (
                  <div
                    className="absolute top-1 bottom-1 z-0 rounded-full bg-background shadow-sm transition-[inset-inline-start] duration-200 ease-out"
                    style={{
                      width: `calc(${share}% - 6px)`,
                      insetInlineStart:
                        activeIndex === 0 ? 4 : `calc(${share * activeIndex}% + 2px)`,
                    }}
                  />
                )}
                {tabs.map((t) => (
                  <TabsTrigger key={t.value} value={t.value} className={TRIGGER_CLASS}>
                    {t.label}
                  </TabsTrigger>
                ))}
              </TabsList>
              {workflowCode && (
                <TabsContent value="code">
                  <CodeEditor
                    value={workflowCode}
                    onChange={() => {}}
                    height={editorHeight(workflowCode)}
                    readOnly
                  />
                </TabsContent>
              )}
              {!workflowSpec && signatureCode && (
                <TabsContent value="signature">
                  <CodeEditor
                    value={signatureCode}
                    onChange={() => {}}
                    height={editorHeight(signatureCode)}
                    readOnly
                  />
                </TabsContent>
              )}
              {metricCode && (
                <TabsContent value="metric">
                  <CodeEditor
                    value={metricCode}
                    onChange={() => {}}
                    height={editorHeight(metricCode)}
                    readOnly
                  />
                </TabsContent>
              )}
              {workflowSpec && (
                <TabsContent value="workflow">
                  <WorkflowGraphView spec={workflowSpec} />
                </TabsContent>
              )}
            </Tabs>
          </CardContent>
        </Card>
      )}

    </>
  );
}
