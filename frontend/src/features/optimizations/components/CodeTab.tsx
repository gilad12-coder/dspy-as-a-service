"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Code } from "@/shared/ui/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import {
  SLIDING_PILL_TABS_INDICATOR_CLASS,
  SLIDING_PILL_TABS_LIST_CLASS,
  SLIDING_PILL_TABS_TRIGGER_CLASS,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/shared/ui/primitives/tabs";
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
  const activeIndex = Math.max(
    0,
    tabs.findIndex((t) => t.value === activeCodeTab),
  );
  const share = 100 / Math.max(1, tabs.length);

  return (
    <div className="space-y-6" data-tutorial="code-output">
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
              <TabsList className={SLIDING_PILL_TABS_LIST_CLASS}>
                {tabs.length > 1 && (
                  <div
                    className={SLIDING_PILL_TABS_INDICATOR_CLASS}
                    style={{
                      width: `calc(${share}% - 6px)`,
                      insetInlineStart:
                        activeIndex === 0 ? 4 : `calc(${share * activeIndex}% + 2px)`,
                    }}
                  />
                )}
                {tabs.map((t) => (
                  <TabsTrigger
                    key={t.value}
                    value={t.value}
                    className={SLIDING_PILL_TABS_TRIGGER_CLASS}
                  >
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
    </div>
  );
}
