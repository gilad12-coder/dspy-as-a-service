"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Code } from "@/shared/ui/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { Skeleton } from "@/shared/ui/skeleton";
import { tip } from "@/shared/lib/tooltips";
import { msg } from "@/shared/lib/messages";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={180} borderRadius={8} />,
});

export function CodeTab({
  signatureCode,
  metricCode,
}: {
  signatureCode: string;
  metricCode: string;
}) {
  const [activeCodeTab, setActiveCodeTab] = useState<string>("signature");
  return (
    <>
      <FadeIn>
        <p className="text-sm text-muted-foreground">
          {msg("auto.features.optimizations.components.codetab.1")}
        </p>
      </FadeIn>
      {(signatureCode || metricCode) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Code className="size-4" />
              <HelpTip text={tip("code.signature_metric")}>
                {msg("auto.features.optimizations.components.codetab.2")}
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs
              defaultValue={signatureCode ? "signature" : "metric"}
              dir="ltr"
              onValueChange={setActiveCodeTab}
            >
              <TabsList className="relative inline-flex h-auto w-full gap-1 rounded-full border border-border/60 bg-muted/50 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]">
                {signatureCode && metricCode && (
                  <div
                    className="absolute top-1 bottom-1 z-0 w-[calc(50%-6px)] rounded-full bg-background shadow-sm transition-[inset-inline-start] duration-200 ease-out"
                    style={{
                      insetInlineStart: activeCodeTab === "signature" ? 4 : "calc(50% + 2px)",
                    }}
                  />
                )}
                {signatureCode && (
                  <TabsTrigger
                    value="signature"
                    className="relative z-10 rounded-full px-4 py-2 text-sm font-semibold cursor-pointer border-none bg-transparent text-foreground/65 shadow-none transition-[color,transform] data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:border-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 gap-1.5"
                  >
                    {msg("auto.features.optimizations.components.codetab.3")}
                  </TabsTrigger>
                )}
                {metricCode && (
                  <TabsTrigger
                    value="metric"
                    className="relative z-10 rounded-full px-4 py-2 text-sm font-semibold cursor-pointer border-none bg-transparent text-foreground/65 shadow-none transition-[color,transform] data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:border-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 gap-1.5"
                  >
                    {msg("auto.features.optimizations.components.codetab.4")}
                  </TabsTrigger>
                )}
              </TabsList>
              {signatureCode && (
                <TabsContent value="signature">
                  <CodeEditor
                    value={signatureCode}
                    onChange={() => {}}
                    height={`${(signatureCode.split("\n").length + 1) * 19.6 + 8}px`}
                    readOnly
                  />
                </TabsContent>
              )}
              {metricCode && (
                <TabsContent value="metric">
                  <CodeEditor
                    value={metricCode}
                    onChange={() => {}}
                    height={`${(metricCode.split("\n").length + 1) * 19.6 + 8}px`}
                    readOnly
                  />
                </TabsContent>
              )}
            </Tabs>
          </CardContent>
        </Card>
      )}

    </>
  );
}
