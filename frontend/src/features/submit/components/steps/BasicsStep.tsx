"use client";

import { useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/shared/ui/primitives/card";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";
import { useUserPrefs } from "@/features/settings";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

export function BasicsStep({ w }: { w: SubmitWizardContext }) {
  const {
    jobName,
    setJobName,
    jobDescription,
    setJobDescription,
    jobType,
    setOptimizationType,
    isPrivate,
    setIsPrivate,
  } = w;
  const { prefs } = useUserPrefs();
  const advancedMode = prefs.advancedMode;

  useEffect(() => {
    if (!advancedMode && jobType !== "run") setOptimizationType("run");
  }, [advancedMode, jobType, setOptimizationType]);

  return (
    <Card
      className="border-border/50 bg-card/80 backdrop-blur-xl shadow-lg"
      data-tutorial="wizard-step-1"
    >
      <CardHeader>
        <CardTitle className="text-lg">
          {msg("auto.features.submit.components.steps.basicsstep.1")}
        </CardTitle>
        <CardDescription>
          {msg("auto.features.submit.components.steps.basicsstep.2")}
          {TERMS.optimization}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>
            {msg("auto.features.submit.components.steps.basicsstep.3")}
            {TERMS.optimization}
          </Label>
          <Input
            placeholder={msg("auto.features.submit.components.steps.basicsstep.literal.1")}
            value={jobName}
            onChange={(e) => setJobName(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>{msg("auto.features.submit.components.steps.basicsstep.4")}</Label>
            <span
              className={cn(
                "text-[0.625rem] tabular-nums transition-colors",
                jobDescription.length > 280
                  ? "text-destructive font-medium"
                  : "text-muted-foreground/50",
              )}
            >
              {jobDescription.length}
              {msg("auto.features.submit.components.steps.basicsstep.5")}
            </span>
          </div>
          <textarea
            data-tutorial="job-description"
            value={jobDescription}
            onChange={(e) => {
              if (e.target.value.length <= 280) setJobDescription(e.target.value);
            }}
            placeholder={formatMsg("auto.features.submit.components.steps.basicsstep.template.1", {
              p1: TERMS.optimization,
            })}
            dir="auto"
            rows={4}
            className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:border-ring disabled:cursor-not-allowed disabled:opacity-50 resize-none"
          />
        </div>
        <div className="space-y-3">
          <Label>{msg("submit.basics.privacy.label")}</Label>
          <div className="relative inline-flex w-full rounded-lg bg-muted p-1 gap-1">
            <div
              className="absolute top-1 bottom-1 w-[calc(50%-6px)] rounded-md bg-background shadow-sm transition-[inset-inline-start] duration-100 ease-out"
              style={{ insetInlineStart: isPrivate ? 4 : "calc(50% + 2px)" }}
            />
            {(
              [
                [
                  true,
                  msg("submit.basics.privacy.private"),
                  msg("submit.basics.privacy.private_desc"),
                ],
                [
                  false,
                  msg("submit.basics.privacy.public"),
                  msg("submit.basics.privacy.public_desc"),
                ],
              ] as const
            ).map(([val, label, desc]) => (
              <button
                key={String(val)}
                type="button"
                onClick={() => setIsPrivate(val)}
                className={cn(
                  "relative z-10 flex-1 rounded-md px-4 py-2.5 cursor-pointer text-center transition-colors duration-200",
                  isPrivate === val
                    ? "text-foreground"
                    : "text-foreground/60 hover:text-foreground",
                )}
              >
                <span className="text-sm font-medium">{label}</span>
                <span
                  className={cn(
                    "block text-[0.6875rem] mt-0.5 transition-colors duration-200",
                    isPrivate === val ? "text-muted-foreground" : "text-foreground/40",
                  )}
                >
                  {desc}
                </span>
              </button>
            ))}
          </div>
        </div>
        {advancedMode && (
          <>
            <Separator />
            <div className="space-y-3">
              <Label>
                {msg("auto.features.submit.components.steps.basicsstep.6")}
                {TERMS.optimization}
              </Label>
              <div className="relative inline-flex w-full rounded-lg bg-muted p-1 gap-1">
                <div
                  className="absolute top-1 bottom-1 w-[calc(50%-6px)] rounded-md bg-background shadow-sm transition-[inset-inline-start] duration-100 ease-out"
                  style={{ insetInlineStart: jobType === "run" ? 4 : "calc(50% + 2px)" }}
                />
                {(
                  [
                    [
                      "run",
                      TERMS.optimizationTypeRun,
                      formatMsg("auto.features.submit.components.steps.basicsstep.template.2", {
                        p1: TERMS.optimization,
                        p2: TERMS.model,
                      }),
                    ],
                    [
                      "grid_search",
                      TERMS.optimizationTypeGrid,
                      formatMsg("auto.features.submit.components.steps.basicsstep.template.3", {
                        p1: TERMS.optimizationTypeGrid,
                      }),
                    ],
                  ] as const
                ).map(([val, label, desc]) => (
                  <button
                    key={val}
                    type="button"
                    onClick={() => setOptimizationType(val)}
                    className={cn(
                      "relative z-10 flex-1 rounded-md px-4 py-2.5 cursor-pointer text-center transition-colors duration-200",
                      jobType === val
                        ? "text-foreground"
                        : "text-foreground/60 hover:text-foreground",
                    )}
                  >
                    <span className="text-sm font-medium">{label}</span>
                    <span
                      className={cn(
                        "block text-[0.6875rem] mt-0.5 transition-colors duration-200",
                        jobType === val ? "text-muted-foreground" : "text-foreground/40",
                      )}
                    >
                      {desc}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
