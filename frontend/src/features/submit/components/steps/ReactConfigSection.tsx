"use client";

import * as React from "react";
import {
  Warning,
  CheckCircle,
  CaretLeft,
  CaretRight,
  CircleNotch,
  ArrowsClockwise,
} from "@/shared/ui/icons";

import { Label } from "@/shared/ui/primitives/label";
import { Input } from "@/shared/ui/primitives/input";
import { HelpTip } from "@/shared/ui/help-tip";
import { cn } from "@/shared/lib/utils";
import { tip } from "@/shared/lib/tooltips";
import { formatMsg, msg } from "@/shared/lib/messages";
import { probeMcp, type McpProbeTool } from "@/shared/lib/api";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

type ProbeStatus =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "ok"; toolCount: number; tools: McpProbeTool[] }
  | { kind: "error"; detail: string };

// How long the URL/auth fields must sit still before the connection check
// fires — long enough to skip mid-typing states, short enough to feel live.
const PROBE_DEBOUNCE_MS = 700;

const NAV_BUTTON_CLASS =
  "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-md border border-border/60 bg-background text-muted-foreground transition-colors hover:border-[#C8A882] hover:text-foreground disabled:pointer-events-none disabled:opacity-40";

export function ReactConfigSection({ w }: { w: SubmitWizardContext }) {
  const { reactConfig, updateReactConfig } = w;
  const [probe, setProbe] = React.useState<ProbeStatus>({ kind: "idle" });
  // Monotonic sequence so a slow response for an old URL can't overwrite the
  // status of the current one.
  const probeSeqRef = React.useRef(0);
  const [toolIndex, setToolIndex] = React.useState(0);

  const runProbe = React.useCallback((url: string, auth: string) => {
    const seq = ++probeSeqRef.current;
    setProbe({ kind: "checking" });
    probeMcp({ mcp_url: url, ...(auth.trim() ? { auth_header: auth.trim() } : {}) })
      .then((res) => {
        if (probeSeqRef.current !== seq) return;
        setProbe(
          res.ok
            ? { kind: "ok", toolCount: res.tool_count, tools: res.tools }
            : { kind: "error", detail: res.error ?? "" },
        );
      })
      .catch((err: unknown) => {
        if (probeSeqRef.current !== seq) return;
        setProbe({ kind: "error", detail: err instanceof Error ? err.message : String(err) });
      });
  }, []);

  React.useEffect(() => {
    const url = reactConfig.mcpUrl.trim();
    if (!url) {
      probeSeqRef.current++;
      setProbe({ kind: "idle" });
      return;
    }
    const t = window.setTimeout(() => runProbe(url, reactConfig.mcpAuthHeader), PROBE_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [reactConfig.mcpUrl, reactConfig.mcpAuthHeader, runProbe]);

  // Each probe result restarts the tool pager from the first tool.
  React.useEffect(() => {
    setToolIndex(0);
  }, [probe]);

  const tools = probe.kind === "ok" ? probe.tools : [];
  const activeTool: McpProbeTool | undefined = tools[toolIndex];

  return (
    <div
      className="space-y-5 rounded-xl border border-border/60 bg-muted/20 p-4"
      data-tutorial="react-config"
    >
      <div className="space-y-1">
        <Label className="font-semibold">{msg("submit.react.section_title")}</Label>
      </div>

      <div className="space-y-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label className="text-xs">
              <HelpTip text={tip("react.mcp_url")}>{msg("submit.react.mcp_url_label")}</HelpTip>
            </Label>
            <Input
              value={reactConfig.mcpUrl}
              dir="ltr"
              placeholder="http://localhost:8000/mcp/"
              className="h-9 font-mono text-xs"
              onChange={(e) => updateReactConfig({ mcpUrl: e.target.value })}
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">
              <HelpTip text={tip("react.auth")}>{msg("submit.react.auth_label")}</HelpTip>
            </Label>
            <Input
              type="password"
              value={reactConfig.mcpAuthHeader}
              dir="ltr"
              autoComplete="off"
              placeholder="Bearer …"
              className="h-9 font-mono text-xs"
              onChange={(e) => updateReactConfig({ mcpAuthHeader: e.target.value })}
            />
          </div>
        </div>

        {probe.kind !== "idle" && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div
                className={cn(
                  "flex min-w-0 items-center gap-1.5 text-[0.6875rem] font-medium",
                  probe.kind === "checking" && "text-muted-foreground",
                  probe.kind === "ok" && "text-[#5A7247]",
                  probe.kind === "error" && "text-[#A3512B]",
                )}
                aria-live="polite"
              >
                {probe.kind === "checking" && (
                  <>
                    <CircleNotch className="size-3 shrink-0 animate-spin" />
                    {msg("submit.react.mcp_checking")}
                  </>
                )}
                {probe.kind === "ok" && (
                  <>
                    <CheckCircle className="size-3 shrink-0" />
                    {formatMsg("submit.react.mcp_connected", { p1: probe.toolCount })}
                  </>
                )}
                {probe.kind === "error" && (
                  <>
                    <Warning className="size-3 shrink-0" />
                    {msg("submit.react.mcp_failed")}
                    <button
                      type="button"
                      onClick={() => runProbe(reactConfig.mcpUrl.trim(), reactConfig.mcpAuthHeader)}
                      className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border/60 bg-background px-1.5 py-0.5 text-[0.625rem] font-medium text-foreground transition-colors hover:border-[#C8A882]"
                    >
                      <ArrowsClockwise className="size-2.5" />
                      {msg("submit.react.mcp_retry")}
                    </button>
                  </>
                )}
              </div>
              {tools.length > 1 && (
                <div className="flex shrink-0 items-center gap-1.5" dir="ltr">
                  <button
                    type="button"
                    aria-label={msg("submit.react.tools_prev")}
                    disabled={toolIndex === 0}
                    onClick={() => setToolIndex((i) => Math.max(0, i - 1))}
                    className={NAV_BUTTON_CLASS}
                  >
                    <CaretLeft className="size-3.5" />
                  </button>
                  <span className="text-[0.625rem] font-medium text-muted-foreground tabular-nums">
                    {toolIndex + 1} / {tools.length}
                  </span>
                  <button
                    type="button"
                    aria-label={msg("submit.react.tools_next")}
                    disabled={toolIndex >= tools.length - 1}
                    onClick={() => setToolIndex((i) => Math.min(tools.length - 1, i + 1))}
                    className={NAV_BUTTON_CLASS}
                  >
                    <CaretRight className="size-3.5" />
                  </button>
                </div>
              )}
            </div>
            {activeTool && (
              <div
                key={activeTool.name}
                role="group"
                aria-label={msg("submit.react.tools_list_label")}
                dir="ltr"
                className="space-y-1.5 rounded-lg border border-border/60 bg-background/80 p-3 motion-safe:animate-in motion-safe:fade-in motion-safe:duration-200"
              >
                <p className="font-mono text-[0.6875rem] leading-tight font-medium break-all text-foreground">
                  {activeTool.name}
                </p>
                <p className="text-[0.6875rem] leading-relaxed break-words whitespace-pre-line text-muted-foreground">
                  {activeTool.description ?? (
                    <span className="text-muted-foreground/70 italic">
                      {msg("submit.react.tool_no_description")}
                    </span>
                  )}
                </p>
              </div>
            )}
            {probe.kind === "error" && probe.detail && (
              <p
                className="break-words font-mono text-[0.625rem] leading-relaxed text-muted-foreground"
                dir="ltr"
              >
                {probe.detail}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
