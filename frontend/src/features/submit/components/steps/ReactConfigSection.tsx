"use client";

import * as React from "react";
import { AlertTriangle, CheckCircle2, Loader2, RefreshCw } from "lucide-react";

import { Label } from "@/shared/ui/primitives/label";
import { Input } from "@/shared/ui/primitives/input";
import { HelpTip } from "@/shared/ui/help-tip";
import { cn } from "@/shared/lib/utils";
import { tip } from "@/shared/lib/tooltips";
import { formatMsg, msg } from "@/shared/lib/messages";
import { probeMcp } from "@/shared/lib/api";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

type ProbeStatus =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "ok"; toolCount: number; toolNames: string[] }
  | { kind: "error"; detail: string };

// How long the URL/auth fields must sit still before the connection check
// fires — long enough to skip mid-typing states, short enough to feel live.
const PROBE_DEBOUNCE_MS = 700;

export function ReactConfigSection({ w }: { w: SubmitWizardContext }) {
  const { reactConfig, updateReactConfig } = w;
  const [probe, setProbe] = React.useState<ProbeStatus>({ kind: "idle" });
  // Monotonic sequence so a slow response for an old URL can't overwrite the
  // status of the current one.
  const probeSeqRef = React.useRef(0);

  const runProbe = React.useCallback((url: string, auth: string) => {
    const seq = ++probeSeqRef.current;
    setProbe({ kind: "checking" });
    probeMcp({ mcp_url: url, ...(auth.trim() ? { auth_header: auth.trim() } : {}) })
      .then((res) => {
        if (probeSeqRef.current !== seq) return;
        setProbe(
          res.ok
            ? { kind: "ok", toolCount: res.tool_count, toolNames: res.tool_names }
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
          <div className="space-y-1.5">
            <div
              className={cn(
                "flex items-center gap-1.5 text-[0.6875rem] font-medium",
                probe.kind === "checking" && "text-muted-foreground",
                probe.kind === "ok" && "text-[#5A7247]",
                probe.kind === "error" && "text-[#A3512B]",
              )}
              aria-live="polite"
            >
              {probe.kind === "checking" && (
                <>
                  <Loader2 className="size-3 shrink-0 animate-spin" />
                  {msg("submit.react.mcp_checking")}
                </>
              )}
              {probe.kind === "ok" && (
                <>
                  <CheckCircle2 className="size-3 shrink-0" />
                  {formatMsg("submit.react.mcp_connected", { p1: probe.toolCount })}
                </>
              )}
              {probe.kind === "error" && (
                <>
                  <AlertTriangle className="size-3 shrink-0" />
                  {msg("submit.react.mcp_failed")}
                  <button
                    type="button"
                    onClick={() => runProbe(reactConfig.mcpUrl.trim(), reactConfig.mcpAuthHeader)}
                    className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border/60 bg-background px-1.5 py-0.5 text-[0.625rem] font-medium text-foreground transition-colors hover:border-[#C8A882]"
                  >
                    <RefreshCw className="size-2.5" />
                    {msg("submit.react.mcp_retry")}
                  </button>
                </>
              )}
            </div>
            {probe.kind === "ok" && probe.toolNames.length > 0 && (
              <div className="flex flex-wrap gap-1" dir="ltr">
                {probe.toolNames.slice(0, 8).map((name) => (
                  <span
                    key={name}
                    className="rounded-md border border-border/50 bg-background px-1.5 py-0.5 font-mono text-[0.625rem] text-muted-foreground"
                  >
                    {name}
                  </span>
                ))}
                {probe.toolNames.length > 8 && (
                  <span className="px-1 py-0.5 text-[0.625rem] text-muted-foreground">
                    +{probe.toolNames.length - 8}
                  </span>
                )}
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
