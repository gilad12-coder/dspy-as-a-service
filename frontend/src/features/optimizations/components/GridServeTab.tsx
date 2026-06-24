"use client";

/**
 * Grid-level serve playground — lets the user pick any successful pair
 * from a grid search and run it interactively. Defaults to the pair with
 * the highest final score.
 */

import { useEffect, useRef, useState } from "react";
import { Crown, Gauge, Trash2, Trophy } from "lucide-react";
import { toast } from "react-toastify";
import { msg } from "@/shared/lib/messages";

import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { Separator } from "@/shared/ui/primitives/separator";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { cn } from "@/shared/lib/utils";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { tip } from "@/shared/lib/tooltips";
import { getPairServeInfo, servePairProgramStream } from "@/shared/lib/api";
import type { OptimizationStatusResponse, PairResult, ServeInfoResponse } from "@/shared/types/api";

import { CopyButton } from "./ui-primitives";
import { ServeChat } from "./ServeChat";
import { ServeCodeSnippets } from "./ServeCodeSnippets";
import { computePairScores } from "../lib/pair-scores";

function shortEffort(value: string | null | undefined): string | null {
  if (!value) return null;
  const v = value.toLowerCase();
  if (v === "minimal") return "min";
  if (v === "medium") return "med";
  return v;
}

function pairLabel(p: PairResult): string {
  const gen = p.generation_model.split("/").pop();
  const ref = p.reflection_model.split("/").pop();
  const genE = shortEffort(p.generation_reasoning_effort);
  const refE = shortEffort(p.reflection_reasoning_effort);
  const genStr = genE ? `${gen}·${genE}` : gen;
  const refStr = refE ? `${ref}·${refE}` : ref;
  return `${genStr} × ${refStr}`;
}

type RunEntry = {
  inputs: Record<string, string>;
  outputs: Record<string, unknown>;
  model: string;
  ts: number;
};

export function GridServeTab({ job }: { job: OptimizationStatusResponse }) {
  const pairs = job.grid_result?.pair_results ?? [];
  const scoring = computePairScores(pairs);

  const servable = pairs.filter(
    (p) => !p.error && p.optimized_test_metric != null && !!p.program_artifact,
  );

  const defaultPair =
    scoring.overallWinner ?? scoring.qualityWinner ?? servable[0]?.pair_index ?? null;

  // Persist the user's pair choice per-optimization so it survives SSE
  // re-renders and page reloads (browser cache). Falls back to the
  // quality winner on first visit.
  const storageKey = `grid-serve:pair:${job.optimization_id}`;
  const [selectedPair, setSelectedPair] = useState<number | null>(() => {
    if (typeof window === "undefined") return defaultPair;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw == null) return defaultPair;
      const n = parseInt(raw, 10);
      if (!Number.isFinite(n)) return defaultPair;
      // Only honor stored choice if the pair is still servable.
      return servable.some((p) => p.pair_index === n) ? n : defaultPair;
    } catch {
      return defaultPair;
    }
  });
  useEffect(() => {
    if (selectedPair == null) return;
    try {
      window.localStorage.setItem(storageKey, String(selectedPair));
    } catch {
      /* storage unavailable (private mode, quota) — selection is in-memory only */
    }
  }, [selectedPair, storageKey]);
  const [serveInfo, setServeInfo] = useState<ServeInfoResponse | null>(null);
  const [runHistory, setRunHistory] = useState<RunEntry[]>([]);
  const [streamingRun, setStreamingRun] = useState<{
    inputs: Record<string, string>;
    partial: Record<string, string>;
  } | null>(null);
  const [serveLoading, setServeLoading] = useState(false);
  const [serveError, setServeError] = useState<string | null>(null);

  const streamReqIdRef = useRef(0);
  const streamAbortRef = useRef<AbortController | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const textareaRefs = useRef<Record<string, HTMLTextAreaElement | null>>({});

  useEffect(() => {
    if (selectedPair == null) {
      setServeInfo(null);
      return;
    }
    let cancelled = false;
    getPairServeInfo(job.optimization_id, selectedPair)
      .then((info) => {
        if (!cancelled) setServeInfo(info);
      })
      .catch(() => {
        if (!cancelled) setServeInfo(null);
      });
    return () => {
      cancelled = true;
    };
  }, [job.optimization_id, selectedPair]);

  useEffect(() => {
    streamAbortRef.current?.abort();
    setRunHistory([]);
    setStreamingRun(null);
    setServeLoading(false);
    setServeError(null);
  }, [selectedPair]);

  useEffect(() => {
    if (chatScrollRef.current) {
      const el = chatScrollRef.current;
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 300) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [runHistory, streamingRun]);

  useEffect(() => {
    return () => {
      streamAbortRef.current?.abort();
    };
  }, []);

  if (!job.grid_result) return null;

  const readInputs = () => {
    const vals: Record<string, string> = {};
    for (const f of serveInfo?.input_fields ?? []) vals[f] = textareaRefs.current[f]?.value ?? "";
    return vals;
  };

  const handleServe = async (overrideInputs?: Record<string, string>) => {
    if (!serveInfo || selectedPair == null) return;
    const inputs = overrideInputs ?? readInputs();
    const missing = serveInfo.input_fields.filter((f) => !inputs[f]?.trim());
    if (missing.length > 0) {
      toast.error(
        <div>
          {msg("auto.features.optimizations.components.gridservetab.1")}
          <br />
          {missing.join(", ")}
        </div>,
      );
      return;
    }
    streamAbortRef.current?.abort();
    const reqId = ++streamReqIdRef.current;
    const controller = new AbortController();
    streamAbortRef.current = controller;
    setServeLoading(true);
    setServeError(null);
    setStreamingRun({ inputs: { ...inputs }, partial: {} });
    if (!overrideInputs) {
      Object.values(textareaRefs.current).forEach((el) => {
        if (el) {
          el.value = "";
          el.style.height = "auto";
        }
      });
    }
    const isStale = () => reqId !== streamReqIdRef.current;
    await servePairProgramStream(job.optimization_id, selectedPair, inputs, {
      signal: controller.signal,
      onToken: (field, chunk) => {
        if (isStale()) return;
        setStreamingRun((prev) =>
          prev
            ? {
                ...prev,
                partial: { ...prev.partial, [field]: (prev.partial[field] ?? "") + chunk },
              }
            : prev,
        );
      },
      onFinal: (res) => {
        if (isStale()) return;
        setRunHistory((prev) => [
          { inputs: { ...inputs }, outputs: res.outputs, model: res.model_used, ts: Date.now() },
          ...prev,
        ]);
        setStreamingRun(null);
      },
      onError: (message) => {
        if (isStale()) return;
        setServeError(message);
        setStreamingRun(null);
      },
    });
    if (!isStale()) setServeLoading(false);
  };

  const handleClearHistory = () => {
    setRunHistory([]);
    setServeError(null);
  };

  if (servable.length === 0) {
    return (
      <div className="rounded-xl border border-border/50 bg-card/80 p-8 text-center">
        <p className="text-sm text-muted-foreground">
          {msg("auto.features.optimizations.components.gridservetab.2")}
        </p>
      </div>
    );
  }

  const selected = pairs.find((p) => p.pair_index === selectedPair);
  const API = getRuntimeEnv().apiUrl;
  const endpoint = `${API}/serve/${job.optimization_id}/pair/${selectedPair}`;

  return (
    <div className="space-y-4">
      <FadeIn>
        <p className="text-sm text-muted-foreground">
          {msg("auto.features.optimizations.components.gridservetab.3")}
        </p>
      </FadeIn>

      <FadeIn>
        <div className="rounded-xl border border-border/50 bg-card/80 p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-medium text-muted-foreground">
              <HelpTip text={tip("grid.best_pair_default")}>
                {msg("auto.features.optimizations.components.gridservetab.4")}
              </HelpTip>
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {servable.map((p) => {
              const s = scoring.byIndex[p.pair_index];
              const isOverall = scoring.overallWinner === p.pair_index;
              const isQuality = scoring.qualityWinner === p.pair_index;
              const isSpeed = scoring.speedWinner === p.pair_index;
              const isSelected = selectedPair === p.pair_index;
              return (
                <button
                  key={p.pair_index}
                  type="button"
                  onClick={() => setSelectedPair(p.pair_index)}
                  className={cn(
                    "group relative rounded-lg border p-3 text-start transition-all duration-200",
                    "hover:shadow-sm",
                    isSelected
                      ? "border-[#3D2E22]/60 bg-[#3D2E22]/[0.06] ring-1 ring-[#3D2E22]/30"
                      : "border-border/60 bg-background/40 hover:border-border",
                  )}
                  aria-pressed={isSelected}
                >
                  <div className="flex items-center gap-1.5 mb-1.5">
                    {isOverall && <Crown className="size-3 text-[#C8A882] shrink-0" />}
                    <span
                      className="font-mono text-[0.6875rem] truncate flex-1"
                      dir="ltr"
                      title={pairLabel(p)}
                    >
                      {pairLabel(p)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] tabular-nums text-muted-foreground">
                    <span className={cn("flex items-center gap-1", isQuality && "text-[#3D2E22]")}>
                      <Trophy className="size-2.5" />
                      {s ? `${Math.round(s.quality * 100)}%` : "—"}
                    </span>
                    <span className={cn("flex items-center gap-1", isSpeed && "text-[#3D2E22]")}>
                      <Gauge className="size-2.5" />
                      {p.avg_response_time_ms != null
                        ? `${(p.avg_response_time_ms / 1000).toFixed(1)}s`
                        : "—"}
                    </span>
                    <span
                      className={cn(
                        "flex items-center gap-1 ms-auto font-semibold",
                        isOverall && "text-[#3D2E22]",
                      )}
                    >
                      {msg("auto.features.optimizations.components.gridservetab.5")}
                      {s ? ` ${Math.round(s.quality * 100)}%` : " —"}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </FadeIn>

      {serveInfo && (
        <>
          {runHistory.length > 0 && (
            <FadeIn>
              <div className="flex items-center justify-end pb-3 border-b border-border/60">
                <TooltipButton
                  tooltip={msg("auto.features.optimizations.components.gridservetab.6")}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    onClick={handleClearHistory}
                    aria-label={msg(
                      "auto.features.optimizations.components.gridservetab.literal.1",
                    )}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </TooltipButton>
              </div>
            </FadeIn>
          )}

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
            demos={selected?.program_artifact?.optimized_prompt?.demos ?? []}
          />

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                <HelpTip text={tip("serve.section_pair")}>
                  {msg("auto.features.optimizations.components.gridservetab.7")}
                </HelpTip>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <p className="text-[0.625rem] text-muted-foreground uppercase tracking-wider">
                  <HelpTip text={tip("serve.api_url_pair")}>
                    {msg("auto.features.optimizations.components.gridservetab.8")}
                  </HelpTip>
                </p>
                <div className="rounded-lg bg-muted/40 p-2.5 pe-8 relative group" dir="ltr">
                  <code className="text-xs font-mono break-all">
                    {msg("auto.features.optimizations.components.gridservetab.9")}
                    {endpoint}
                  </code>
                  <CopyButton
                    text={endpoint}
                    className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100"
                  />
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <p className="text-[0.625rem] text-muted-foreground uppercase tracking-wider">
                  <HelpTip text={tip("serve.integration_code")}>
                    {msg("auto.features.optimizations.components.gridservetab.10")}
                  </HelpTip>
                </p>
                <ServeCodeSnippets
                  serveInfo={serveInfo}
                  optimizationId={job.optimization_id}
                  pairIndex={selectedPair ?? undefined}
                />
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
