"use client";

import * as React from "react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { TERMS } from "@/shared/lib/terms";

import type { AgentToolCall } from "@/shared/ui/agent/types";

import { ToolCallRow } from "./ToolCallRow";
import { PassDot } from "./result-card-atoms";

interface TestResultsCardProps {
  call: AgentToolCall;
}

interface TestItem {
  index?: number;
  outputs?: Record<string, unknown>;
  score?: number;
  pass?: boolean;
}

interface TestResult {
  baseline?: TestItem[];
  optimized?: TestItem[];
}

interface JoinedRow {
  index: number;
  base?: TestItem;
  opt?: TestItem;
}

const MAX_ROWS = 8;

function extractResult(call: AgentToolCall): TestResult | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const result = payload.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const r = result as TestResult;
  if (!Array.isArray(r.baseline) && !Array.isArray(r.optimized)) return null;
  return r;
}

function passCount(items: TestItem[] | undefined): { pass: number; total: number } {
  if (!items) return { pass: 0, total: 0 };
  return { pass: items.filter((i) => i.pass).length, total: items.length };
}

/** Join baseline and optimized examples by index, changed rows first. */
function joinRows(data: TestResult): JoinedRow[] {
  const byIndex = new Map<number, JoinedRow>();
  const put = (item: TestItem, side: "base" | "opt") => {
    const idx = typeof item.index === "number" ? item.index : -1;
    const row = byIndex.get(idx) ?? { index: idx };
    row[side] = item;
    byIndex.set(idx, row);
  };
  (data.baseline ?? []).forEach((i) => put(i, "base"));
  (data.optimized ?? []).forEach((i) => put(i, "opt"));
  const changed = (r: JoinedRow) =>
    r.base != null && r.opt != null && Boolean(r.base.pass) !== Boolean(r.opt.pass);
  return [...byIndex.values()].sort((a, b) => {
    const ca = changed(a) ? 0 : 1;
    const cb = changed(b) ? 0 : 1;
    return ca - cb || a.index - b.index;
  });
}

function previewOutputs(item: TestItem | undefined): string {
  if (!item?.outputs) return "";
  return Object.values(item.outputs)
    .map((v) => (typeof v === "string" ? v : JSON.stringify(v)))
    .join(" · ")
    .slice(0, 120);
}

function scoreText(row: JoinedRow): string {
  const s = row.opt?.score ?? row.base?.score;
  return typeof s === "number" ? s.toFixed(2) : "";
}

/**
 * Result card for ``get_test_results`` — summarizes baseline vs optimized pass
 * rates and lists per-example outcomes (a filled dot on pass, hollow on fail),
 * surfacing the examples the optimization fixed or regressed first.
 */
export function TestResultsCard({ call }: TestResultsCardProps) {
  const data = extractResult(call);
  const base = passCount(data?.baseline);
  const opt = passCount(data?.optimized);
  const summary =
    call.status === "running" || !data
      ? null
      : formatMsg("auto.features.agent.panel.components.testresultscard.passed", {
          p1: opt.pass,
          p2: opt.total,
        });

  if (!data) {
    return <ToolCallRow call={call} summary={summary} />;
  }

  const rows = joinRows(data);
  if (rows.length === 0) {
    return (
      <ToolCallRow
        call={call}
        summary={summary}
        customBody={
          <div className="text-[0.75rem] italic text-muted-foreground/70">
            {msg("auto.features.agent.panel.components.testresultscard.empty")}
          </div>
        }
      />
    );
  }

  const optColor =
    opt.pass > base.pass ? "var(--success)" : opt.pass < base.pass ? "var(--danger)" : undefined;

  const customBody = (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.75rem]">
        <span>
          <span className="text-muted-foreground/60">{TERMS.baseline} </span>
          <span dir="ltr" className="font-mono tabular-nums">
            {base.pass}/{base.total}
          </span>
        </span>
        <span>
          <span className="text-muted-foreground/60">
            {msg("auto.features.agent.panel.components.testresultscard.optimized")}{" "}
          </span>
          <span dir="ltr" className="font-mono tabular-nums" style={{ color: optColor }}>
            {opt.pass}/{opt.total}
          </span>
        </span>
      </div>

      <ul className="divide-y divide-border/40">
        {rows.slice(0, MAX_ROWS).map((row) => {
          const fixed = row.base && row.opt && !row.base.pass && row.opt.pass;
          const regressed = row.base && row.opt && row.base.pass && !row.opt.pass;
          const preview = previewOutputs(row.opt) || previewOutputs(row.base);
          return (
            <li key={row.index} className="flex items-center gap-2 py-1">
              {row.index >= 0 && (
                <span dir="ltr" className="w-8 shrink-0 font-mono text-[0.625rem] text-muted-foreground/50">
                  #{row.index}
                </span>
              )}
              <span dir="ltr" className="inline-flex shrink-0 items-center gap-1">
                <Dot item={row.base} />
                <span className="text-muted-foreground/40">→</span>
                <Dot item={row.opt} />
              </span>
              {fixed && (
                <OutcomeTag
                  label={msg("auto.features.agent.panel.components.testresultscard.fixed")}
                  tone="var(--success)"
                />
              )}
              {regressed && (
                <OutcomeTag
                  label={msg("auto.features.agent.panel.components.testresultscard.regressed")}
                  tone="var(--danger)"
                />
              )}
              {preview && (
                <span dir="auto" className="min-w-0 flex-1 truncate text-[0.625rem] text-foreground/55">
                  {preview}
                </span>
              )}
              {scoreText(row) && (
                <span
                  dir="ltr"
                  className="ms-auto shrink-0 font-mono text-[0.625rem] tabular-nums text-muted-foreground/60"
                >
                  {scoreText(row)}
                </span>
              )}
            </li>
          );
        })}
      </ul>

      {rows.length > MAX_ROWS && (
        <div className="text-[0.625rem] italic text-muted-foreground/60">
          {formatMsg("auto.features.agent.panel.components.resultcards.more", {
            p1: rows.length - MAX_ROWS,
          })}
        </div>
      )}
    </div>
  );

  return <ToolCallRow call={call} summary={summary} customBody={customBody} />;
}

function Dot({ item }: { item: TestItem | undefined }) {
  if (!item) return <span className="inline-block size-2 rounded-full bg-muted-foreground/20" />;
  return <PassDot pass={Boolean(item.pass)} />;
}

function OutcomeTag({ label, tone }: { label: string; tone: string }) {
  return (
    <span
      className="inline-flex shrink-0 items-center rounded-full px-1.5 py-0.5 text-[0.5625rem] font-medium leading-none"
      style={{ backgroundColor: `color-mix(in oklab, ${tone} 14%, transparent)`, color: tone }}
    >
      {label}
    </span>
  );
}
