"use client";

import * as React from "react";
import Link from "next/link";
import { CheckCircle, PushPin } from "@/shared/ui/icons";
import { formatMsg, msg } from "@/shared/lib/messages";

import type { AgentToolCall } from "@/shared/ui/agent/types";

import { ToolCallRow } from "./ToolCallRow";

interface TaggingSessionsCardProps {
  call: AgentToolCall;
}

interface TaggingSessionSummary {
  id?: string;
  name?: string;
  phase?: string;
  row_count?: number;
  tagged_count?: number;
  pinned?: boolean;
  mode?: string | null;
}

interface TaggingListResult {
  items?: TaggingSessionSummary[];
  total?: number;
}

const MAX_ROWS = 6;

function extractResult(call: AgentToolCall): TaggingListResult | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const result = payload.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const r = result as TaggingListResult;
  if (!Array.isArray(r.items)) return null;
  return r;
}

function isComplete(s: TaggingSessionSummary): boolean {
  if (s.phase === "complete") return true;
  const rows = s.row_count ?? 0;
  return rows > 0 && (s.tagged_count ?? 0) >= rows;
}

function buildSummary(data: TaggingListResult | null, isRunning: boolean): string | null {
  if (isRunning || !data) return null;
  const total = data.total ?? data.items?.length ?? 0;
  if (total === 0) return msg("auto.features.agent.panel.components.taggingsessionscard.empty");
  return formatMsg("auto.features.agent.panel.components.taggingsessionscard.summary", { p1: total });
}

/**
 * Result card for ``list_tagging_sessions_for_agent`` — the caller's tagger
 * sessions as a compact list with tagging progress and a link into each
 * ``/tagger/{id}``. Read-only: the generalist answers questions about sessions;
 * the dedicated tagger-assist agent still owns the tagging itself.
 */
export function TaggingSessionsCard({ call }: TaggingSessionsCardProps) {
  const data = extractResult(call);
  const summary = buildSummary(data, call.status === "running");

  if (!data) {
    return <ToolCallRow call={call} summary={summary} />;
  }

  const items = data.items ?? [];

  const customBody =
    items.length === 0 ? (
      <div className="text-[0.75rem] italic text-muted-foreground/70">
        {msg("auto.features.agent.panel.components.taggingsessionscard.empty")}
      </div>
    ) : (
      <div className="space-y-2">
        <ul className="divide-y divide-border/40">
          {items.slice(0, MAX_ROWS).map((s, idx) => (
            <li key={s.id ?? idx} className="flex items-center gap-2 py-1.5 first:pt-0 last:pb-0">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1">
                  {s.pinned && (
                    <PushPin className="size-2.5 shrink-0 text-muted-foreground/50" aria-hidden="true" />
                  )}
                  {s.id ? (
                    <Link
                      href={`/tagger/${s.id}`}
                      dir="auto"
                      className="min-w-0 truncate text-[0.75rem] font-medium text-foreground/90 transition-colors hover:text-foreground"
                    >
                      {s.name?.trim() || s.id.slice(0, 8)}
                    </Link>
                  ) : (
                    <span dir="auto" className="min-w-0 truncate text-[0.75rem] font-medium text-foreground/90">
                      {s.name?.trim() || "—"}
                    </span>
                  )}
                </div>
                <div dir="auto" className="text-[0.625rem] text-muted-foreground/60">
                  {formatMsg("auto.features.agent.panel.components.taggingsessionscard.tagged", {
                    p1: s.tagged_count ?? 0,
                    p2: s.row_count ?? 0,
                  })}
                </div>
              </div>
              {isComplete(s) && (
                <CheckCircle
                  className="size-3.5 shrink-0"
                  style={{ color: "var(--success)" }}
                  aria-hidden="true"
                />
              )}
            </li>
          ))}
        </ul>
        {items.length > MAX_ROWS && (
          <div className="text-[0.625rem] italic text-muted-foreground/60">
            {formatMsg("auto.features.agent.panel.components.resultcards.more", {
              p1: items.length - MAX_ROWS,
            })}
          </div>
        )}
      </div>
    );

  return <ToolCallRow call={call} summary={summary} customBody={customBody} />;
}
