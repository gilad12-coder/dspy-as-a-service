"use client";

import * as React from "react";
import dynamic from "next/dynamic";

import { cn } from "@/shared/lib/utils";

import { ThinkingSection } from "./thinking-section";
import type { AgentMessage, AgentThinking, AgentToolCall } from "./types";

// react-markdown + remark-gfm (the micromark/mdast/hast tree, ~40-60 KB gzip)
// only matter once the agent chat actually renders a message, but the panel is
// mounted on every authenticated route. Lazy-load it so the parser stays out of
// the global first-load chunk; the Suspense fallback shows the raw text meanwhile.
const MessageMarkdown = dynamic(
  () => import("./message-markdown").then((m) => m.MessageMarkdown),
  { ssr: false },
);

interface AgentBubbleProps {
  msg: AgentMessage;
  thinking?: AgentThinking;
  renderToolCall?: (call: AgentToolCall, ctx: { isRetry: boolean }) => React.ReactNode;
  onRunCode?: (code: string, language: string) => void;
  toolCallsBeforeContent?: boolean;
  className?: string;
}

export function AgentBubble({
  msg,
  thinking,
  renderToolCall,
  onRunCode,
  toolCallsBeforeContent = false,
  className,
}: AgentBubbleProps) {
  const visibleToolCalls = React.useMemo<Array<{ call: AgentToolCall; isRetry: boolean }>>(() => {
    const calls = msg.toolCalls;
    if (!calls?.length) return [];
    const result: Array<{ call: AgentToolCall; isRetry: boolean }> = [];
    for (let i = 0; i < calls.length; i++) {
      const call = calls[i];
      if (!call) continue;
      if (call.status === "error") {
        const superseded = calls
          .slice(i + 1)
          .some(
            (later) =>
              later.tool === call.tool && (later.status === "done" || later.status === "running"),
          );
        if (superseded) continue;
      }
      const isRetry = calls
        .slice(0, i)
        .some((prior) => prior.tool === call.tool && prior.status === "error");
      result.push({ call, isRetry });
    }
    return result;
  }, [msg.toolCalls]);

  const hasTools = visibleToolCalls.length > 0 && Boolean(renderToolCall);
  const hasText = msg.content.trim().length > 0;
  const hasThinking = Boolean(
    thinking && (thinking.reasoning || (thinking.streaming && thinking.startedAt)),
  );

  if (!hasTools && !hasText && !hasThinking) return null;

  const toolCalls = hasTools && renderToolCall && (
    <div
      className={cn(
        "flex flex-col gap-1.5 px-2.5 py-2",
        (hasThinking || (!toolCallsBeforeContent && hasText)) && "border-t border-[#3D2E22]/[0.08]",
      )}
    >
      {visibleToolCalls.map(({ call, isRetry }) => (
        <React.Fragment key={call.id}>{renderToolCall(call, { isRetry })}</React.Fragment>
      ))}
    </div>
  );

  return (
    <div
      className={cn(
        "max-w-[88%] overflow-hidden rounded-[22px] rounded-ee-[4px] bg-muted/60 shadow-sm",
        className,
      )}
    >
      {hasThinking && thinking && <ThinkingSection thinking={thinking} />}
      {toolCallsBeforeContent && toolCalls}
      {hasText && (
        <div
          className={cn(
            "px-4 py-3 text-sm leading-relaxed text-foreground",
            (hasThinking || (toolCallsBeforeContent && hasTools)) &&
              "border-t border-[#3D2E22]/[0.08]",
          )}
        >
          <React.Suspense
            fallback={<span className="whitespace-pre-wrap">{msg.content}</span>}
          >
            <MessageMarkdown content={msg.content} onRunCode={onRunCode} />
          </React.Suspense>
        </div>
      )}
      {!toolCallsBeforeContent && toolCalls}
    </div>
  );
}
