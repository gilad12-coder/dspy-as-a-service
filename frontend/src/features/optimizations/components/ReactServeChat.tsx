"use client";

import * as React from "react";
import { ChatText, CircleNotch, XCircle } from "@/shared/ui/icons";

import { AgentThread } from "@/shared/ui/agent/agent-thread";
import { ChatTranscript } from "@/shared/ui/agent/chat-transcript";
import { Composer } from "@/shared/ui/agent/composer";
import type { AgentThinking, AgentToolCall } from "@/shared/ui/agent/types";
import { EmptyState } from "@/shared/ui/empty-state";
import { RetryIconButton } from "@/shared/ui/retry-icon-button";
import { msg } from "@/shared/lib/messages";

import { ApprovalCard, ToolCallRow, TrustToggle, useTrustMode } from "@/features/agent-panel";

import { useReactServeChat } from "../hooks/use-react-serve-chat";

export interface ReactServeChatProps {
  optimizationId: string;
}

// Live, tool-using chat for a served ReAct run. Reuses the generalist agent's
// chat primitives (thread, transcript, tool-call rows, approval card, trust
// toggle, composer) so it looks and behaves identically — only the transport
// (`/serve/{id}/chat`) and the absence of wizard concerns differ.
export function ReactServeChat({ optimizationId }: ReactServeChatProps) {
  const { mode: trustMode, next: cycleTrust } = useTrustMode();
  const agent = useReactServeChat(optimizationId, trustMode);
  const [draft, setDraft] = React.useState("");
  const streaming = agent.status === "streaming";

  const thinking: AgentThinking = {
    reasoning: agent.reasoning,
    startedAt: agent.reasoningStartedAt,
    endedAt: agent.reasoningEndedAt,
    streaming,
  };

  const renderToolCall = React.useCallback(
    (call: AgentToolCall, ctx: { isRetry: boolean }) => (
      <ToolCallRow call={call} isRetry={ctx.isRetry} summary={null} />
    ),
    [],
  );

  const handleSubmit = () => {
    const trimmed = draft.trim();
    if (!trimmed || streaming) return;
    agent.send(trimmed);
    setDraft("");
  };

  const emptyState = (
    <EmptyState
      icon={ChatText}
      iconWrap="circle"
      variant="compact"
      title={msg("optimizations.react.chat_empty_title")}
      description={msg("optimizations.react.chat_empty_desc")}
    />
  );

  return (
    <div className="flex flex-col min-w-0 max-h-[560px] pt-2">
      <div className="flex items-center justify-end pb-2">
        <TrustToggle mode={trustMode} onCycle={cycleTrust} />
      </div>

      <AgentThread
        isEmpty={agent.messages.length === 0}
        emptyState={emptyState}
        scrollDeps={[
          agent.messages.length,
          agent.messages[agent.messages.length - 1]?.content,
          agent.messages[agent.messages.length - 1]?.toolCalls?.length,
          agent.reasoning,
          agent.statusLabel,
          agent.pendingApproval?.id ?? "",
        ]}
      >
        <ChatTranscript
          messages={agent.messages}
          streaming={streaming}
          editAndResend={agent.editAndResend}
          thinking={thinking}
          renderToolCall={renderToolCall}
          toolCallsBeforeContent
          animatePairs
          trailing={() => (
            <>
              {streaming && agent.statusLabel && !agent.pendingApproval && (
                <div
                  className="flex items-center gap-1.5 px-2 py-1 text-xs text-muted-foreground"
                  role="status"
                  aria-live="polite"
                >
                  <CircleNotch className="size-3 animate-spin" aria-hidden="true" />
                  <span dir="auto">{agent.statusLabel}</span>
                </div>
              )}
              {agent.pendingApproval && (
                <ApprovalCard payload={agent.pendingApproval} onResolve={agent.confirmApproval} />
              )}
              {agent.error && (
                <div className="flex items-start gap-1.5 rounded-lg border border-[#9B2C1F]/20 bg-[#FCEFEB]/60 px-2.5 py-2 text-xs text-[#7A1E13]">
                  <XCircle className="mt-0.5 size-3 shrink-0 text-[#9B2C1F]" />
                  <span className="min-w-0 flex-1 break-words" dir="auto">
                    {agent.error}
                  </span>
                  <RetryIconButton
                    label={msg("optimizations.react.chat_retry")}
                    onClick={agent.retry}
                    className="size-7 border-[#9B2C1F]/25 bg-transparent text-[#7A1E13] shadow-none hover:bg-[#9B2C1F]/10 hover:text-[#7A1E13]"
                  />
                </div>
              )}
            </>
          )}
        />
      </AgentThread>

      <Composer
        value={draft}
        onChange={setDraft}
        onSubmit={handleSubmit}
        onStop={agent.stop}
        placeholder={msg("optimizations.react.chat_placeholder")}
        streaming={streaming}
        sendAriaLabel={msg("optimizations.react.chat_send_aria")}
        stopAriaLabel={msg("optimizations.react.chat_stop_aria")}
        layout="inline"
      />
    </div>
  );
}
