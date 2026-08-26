"use client";

import * as React from "react";
import { ChatText, CircleNotch } from "@/shared/ui/icons";

import { AgentThread } from "@/shared/ui/agent/agent-thread";
import { ChatErrorBanner } from "@/shared/ui/agent/chat-error-banner";
import { ChatTranscript } from "@/shared/ui/agent/chat-transcript";
import { Composer } from "@/shared/ui/agent/composer";
import type { AgentThinking, AgentToolCall } from "@/shared/ui/agent/types";
import { EmptyState } from "@/shared/ui/empty-state";
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
                <ChatErrorBanner
                  message={agent.error}
                  retryLabel={msg("optimizations.react.chat_retry")}
                  onRetry={agent.retry}
                />
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
