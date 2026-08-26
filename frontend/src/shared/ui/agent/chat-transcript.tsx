"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";

import { AgentBubble } from "./agent-bubble";
import { MessageActions } from "./message-actions";
import { TtftIndicator } from "./ttft-indicator";
import { UserBubble, UserBubbleEditor } from "./user-bubble";
import type { AgentMessage, AgentThinking, AgentToolCall } from "./types";

type Pair = {
  key: string;
  user: { msg: AgentMessage; index: number } | null;
  agent: AgentMessage | null;
};

interface ChatTranscriptState {
  isEditingAny: boolean;
  editingIndex: number | null;
}

interface ChatTranscriptProps {
  messages: readonly AgentMessage[];
  streaming: boolean;
  editAndResend: (index: number, content: string) => void;
  thinking?: AgentThinking;
  renderToolCall?: (call: AgentToolCall, ctx: { isRetry: boolean }) => React.ReactNode;
  toolCallsBeforeContent?: boolean;
  onRunCode?: (code: string, language: string) => void;
  animatePairs?: boolean;
  trailing?: (state: ChatTranscriptState) => React.ReactNode;
}

// Owns the "messages → user/agent pairs → bubbles + edit-and-resend wiring"
// glue that was duplicated between the generalist and code-agent panels.
// The trailing slot keeps panel-specific UI (approval cards, breadcrumb,
// error rows) composable and aware of the editing state for gating.
export function ChatTranscript({
  messages,
  streaming,
  editAndResend,
  thinking,
  renderToolCall,
  toolCallsBeforeContent,
  onRunCode,
  animatePairs,
  trailing,
}: ChatTranscriptProps) {
  const [editingIndex, setEditingIndex] = React.useState<number | null>(null);
  const [editDraft, setEditDraft] = React.useState("");
  const isEditingAny = editingIndex !== null;

  const pairs = React.useMemo<Pair[]>(() => {
    const result: Pair[] = [];
    let currentUser: { msg: AgentMessage; index: number } | null = null;
    let seq = 0;
    messages.forEach((m, idx) => {
      if (m.role === "user") {
        if (currentUser) {
          result.push({ key: `t-${seq++}`, user: currentUser, agent: null });
        }
        currentUser = { msg: m, index: idx };
      } else {
        result.push({ key: `t-${seq++}`, user: currentUser, agent: m });
        currentUser = null;
      }
    });
    if (currentUser) {
      result.push({ key: `t-${seq++}`, user: currentUser, agent: null });
    }
    return result;
  }, [messages]);

  const latestAgentKey = React.useMemo(() => {
    for (let i = pairs.length - 1; i >= 0; i--) {
      const p = pairs[i];
      if (p && p.agent) return p.key;
    }
    return null;
  }, [pairs]);

  // Reset editing state if the message we were editing falls off the array
  // (e.g. agent reset). Without this, a stale editingIndex would gate the
  // pair filter against a no-longer-existing index.
  React.useEffect(() => {
    if (editingIndex === null) return;
    if (editingIndex >= messages.length || messages[editingIndex]?.role !== "user") {
      setEditingIndex(null);
      setEditDraft("");
    }
  }, [editingIndex, messages]);

  const startEdit = React.useCallback((index: number, content: string) => {
    setEditingIndex(index);
    setEditDraft(content);
  }, []);

  const cancelEdit = React.useCallback(() => {
    setEditingIndex(null);
    setEditDraft("");
  }, []);

  const submitEdit = React.useCallback(() => {
    if (editingIndex === null) return;
    const trimmed = editDraft.trim();
    if (!trimmed) return;
    editAndResend(editingIndex, trimmed);
    setEditingIndex(null);
    setEditDraft("");
  }, [editingIndex, editDraft, editAndResend]);

  const handleRegenerate = React.useCallback(
    (userIndex: number, userContent: string) => {
      if (streaming) return;
      editAndResend(userIndex, userContent);
    },
    [streaming, editAndResend],
  );

  const renderPairBody = (pair: Pair) => {
    const isEditing = pair.user !== null && editingIndex === pair.user.index;
    const agentMsg = pair.agent;
    const agentText = agentMsg?.content.trim() ?? "";
    const isStreamingThisPair = streaming && pair.key === latestAgentKey;
    const showActions =
      !isEditing &&
      agentMsg !== null &&
      !isStreamingThisPair &&
      (agentText.length > 0 || Boolean(agentMsg.model));

    // The send → first-token gap: mirrors exactly when AgentBubble collapses to
    // null (no text, tools, or reasoning yet). The ember stands in for that
    // empty placeholder and is unmounted the moment any of those signals lands.
    const pairThinking = pair.key === latestAgentKey ? thinking : undefined;
    const hasThinking = Boolean(
      pairThinking && (pairThinking.reasoning || (pairThinking.streaming && pairThinking.startedAt)),
    );
    const isWaitingFirstToken =
      isStreamingThisPair &&
      agentText.length === 0 &&
      !(agentMsg?.toolCalls?.length ?? 0) &&
      !hasThinking;

    return (
      <>
        {pair.user &&
          (isEditing ? (
            <UserBubbleEditor
              value={editDraft}
              onChange={setEditDraft}
              onSubmit={submitEdit}
              onCancel={cancelEdit}
              disabled={streaming}
            />
          ) : (
            <UserBubble
              content={pair.user.msg.content}
              editable={!streaming}
              onEdit={() => pair.user && startEdit(pair.user.index, pair.user.msg.content)}
            />
          ))}

        {agentMsg && !isEditing && (
          <div className="flex justify-end">
            <div className="flex flex-col items-end gap-1 max-w-[88%]">
              {isWaitingFirstToken && <TtftIndicator />}
              <AgentBubble
                msg={agentMsg}
                thinking={pair.key === latestAgentKey ? thinking : undefined}
                renderToolCall={renderToolCall}
                toolCallsBeforeContent={toolCallsBeforeContent}
                onRunCode={onRunCode}
                className="max-w-full"
              />
              {showActions && (
                <MessageActions
                  text={agentMsg.content}
                  model={agentMsg.model}
                  servedModel={agentMsg.servedModel}
                  onRegenerate={
                    pair.user
                      ? () =>
                          pair.user &&
                          handleRegenerate(pair.user.index, pair.user.msg.content)
                      : undefined
                  }
                />
              )}
            </div>
          </div>
        )}
      </>
    );
  };

  const visiblePairs = pairs.filter((pair) => {
    if (!isEditingAny || editingIndex === null) return true;
    if (pair.user === null) return true;
    return pair.user.index <= editingIndex;
  });

  const pairsNode = animatePairs ? (
    <AnimatePresence initial={false}>
      {visiblePairs.map((pair) => (
        <motion.div
          key={pair.key}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
          className="space-y-1.5"
        >
          {renderPairBody(pair)}
        </motion.div>
      ))}
    </AnimatePresence>
  ) : (
    visiblePairs.map((pair) => (
      <div key={pair.key} className="space-y-1.5">
        {renderPairBody(pair)}
      </div>
    ))
  );

  return (
    <>
      {pairsNode}
      {trailing?.({ isEditingAny, editingIndex })}
    </>
  );
}
