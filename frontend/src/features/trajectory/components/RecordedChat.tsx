"use client";

import { History } from "lucide-react";

import { AgentBubble } from "@/shared/ui/agent/agent-bubble";
import { UserBubble } from "@/shared/ui/agent/user-bubble";
import type { AgentMessage } from "@/shared/ui/agent";
import { HelpTip } from "@/shared/ui/help-tip";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";

type ChatRole = "user" | "assistant" | "system" | "tool";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

// system/tool turns rarely appear in a captured chat_history, but when they do
// they don't belong in a user/assistant bubble — render them as a centered note
// so the alternating conversation still reads cleanly.
function RecordedSystemNote({ role, content }: ChatMessage) {
  return (
    <div className="flex justify-center">
      <div
        className="max-w-[92%] rounded-lg border border-[#DDD4C8]/50 bg-background/60 px-3 py-1.5 text-[0.6875rem] leading-relaxed text-muted-foreground"
        dir="auto"
        style={{ wordBreak: "break-word" }}
      >
        <span className="me-1.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground/60">
          {role}
        </span>
        {content}
      </div>
    </div>
  );
}

/**
 * Renders a captured ``chat_history`` as a read-only replica of the live agent
 * chat — same bubbles, same alignment — but framed by a header that marks it
 * unmistakably as a recorded transcript rather than an active conversation.
 * Reuses the shared {@link UserBubble}/{@link AgentBubble} so the look tracks
 * the real chat automatically; drops every interactive affordance (composer,
 * edit-and-resend, regenerate) because there is nothing live to act on.
 */
export function RecordedChatTranscript({ messages }: { messages: readonly ChatMessage[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-[#DDD4C8]/60 bg-[#FBF7F1]/60 shadow-[0_1px_2px_rgba(28,22,18,0.02)]">
      <HelpTip text={msg("trajectory.chat.recorded_label.explain")} className="w-full">
        <div className="flex w-full items-center gap-1.5 border-b border-[#DDD4C8]/50 bg-[#F4ECE0]/70 px-3 py-1.5">
          <History className="size-3 text-[#7A6A52]" aria-hidden="true" />
          <span className="min-w-0 truncate text-[9px] font-semibold uppercase tracking-wider text-[#7A6A52]">
            {msg("trajectory.chat.recorded_label")}
          </span>
          <span className="ms-auto shrink-0 text-[9px] tabular-nums text-muted-foreground/70">
            {formatMsg("trajectory.chat.recorded_count", { n: messages.length })}
          </span>
        </div>
      </HelpTip>
      <div className="space-y-2.5 px-3 py-3">
        {messages.map((m, idx) => {
          if (m.role === "user") {
            return <UserBubble key={idx} content={m.content} editable={false} />;
          }
          if (m.role === "assistant") {
            const agentMsg: AgentMessage = { role: "assistant", content: m.content };
            return (
              <div key={idx} className={cn("flex justify-end")}>
                <AgentBubble msg={agentMsg} className="max-w-full" />
              </div>
            );
          }
          return <RecordedSystemNote key={idx} role={m.role} content={m.content} />;
        })}
      </div>
    </div>
  );
}
