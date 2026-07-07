"use client";

import { useEffect, useState } from "react";
import { Loader2, Plus, RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { AgentThread, ChatTranscript, Composer } from "@/shared/ui/agent";
import type { AgentMessage } from "@/shared/ui/agent";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { AssistState } from "../lib/types";

interface Props {
  assist: AssistState;
  busy: boolean;
  quickReplies: string[];
  error: string | null;
  onSend: (content: string) => void;
  onRetry: () => void;
  onSkip: () => void;
  onConfirmRubric: (rubric: string[]) => void;
}

/**
 * The dataset interview: a focused chat column where the AI asks a handful of
 * grounded questions and distills the answers into an editable labeling
 * rubric. Human-first by design — nothing is tagged until the user confirms
 * the rubric.
 */
export function TaggerInterview({
  assist,
  busy,
  quickReplies,
  error,
  onSend,
  onRetry,
  onSkip,
  onConfirmRubric,
}: Props) {
  const [draft, setDraft] = useState("");
  const done = assist.interview.done;
  const messages: AgentMessage[] = assist.interview.turns.map((turn) => ({
    role: turn.role,
    content: turn.content,
  }));

  const submit = () => {
    const content = draft.trim();
    if (!content || busy || done) return;
    setDraft("");
    onSend(content);
  };

  return (
    <div className="mx-auto flex h-[calc(100dvh-var(--header-height,53px)-3rem)] max-w-2xl flex-col md:h-[calc(100dvh-var(--header-height,53px)-4rem)]">
      <div className="px-1 pb-3">
        <h2 className="text-base font-semibold text-foreground">
          {msg("tagger.assist.interview.title")}
        </h2>
        <p className="text-sm text-muted-foreground">
          {msg("tagger.assist.interview.subtitle")}
        </p>
      </div>

      {done ? (
        <RubricCard assist={assist} onConfirm={onConfirmRubric} />
      ) : (
        <Card className="flex min-h-0 flex-1 flex-col overflow-hidden py-0 gap-0">
          <AgentThread
            scrollDeps={[messages.length, busy]}
            isEmpty={messages.length === 0}
            emptyState={
              <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
                <Loader2 className="size-5 animate-spin" />
                <p className="text-sm">{msg("tagger.assist.interview.reading")}</p>
              </div>
            }
          >
            <ChatTranscript
              messages={messages}
              streaming={busy}
              editAndResend={() => {}}
              trailing={() =>
                busy && messages.length > 0 ? (
                  <div className="flex items-center gap-2 px-1 py-2 text-muted-foreground">
                    <Loader2 className="size-3.5 animate-spin" />
                    <span className="text-xs">{msg("tagger.assist.interview.thinking")}</span>
                  </div>
                ) : null
              }
            />
          </AgentThread>

          {error && (
            <div className="flex items-center justify-between gap-3 border-t border-border/40 px-4 py-2.5">
              <p className="text-sm text-destructive">{msg("tagger.assist.interview.error")}</p>
              <Button variant="outline" size="sm" onClick={onRetry} className="gap-1.5 shrink-0">
                <RotateCcw className="size-3.5" />
                {msg("tagger.assist.retry")}
              </Button>
            </div>
          )}

          {!error && quickReplies.length > 0 && !busy && (
            <div className="flex flex-wrap gap-1.5 border-t border-border/40 px-4 pt-2.5">
              {quickReplies.map((reply) => (
                <button
                  key={reply}
                  type="button"
                  onClick={() => onSend(reply)}
                  className={cn(
                    "rounded-full border border-border bg-background px-3 py-1 text-xs",
                    "text-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 cursor-pointer",
                  )}
                >
                  {reply}
                </button>
              ))}
            </div>
          )}

          <Composer
            value={draft}
            onChange={setDraft}
            onSubmit={submit}
            disabled={busy || messages.length === 0}
            streaming={busy}
            placeholder={msg("tagger.assist.interview.placeholder")}
          />
        </Card>
      )}

      {!done && (
        <div className="flex justify-center pt-2">
          <Button variant="ghost" size="sm" onClick={onSkip} className="text-muted-foreground">
            {msg("tagger.assist.interview.skip")}
          </Button>
        </div>
      )}
    </div>
  );
}

/**
 * The distilled labeling rubric, editable in place before anything runs. The
 * rubric is the shared memory of the whole session — confirming it is the
 * moment the user takes ownership of what the AI believes.
 */
function RubricCard({
  assist,
  onConfirm,
}: {
  assist: AssistState;
  onConfirm: (rubric: string[]) => void;
}) {
  const [rules, setRules] = useState<string[]>(assist.rubric);
  useEffect(() => setRules(assist.rubric), [assist.rubric]);

  const update = (idx: number, value: string) =>
    setRules((prev) => prev.map((r, i) => (i === idx ? value : r)));
  const remove = (idx: number) => setRules((prev) => prev.filter((_, i) => i !== idx));
  const cleaned = rules.map((r) => r.trim()).filter(Boolean);

  return (
    <Card className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <CardHeader>
        <CardTitle className="text-base">{msg("tagger.assist.rubric.title")}</CardTitle>
        <CardDescription>{msg("tagger.assist.rubric.description")}</CardDescription>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
        {rules.length === 0 && (
          <p className="text-sm text-muted-foreground">{msg("tagger.assist.rubric.empty")}</p>
        )}
        {rules.map((rule, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <span className="mt-2.5 size-1.5 shrink-0 rounded-full bg-primary/50" />
            <textarea
              value={rule}
              onChange={(e) => update(idx, e.target.value)}
              rows={2}
              className={cn(
                "flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm",
                "leading-relaxed outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
              )}
              dir="auto"
            />
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => remove(idx)}
              aria-label={msg("tagger.assist.rubric.remove")}
              className="mt-1.5"
            >
              <Trash2 className="size-3.5 text-muted-foreground" />
            </Button>
          </div>
        ))}
        <Button
          variant="outline"
          size="sm"
          onClick={() => setRules((prev) => [...prev, ""])}
          className="mt-1 gap-1.5 self-start"
        >
          <Plus className="size-3.5" />
          {msg("tagger.assist.rubric.add")}
        </Button>
      </CardContent>
      <div className="border-t border-border/40 p-4">
        <Button onClick={() => onConfirm(cleaned)} size="lg" className="w-full">
          {assist.mode === "copilot"
            ? msg("tagger.assist.rubric.confirm_copilot")
            : msg("tagger.assist.rubric.confirm_autopilot")}
        </Button>
      </div>
    </Card>
  );
}
