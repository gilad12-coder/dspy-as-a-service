"use client";

import { useEffect, useState } from "react";
import { Loader2, Plus, RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { AgentThread, ChatTranscript, Composer, QuestionChoices } from "@/shared/ui/agent";
import type { AgentMessage, AgentThinking } from "@/shared/ui/agent";
import type { InterviewOption } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { AssistState, TaggerConfig } from "../lib/types";

interface Props {
  config: TaggerConfig;
  assist: AssistState;
  busy: boolean;
  streamText: string;
  thinking: AgentThinking | null;
  options: InterviewOption[];
  error: string | null;
  onSend: (content: string) => void;
  onEditResend: (index: number, content: string) => void;
  onStop: () => void;
  onRetry: () => void;
  onSkip: () => void;
  onExit: () => void;
  onTaskOverrideChange: (override: { question?: string; prompt?: string }) => void;
  onConfirmRubric: (rubric: string[]) => void;
}

/**
 * The dataset interview: the same live chat experience as the generalist
 * agent — streamed reply tokens, a collapsible thinking section, stop, and
 * edit-and-resend on any earlier answer — driving a purpose-built
 * interviewer. It ends in an editable labeling guide that also carries the
 * task's own prompt, so the whole task definition is confirmable in one place.
 */
export function TaggerInterview({
  config,
  assist,
  busy,
  streamText,
  thinking,
  options,
  error,
  onSend,
  onEditResend,
  onStop,
  onRetry,
  onSkip,
  onExit,
  onTaskOverrideChange,
  onConfirmRubric,
}: Props) {
  const [draft, setDraft] = useState("");
  const done = assist.interview.done;
  const messages: AgentMessage[] = assist.interview.turns.map((turn) => ({
    role: turn.role,
    content: turn.content,
    model: turn.model ?? null,
  }));
  // The in-flight assistant reply streams into a trailing synthetic message,
  // exactly how the agent panel renders its live turn.
  if (busy && (streamText || thinking)) {
    messages.push({ role: "assistant", content: streamText });
  }

  const submit = () => {
    const content = draft.trim();
    if (!content || busy || done) return;
    setDraft("");
    onSend(content);
  };

  return (
    <div className="flex h-[calc(100dvh-var(--header-height,53px)-3rem)] w-full flex-col md:h-[calc(100dvh-var(--header-height,53px)-4rem)]">
      {done ? (
        <RubricCard
          config={config}
          assist={assist}
          onTaskOverrideChange={onTaskOverrideChange}
          onConfirm={onConfirmRubric}
        />
      ) : (
        <Card className="flex min-h-0 flex-1 flex-col overflow-hidden py-0 gap-0">
          <div className="flex items-start justify-between gap-3 border-b border-border/40 px-4 py-3 shrink-0">
            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-foreground">
                {msg("tagger.assist.interview.title")}
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {msg("tagger.assist.interview.subtitle")}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onExit}
              className="shrink-0 gap-1.5 text-muted-foreground"
            >
              <RotateCcw className="size-3.5" />
              {msg("tagger.exit")}
            </Button>
          </div>

          <AgentThread
            scrollDeps={[messages.length, streamText, thinking?.reasoning]}
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
              editAndResend={onEditResend}
              thinking={thinking ?? undefined}
              animatePairs
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

          {!error && options.length > 0 && !busy && (
            <QuestionChoices
              options={options}
              onSelect={onSend}
              hint={msg("tagger.assist.interview.choices_hint")}
              ariaLabel={msg("tagger.assist.interview.choices_label")}
            />
          )}

          <Composer
            value={draft}
            onChange={setDraft}
            onSubmit={submit}
            onStop={onStop}
            disabled={!busy && messages.length === 0}
            streaming={busy}
            placeholder={msg("tagger.assist.interview.placeholder")}
          />

          <div className="flex justify-center border-t border-border/40 py-1.5 shrink-0">
            <Button variant="ghost" size="sm" onClick={onSkip} className="text-muted-foreground">
              {msg("tagger.assist.interview.skip")}
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}

/**
 * The distilled labeling guide, editable in place before anything runs —
 * including the binary task's own question, so refinements from the interview
 * land on the tagging prompt itself. Confirming it is the moment the user
 * takes ownership of what the AI believes.
 */
function RubricCard({
  config,
  assist,
  onTaskOverrideChange,
  onConfirm,
}: {
  config: TaggerConfig;
  assist: AssistState;
  onTaskOverrideChange: (override: { question?: string; prompt?: string }) => void;
  onConfirm: (rubric: string[]) => void;
}) {
  const [rules, setRules] = useState<string[]>(assist.rubric);
  useEffect(() => setRules(assist.rubric), [assist.rubric]);
  // Freetext carries no standalone prompt — its task is captured by the rubric
  // rules below (the interview establishes it), so only binary shows a task field.
  const taskKey = config.mode === "binary" ? "question" : null;
  const taskValue = taskKey ? String(config[taskKey] ?? "") : "";

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
        {taskKey && (
          <div className="mb-2 flex flex-col gap-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              {config.mode === "binary"
                ? msg("tagger.assist.rubric.task_question")
                : msg("tagger.assist.rubric.task_prompt")}
            </label>
            <input
              type="text"
              value={taskValue}
              onChange={(e) => onTaskOverrideChange({ [taskKey]: e.target.value })}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              dir="auto"
            />
          </div>
        )}
        {config.mode === "multiclass" && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {(config.categories ?? []).map((cat) => (
              <span
                key={cat.id}
                className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground"
                dir="auto"
              >
                {cat.label}
              </span>
            ))}
          </div>
        )}
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
