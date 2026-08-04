"use client";

import * as React from "react";
import { CircleNotch, Plus, ArrowCounterClockwise, Trash } from "@/shared/ui/icons";

import { Button } from "@/shared/ui/primitives/button";
import {
  AgentThread,
  ChatTranscript,
  Composer,
  ComposerModelMenu,
  QuestionChoices,
  QuestionChoicesSkeleton,
} from "@/shared/ui/agent";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { CodeInterviewState } from "@/shared/hooks/use-code-interview";

interface Props {
  interview: CodeInterviewState;
  className?: string;
}

/**
 * The Signature & Metric interview: the same live chat experience as the
 * generalist agent — streamed reply tokens, a collapsible thinking section,
 * stop, and edit-and-resend on any earlier answer — driving a purpose-built
 * interviewer. It ends in an editable authoring brief; confirming it hands
 * the directives to the seed generation.
 */
export function CodeInterviewPanel({ interview, className }: Props) {
  const [draft, setDraft] = React.useState("");

  const submit = () => {
    const content = draft.trim();
    if (!content || interview.busy || interview.done) return;
    setDraft("");
    interview.send(content);
  };

  return (
    <div className={cn("flex h-full min-h-0 flex-col overflow-hidden", className)}>
      <div className="flex items-start justify-between gap-3 border-b border-border/40 px-4 py-3 shrink-0">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-foreground">
            {msg("submit.code.interview.title")}
          </h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {msg("submit.code.interview.subtitle")}
          </p>
        </div>
        {/* The header spans both the chat and the brief card, so a re-run is
            one click away from the brief too — mirroring the tagger. */}
        <Button
          variant="ghost"
          size="sm"
          onClick={interview.reset}
          disabled={interview.busy || (interview.messages.length === 0 && !interview.done)}
          className="shrink-0 gap-1.5 text-muted-foreground"
        >
          <ArrowCounterClockwise className="size-3.5" />
          {msg("submit.code.interview.restart")}
        </Button>
      </div>

      {interview.done ? (
        <BriefCard interview={interview} />
      ) : (
        <>
          <AgentThread
            scrollDeps={[
              interview.messages.length,
              interview.streamText,
              interview.thinking?.reasoning,
            ]}
            isEmpty={interview.messages.length === 0}
            emptyState={
              <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
                <CircleNotch className="size-5 animate-spin" />
                <p className="text-sm">{msg("submit.code.interview.reading")}</p>
              </div>
            }
          >
            <ChatTranscript
              messages={interview.messages}
              streaming={interview.busy}
              editAndResend={interview.editAndResend}
              thinking={interview.thinking ?? undefined}
              animatePairs
            />
          </AgentThread>

          {interview.error && (
            <div className="flex items-center justify-between gap-3 border-t border-border/40 px-4 py-2.5">
              <p className="text-sm text-destructive">{msg("submit.code.interview.error")}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={interview.retry}
                className="gap-1.5 shrink-0"
              >
                <ArrowCounterClockwise className="size-3.5" />
                {msg("submit.code.interview.retry")}
              </Button>
            </div>
          )}

          {!interview.error && interview.options.length > 0 && !interview.busy && (
            <QuestionChoices
              options={interview.options}
              onSelect={interview.send}
              hint={msg("submit.code.interview.choices_hint")}
              ariaLabel={msg("submit.code.interview.choices_label")}
            />
          )}
          {!interview.error && interview.busy && interview.pending === "options" && (
            <QuestionChoicesSkeleton />
          )}

          <Composer
            value={draft}
            onChange={setDraft}
            onSubmit={submit}
            onStop={interview.stop}
            disabled={!interview.busy && interview.messages.length === 0}
            streaming={interview.busy}
            placeholder={msg("submit.code.interview.placeholder")}
            modelMenu={
              <ComposerModelMenu
              value={interview.model}
              onChange={interview.setModel}
              effort={interview.reasoningEffort}
              onEffortChange={interview.setReasoningEffort}
            />
            }
          />

          <div className="flex justify-center border-t border-border/40 py-1.5 shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={interview.skip}
              className="text-muted-foreground"
            >
              {msg("submit.code.interview.skip")}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * The distilled authoring brief, editable in place before any code is
 * written. Confirming it is the moment the user takes ownership of what the
 * seed authors will be told to honor.
 */
function BriefCard({ interview }: { interview: CodeInterviewState }) {
  const [directives, setDirectives] = React.useState<string[]>(interview.brief);
  React.useEffect(() => setDirectives(interview.brief), [interview.brief]);

  const update = (idx: number, value: string) =>
    setDirectives((prev) => prev.map((d, i) => (i === idx ? value : d)));
  const remove = (idx: number) => setDirectives((prev) => prev.filter((_, i) => i !== idx));
  const cleaned = directives.map((d) => d.trim()).filter(Boolean);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="px-4 pt-3 shrink-0">
        <h4 className="text-sm font-semibold text-foreground">
          {msg("submit.code.interview.brief.title")}
        </h4>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {msg("submit.code.interview.brief.description")}
        </p>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-4 py-3">
        {directives.length === 0 && (
          <p className="text-sm text-muted-foreground">
            {msg("submit.code.interview.brief.empty")}
          </p>
        )}
        {directives.map((directive, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <span className="mt-2.5 size-1.5 shrink-0 rounded-full bg-primary/50" />
            <textarea
              value={directive}
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
              aria-label={msg("submit.code.interview.brief.remove")}
              className="mt-1.5"
            >
              <Trash className="size-3.5 text-muted-foreground" />
            </Button>
          </div>
        ))}
        <Button
          variant="outline"
          size="sm"
          onClick={() => setDirectives((prev) => [...prev, ""])}
          className="mt-1 gap-1.5 self-start"
        >
          <Plus className="size-3.5" />
          {msg("submit.code.interview.brief.add")}
        </Button>
      </div>
      <div className="border-t border-border/40 p-4 shrink-0">
        <Button onClick={() => interview.confirm(cleaned)} className="w-full">
          {msg("submit.code.interview.brief.confirm")}
        </Button>
      </div>
    </div>
  );
}
