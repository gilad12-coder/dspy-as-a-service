import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { readServerSentEvents } from "@/shared/lib/sse";
import { fetchWithAuthRetry, parseInterviewOptions, type InterviewOption } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import type { AssistPrediction, TaggerConfig } from "./types";

// Resolve lazily — a module-load const races the injected window.__SKYNET_ENV__
// and freezes the build-time localhost fallback. See shared/lib/api.ts.
const apiBase = () => getRuntimeEnv().apiUrl;

export interface InterviewTurnResult {
  message: string;
  options: InterviewOption[];
  rubric: string[];
  done: boolean;
  taskOverride: Partial<Pick<TaggerConfig, "mode" | "question" | "categories" | "prompt">>;
  /** Short session name the interview proposes on its final turn. */
  title: string;
  model?: string | null;
  /** Concrete model the Auto Router picked for this turn, when resolved. */
  served_model?: string | null;
}

export interface InterviewStreamHandlers {
  onReasoningPatch?: (chunk: string) => void;
  onMessagePatch?: (chunk: string) => void;
  /** The reply is fully streamed; options/rubric are still generating. */
  onMessageEnd?: () => void;
  /** The streamed ``done`` field settled: the turn ends in the task contract
   *  (final) or in another question — pick the matching placeholder. */
  onTurnHint?: (final: boolean) => void;
  /** The server dropped the partial reply (retry or leaked structure). */
  onMessageReset?: () => void;
  onDone: (turn: InterviewTurnResult) => void;
  onError: (message: string) => void;
  signal?: AbortSignal;
}

/**
 * Stream one tagger-interview turn via SSE. Mirrors `streamGeneralistAgent`
 * — same transport, same `reasoning_patch` / `message_patch` event shapes —
 * with a terminal `interview_done` instead of the agent's `done`.
 */
export async function streamInterviewTurn(
  sessionId: string,
  req: {
    turns: Array<{ role: string; content: string }>;
    locale?: string;
    model?: string;
    reasoning_effort?: string;
  },
  handlers: InterviewStreamHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetchWithAuthRetry(
      `${apiBase()}/tagging-sessions/${sessionId}/assist/interview/stream`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(req),
        signal: handlers.signal,
      },
    );
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    handlers.onError(msg("tagger.assist.interview.error"));
    return;
  }
  if (!res.ok || !res.body) {
    handlers.onError(msg("tagger.assist.interview.error"));
    return;
  }
  let finished = false;
  try {
    await readServerSentEvents(res.body, ({ event, data }) => {
      const payload = data as Record<string, unknown>;
      switch (event) {
        case "reasoning_patch":
          handlers.onReasoningPatch?.(String(payload.chunk ?? ""));
          break;
        case "message_patch":
          handlers.onMessagePatch?.(String(payload.chunk ?? ""));
          break;
        case "message_end":
          handlers.onMessageEnd?.();
          break;
        case "turn_hint":
          handlers.onTurnHint?.(payload.final === true);
          break;
        case "message_reset":
          handlers.onMessageReset?.();
          break;
        case "interview_done":
          finished = true;
          handlers.onDone({
            message: String(payload.message ?? ""),
            options: parseInterviewOptions(payload.options),
            rubric: Array.isArray(payload.rubric) ? payload.rubric.map(String) : [],
            done: payload.done === true,
            taskOverride:
              payload.task_override && typeof payload.task_override === "object"
                ? (payload.task_override as Partial<
                    Pick<TaggerConfig, "mode" | "question" | "categories" | "prompt">
                  >)
                : {},
            title: typeof payload.title === "string" ? payload.title.trim() : "",
            model: typeof payload.model === "string" && payload.model ? payload.model : null,
            served_model:
              typeof payload.served_model === "string" && payload.served_model
                ? payload.served_model
                : null,
          });
          break;
        case "error":
          finished = true;
          handlers.onError(msg("tagger.assist.interview.error"));
          break;
      }
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    if (!finished) handlers.onError(msg("tagger.assist.interview.error"));
    return;
  }
  if (!finished) handlers.onError(msg("tagger.assist.interview.error"));
}

export interface PredictStreamHandlers {
  /** One row's prediction landed — fired the moment the model writes it. */
  onPrediction: (id: string, prediction: AssistPrediction) => void;
  /** Terminal event: the authoritative merged map (covers missed events). */
  onDone: (predictions: Record<string, AssistPrediction>) => void;
  onError: () => void;
  signal?: AbortSignal;
}

/**
 * Stream per-row label predictions for a review round via SSE. Rows arrive
 * as individual `prediction` events while the batch call is still running;
 * aborting the signal is a deliberate stop, not an error.
 */
export async function streamPredictions(
  sessionId: string,
  rowIds: string[],
  handlers: PredictStreamHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetchWithAuthRetry(
      `${apiBase()}/tagging-sessions/${sessionId}/assist/predict/stream`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ row_ids: rowIds }),
        signal: handlers.signal,
      },
    );
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    handlers.onError();
    return;
  }
  if (!res.ok || !res.body) {
    handlers.onError();
    return;
  }
  let finished = false;
  try {
    await readServerSentEvents(res.body, ({ event, data }) => {
      const payload = data as Record<string, unknown>;
      switch (event) {
        case "prediction": {
          const id = String(payload.id ?? "");
          if (id && payload.prediction && typeof payload.prediction === "object") {
            handlers.onPrediction(id, payload.prediction as AssistPrediction);
          }
          break;
        }
        case "predict_done":
          finished = true;
          handlers.onDone(
            payload.predictions && typeof payload.predictions === "object"
              ? (payload.predictions as Record<string, AssistPrediction>)
              : {},
          );
          break;
        case "error":
          finished = true;
          handlers.onError();
          break;
      }
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    if (!finished) handlers.onError();
    return;
  }
  if (!finished && !handlers.signal?.aborted) handlers.onError();
}
