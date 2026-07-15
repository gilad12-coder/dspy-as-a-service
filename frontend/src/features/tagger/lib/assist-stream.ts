import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { readServerSentEvents } from "@/shared/lib/sse";
import { fetchWithAuthRetry, parseInterviewOptions, type InterviewOption } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";

// Resolve lazily — a module-load const races the injected window.__SKYNET_ENV__
// and freezes the build-time localhost fallback. See shared/lib/api.ts.
const apiBase = () => getRuntimeEnv().apiUrl;

export interface InterviewTurnResult {
  message: string;
  options: InterviewOption[];
  rubric: string[];
  done: boolean;
  model?: string | null;
}

export interface InterviewStreamHandlers {
  onReasoningPatch?: (chunk: string) => void;
  onMessagePatch?: (chunk: string) => void;
  /** The server is retrying a failed attempt — drop streamed partial text. */
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
  req: { turns: Array<{ role: string; content: string }>; locale?: string },
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
            model:
              typeof payload.model === "string" && payload.model ? payload.model : null,
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
