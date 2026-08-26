"use client";

import * as React from "react";

import { transcribeAudio } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";

/** The composer's voice state: recording live, transcribing the take, or
 *  showing a short-lived failure before returning to idle. */
export type DictationState =
  | { kind: "idle" }
  | { kind: "rec" }
  | { kind: "busy" }
  | { kind: "err"; message: string };

const ERR_DISMISS_MS = 2600;

/**
 * Record → transcribe → hand the text back for the draft (ported from the
 * knowledge-system composer). The transcript only lands in the caller's
 * input — it never fires a send on its own. Safari records AAC-in-MP4,
 * everywhere else webm/opus; the clip goes to ``POST /transcribe`` with the
 * UI locale as a soft language hint.
 */
export function useDictation({
  onText,
  language,
}: {
  /** Receives the finished transcript (trimmed, non-empty). */
  onText: (text: string) => void;
  /** BCP-47 UI locale forwarded as a soft STT hint. */
  language?: string;
}) {
  const [state, setState] = React.useState<DictationState>({ kind: "idle" });
  const [seconds, setSeconds] = React.useState(0);
  const recorderRef = React.useRef<MediaRecorder | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const chunksRef = React.useRef<Blob[]>([]);
  const mimeRef = React.useRef("audio/webm;codecs=opus");
  const tickRef = React.useRef<number | undefined>(undefined);
  const onTextRef = React.useRef(onText);
  React.useEffect(() => {
    onTextRef.current = onText;
  }, [onText]);

  const teardown = React.useCallback(() => {
    window.clearInterval(tickRef.current);
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.stop();
    }
    recorderRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const fail = React.useCallback(
    (message: string) => {
      teardown();
      setState({ kind: "err", message });
      window.setTimeout(
        () => setState((s) => (s.kind === "err" ? { kind: "idle" } : s)),
        ERR_DISMISS_MS,
      );
    },
    [teardown],
  );

  const start = React.useCallback(async () => {
    if (recorderRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      fail(msg("agent.composer.mic_unavailable"));
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      fail(msg("agent.composer.mic_unavailable"));
      return;
    }
    // Safari records AAC-in-MP4, everywhere else webm/opus.
    const isSafari = /apple/i.test(navigator.vendor);
    const candidates = isSafari
      ? ["audio/mp4", "audio/webm;codecs=opus"]
      : ["audio/webm;codecs=opus", "audio/mp4"];
    const mime =
      candidates.find((c) => MediaRecorder.isTypeSupported(c)) ?? "audio/webm;codecs=opus";
    mimeRef.current = mime;
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, { mimeType: mime });
    } catch {
      recorder = new MediaRecorder(stream);
    }
    streamRef.current = stream;
    recorderRef.current = recorder;
    chunksRef.current = [];
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data);
    };
    const t0 = Date.now();
    setSeconds(0);
    tickRef.current = window.setInterval(
      () => setSeconds(Math.round((Date.now() - t0) / 1000)),
      500,
    );
    setState({ kind: "rec" });
    recorder.start(1000);
  }, [fail]);

  const cancel = React.useCallback(() => {
    teardown();
    chunksRef.current = [];
    setState({ kind: "idle" });
  }, [teardown]);

  const finish = React.useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeRef.current });
      chunksRef.current = [];
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      recorderRef.current = null;
      void (async () => {
        try {
          const filename = mimeRef.current.startsWith("audio/mp4") ? "take.m4a" : "take.webm";
          const { text } = await transcribeAudio(blob, filename, language);
          const clean = (text ?? "").trim();
          // Silence or a decode miss isn't a draft — surface it as a
          // retryable failure instead of appending nothing.
          if (!clean) throw new Error("empty transcript");
          setState({ kind: "idle" });
          onTextRef.current(clean);
        } catch {
          fail(msg("agent.composer.transcribe_failed"));
        }
      })();
    };
    window.clearInterval(tickRef.current);
    setState({ kind: "busy" });
    recorder.stop();
  }, [fail, language]);

  // A recording left running when the host unmounts would hold the mic open.
  React.useEffect(() => teardown, [teardown]);

  return { state, seconds, start, cancel, finish };
}

/** mm:ss for the live recording timer. */
export function formatRecSeconds(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
