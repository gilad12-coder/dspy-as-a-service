"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  createTaggerSession,
  updateTaggerSession,
  stashTaggerSession,
  type TaggerSessionDetail,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { markRecentSession } from "@/shared/lib/recent-session";
import type { DataRow, Annotation, TaggerConfig, AnnotationMode } from "../lib/types";

/** Window event the sidebar listens for to refresh its saved-session list. */
export const TAGGER_SESSIONS_CHANGED = "tagger-sessions-changed";

const AUTOSAVE_INTERVAL_MS = 60_000;

function isTagged(ann: Annotation, mode: AnnotationMode): boolean {
  if (ann === undefined || ann === null) return false;
  if (mode === "multiclass") return Array.isArray(ann) && ann.length > 0;
  return typeof ann === "string" && ann !== "";
}

/** Default name for a freshly-created session, derived from its config. */
function deriveSessionName(config: TaggerConfig): string {
  if (config.mode === "binary" && config.question?.trim()) {
    return config.question.trim().slice(0, 80);
  }
  if (config.mode === "freetext" && config.prompt?.trim()) {
    return config.prompt.trim().slice(0, 80);
  }
  return msg("tagger.session.untitled");
}

/**
 * Tagger annotation state with server-side persistence.
 *
 * Pass ``initialSession`` to rehydrate a saved session (the ``/tagger/[id]``
 * restore route); omit it to start fresh at the setup wizard. Once annotating
 * begins the session is created server-side and annotation progress is
 * autosaved on a 60-second loop — plus an immediate flush whenever you leave
 * the page — so it survives reloads and follows the user across devices, the
 * way optimizations persist.
 */
export function useTagger(initialSession?: TaggerSessionDetail | null) {
  const router = useRouter();
  const [phase, setPhase] = useState<"setup" | "annotating">(
    (initialSession?.phase as "setup" | "annotating" | undefined) ?? "setup",
  );
  const [config, setConfig] = useState<TaggerConfig | null>(
    (initialSession?.config as TaggerConfig | undefined) ?? null,
  );
  const [data, setData] = useState<DataRow[]>(
    (initialSession?.data as DataRow[] | undefined) ?? [],
  );
  const [columns, setColumns] = useState<string[]>(initialSession?.columns ?? []);
  const [annotations, setAnnotations] = useState<Record<string, Annotation>>(
    (initialSession?.annotations as Record<string, Annotation> | undefined) ?? {},
  );
  const [currentIndex, setCurrentIndex] = useState(initialSession?.current_index ?? 0);
  const [sessionId, setSessionId] = useState<string | null>(initialSession?.id ?? null);

  // Latest annotation progress not yet written to the server, or null when the
  // server is up to date. Buffered here so the 60s autosave loop (and the
  // leave-the-page flush) can batch many edits into one cheap PUT.
  const pendingRef = useRef<{
    annotations: Record<string, unknown>;
    current_index: number;
    phase: "setup" | "annotating";
  } | null>(null);

  // Always-current mirrors so the post-create handoff captures any rows tagged
  // while the dataset was still uploading (before the session id existed).
  const annotationsRef = useRef(annotations);
  const currentIndexRef = useRef(currentIndex);
  useEffect(() => {
    annotationsRef.current = annotations;
    currentIndexRef.current = currentIndex;
  }, [annotations, currentIndex]);

  const startAnnotating = useCallback((cfg: TaggerConfig, rows: DataRow[], cols: string[]) => {
    setConfig(cfg);
    setData(rows);
    setColumns(cols);
    setCurrentIndex(0);
    setAnnotations({});
    setPhase("annotating");
    setSessionId(null);
    // Persist immediately so the session shows up in the sidebar and survives a
    // reload; the returned id then drives the progress autosaves. The dataset is
    // uploaded once here, not on every subsequent annotation.
    void createTaggerSession({
      name: deriveSessionName(cfg),
      phase: "annotating",
      config: cfg as unknown as Record<string, unknown>,
      columns: cols,
      data: rows as unknown as Array<Record<string, unknown>>,
      annotations: {},
      current_index: 0,
    })
      .then((detail) => {
        setSessionId(detail.id);
        window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
        // Move to the session's own URL so leaving and returning (control panel,
        // browser back) rehydrates from the server instead of dropping back to
        // the wizard. Hand off the freshest local state — including any rows
        // tagged while the dataset uploaded — so the gate resumes without a
        // refetch. Skip the redirect if the user already navigated away.
        stashTaggerSession({
          ...detail,
          annotations: annotationsRef.current as Record<string, unknown>,
          current_index: currentIndexRef.current,
        });
        const path = window.location.pathname;
        if (path === "/tagger" || path.startsWith("/tagger/")) {
          router.replace(`/tagger/${detail.id}`);
        }
      })
      .catch(() => {
        // Best-effort: a storage-quota 409 opens the shared modal centrally and
        // the session simply isn't persisted — annotation still works in memory.
      });
  }, [router]);

  const backToSetup = useCallback(() => {
    setConfig(null);
    setData([]);
    setColumns([]);
    setAnnotations({});
    setCurrentIndex(0);
    setPhase("setup");
    // Detach from the saved session — it stays in the sidebar to resume later;
    // the next ``startAnnotating`` creates a new one.
    setSessionId(null);
  }, []);

  // Buffer each edit as pending rather than writing on every keystroke; the
  // autosave loop and the leave-the-page handler drain it. Only the mutable
  // fields are tracked (annotations / cursor / phase) — never the dataset.
  useEffect(() => {
    if (!sessionId || phase !== "annotating") return;
    pendingRef.current = {
      annotations: annotations as unknown as Record<string, unknown>,
      current_index: currentIndex,
      phase,
    };
  }, [sessionId, annotations, currentIndex, phase]);

  // Write any buffered progress to the server. Best-effort: on failure the
  // payload is re-armed (unless a newer edit already replaced it) so the next
  // tick retries. Reads the ref, so it always sends the freshest state.
  const flush = useCallback(() => {
    if (!sessionId || !pendingRef.current) return;
    const payload = pendingRef.current;
    pendingRef.current = null;
    void updateTaggerSession(sessionId, payload).catch(() => {
      pendingRef.current = pendingRef.current ?? payload;
    });
  }, [sessionId]);

  // Autosave progress every 60 seconds — and immediately whenever the user
  // leaves (tab hidden, page unload, or navigating back to the control panel,
  // which unmounts this hook) — so a returning user resumes where they left off
  // without a write on every annotation. At each of those points we also stamp
  // this session as recently visited, so the sidebar's Text-tagging button can
  // resume it when the user returns within the resume window.
  useEffect(() => {
    if (!sessionId) return;
    markRecentSession("tagger", sessionId);
    const interval = window.setInterval(() => {
      markRecentSession("tagger", sessionId);
      flush();
    }, AUTOSAVE_INTERVAL_MS);
    const onLeave = () => {
      markRecentSession("tagger", sessionId);
      flush();
    };
    const onHide = () => {
      if (document.visibilityState === "hidden") onLeave();
    };
    window.addEventListener("pagehide", onLeave);
    document.addEventListener("visibilitychange", onHide);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("pagehide", onLeave);
      document.removeEventListener("visibilitychange", onHide);
      onLeave();
    };
  }, [sessionId, flush]);

  const navigate = useCallback(
    (dir: 1 | -1) => {
      (document.activeElement as HTMLElement)?.blur();
      setCurrentIndex((i) => {
        const next = i + dir;
        if (next < 0 || next >= data.length) return i;
        return next;
      });
    },
    [data.length],
  );

  const goTo = useCallback(
    (idx: number) => {
      (document.activeElement as HTMLElement)?.blur();
      if (idx >= 0 && idx < data.length) setCurrentIndex(idx);
    },
    [data.length],
  );

  const jumpToUntagged = useCallback(() => {
    if (!config) return;
    for (let i = 0; i < data.length; i++) {
      if (!isTagged(annotations[String(data[i]!.id)], config.mode)) {
        setCurrentIndex(i);
        return;
      }
    }
  }, [data, annotations, config]);

  const toggleBinary = useCallback((id: string, value: "yes" | "no") => {
    (document.activeElement as HTMLElement)?.blur();
    setAnnotations((prev) => {
      const next = { ...prev };
      if (next[id] === value) {
        delete next[id];
      } else {
        next[id] = value;
      }
      return next;
    });
  }, []);

  const toggleCategory = useCallback((id: string, catId: string) => {
    (document.activeElement as HTMLElement)?.blur();
    setAnnotations((prev) => {
      const next = { ...prev };
      const current = Array.isArray(next[id]) ? [...(next[id] as string[])] : [];
      const idx = current.indexOf(catId);
      if (idx >= 0) current.splice(idx, 1);
      else current.push(catId);
      if (current.length === 0) delete next[id];
      else next[id] = current;
      return next;
    });
  }, []);

  const setFreetext = useCallback((id: string, text: string) => {
    setAnnotations((prev) => {
      const next = { ...prev };
      if (text.trim()) next[id] = text.trim();
      else delete next[id];
      return next;
    });
  }, []);

  const taggedCount = useMemo(
    () =>
      config ? data.filter((d) => isTagged(annotations[String(d.id)], config.mode)).length : 0,
    [config, data, annotations],
  );

  return {
    phase,
    config,
    data,
    columns,
    annotations,
    currentIndex,
    taggedCount,
    startAnnotating,
    backToSetup,
    navigate,
    goTo,
    jumpToUntagged,
    toggleBinary,
    toggleCategory,
    setFreetext,
  };
}
