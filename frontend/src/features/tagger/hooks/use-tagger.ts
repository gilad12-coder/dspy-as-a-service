"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
import {
  createTaggerSession,
  updateTaggerSession,
  type TaggerSessionDetail,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import type { DataRow, Annotation, TaggerConfig, AnnotationMode } from "../lib/types";

/** Window event the sidebar listens for to refresh its saved-session list. */
export const TAGGER_SESSIONS_CHANGED = "tagger-sessions-changed";

const AUTOSAVE_DEBOUNCE_MS = 1000;

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
 * autosaved (debounced) so it survives reloads and follows the user across
 * devices, the way optimizations persist.
 */
export function useTagger(initialSession?: TaggerSessionDetail | null) {
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

  const startAnnotating = useCallback((cfg: TaggerConfig, rows: DataRow[], cols: string[]) => {
    setConfig(cfg);
    setData(rows);
    setColumns(cols);
    setCurrentIndex(0);
    setAnnotations({});
    setPhase("annotating");
    setSessionId(null);
    // Persist immediately so the session shows up in the sidebar and survives a
    // reload; the returned id then drives the debounced progress autosaves. The
    // dataset is uploaded once here, not on every subsequent annotation.
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
      })
      .catch(() => {
        // Best-effort: a storage-quota 409 opens the shared modal centrally and
        // the session simply isn't persisted — annotation still works in memory.
      });
  }, []);

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

  // Debounced autosave of annotation progress. Only the mutable fields are sent
  // (annotations / cursor / phase) — never the dataset — so saves stay cheap.
  useEffect(() => {
    if (!sessionId || phase !== "annotating") return;
    const handle = window.setTimeout(() => {
      void updateTaggerSession(sessionId, {
        annotations: annotations as unknown as Record<string, unknown>,
        current_index: currentIndex,
        phase,
      }).catch(() => {
        // Autosave is best-effort; the next change retries with the full state.
      });
    }, AUTOSAVE_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [sessionId, annotations, currentIndex, phase]);

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
