"use client";

import * as React from "react";

import { listTaggerSessions, type TaggerSessionSummary } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { TAGGER_SESSIONS_CHANGED } from "../hooks/use-tagger";
import { TaggingSessionCard } from "./TaggingSessionCard";

/**
 * The caller's saved text-labeling sessions, surfaced on the Datasets page so
 * in-progress labeling lives next to the data it came from. Self-contained:
 * fetches its own list and refreshes on {@link TAGGER_SESSIONS_CHANGED} (fired
 * when a session is created, renamed, pinned, or deleted anywhere). Renders
 * nothing until there is at least one session, so the page is unchanged for
 * users who have never labeled.
 */
export function TaggingSessionsPanel() {
  const [sessions, setSessions] = React.useState<TaggerSessionSummary[]>([]);
  const [loaded, setLoaded] = React.useState(false);

  const fetchSessions = React.useCallback(async () => {
    try {
      const res = await listTaggerSessions({ limit: 200 });
      setSessions(res.items);
    } catch {
      setSessions([]);
    } finally {
      setLoaded(true);
    }
  }, []);

  React.useEffect(() => {
    void fetchSessions();
    const onChanged = () => void fetchSessions();
    window.addEventListener(TAGGER_SESSIONS_CHANGED, onChanged);
    return () => window.removeEventListener(TAGGER_SESSIONS_CHANGED, onChanged);
  }, [fetchSessions]);

  if (!loaded || sessions.length === 0) return null;

  return (
    <section className="mt-6">
      <div className="mb-2.5 flex items-baseline gap-2">
        <h2 className="text-sm font-semibold text-foreground">{msg("tagger.session.section_title")}</h2>
        <span className="text-xs tabular-nums text-muted-foreground">{sessions.length}</span>
      </div>
      <div className="flex flex-col gap-2.5">
        {sessions.map((session) => (
          <TaggingSessionCard key={session.id} session={session} onChanged={fetchSessions} />
        ))}
      </div>
    </section>
  );
}
