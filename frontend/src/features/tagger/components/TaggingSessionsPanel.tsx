"use client";

import * as React from "react";
import { CircleAlert, Loader2, Plus, Search, Tags } from "lucide-react";

import { listTaggerSessions, type TaggerSessionSummary } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { Button } from "@/shared/ui/primitives/button";
import { EmptyState } from "@/shared/ui/empty-state";
import { SearchField } from "@/shared/ui/search-field";
import { TAGGER_SESSIONS_CHANGED } from "../hooks/use-tagger";
import { TaggingSessionCard } from "./TaggingSessionCard";

/**
 * Let the caller resume a saved labeling session or begin a new one.
 *
 * Fetches its own list and refreshes on {@link TAGGER_SESSIONS_CHANGED}, fired
 * whenever a session is renamed or deleted.
 */
export function TaggingSessionsPanel({ onStartNew }: { onStartNew: () => void }) {
  const [sessions, setSessions] = React.useState<TaggerSessionSummary[]>([]);
  const [loaded, setLoaded] = React.useState(false);
  const [loadFailed, setLoadFailed] = React.useState(false);
  const [search, setSearch] = React.useState("");

  const fetchSessions = React.useCallback(async () => {
    try {
      const res = await listTaggerSessions({ limit: 200 });
      setSessions(res.items.filter((session) => session.phase !== "complete"));
      setLoadFailed(false);
    } catch {
      setLoadFailed(true);
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

  const filteredSessions = React.useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return sessions;
    return sessions.filter((session) => session.name.toLowerCase().includes(query));
  }, [search, sessions]);

  if (!loaded) {
    return (
      <section
        className="flex min-h-64 w-full items-center justify-center"
        aria-live="polite"
        aria-busy="true"
      >
        <div role="status">
          <Loader2 className="size-6 animate-spin text-primary" />
          <span className="sr-only">{msg("tagger.session.loading")}</span>
        </div>
      </section>
    );
  }

  if (loadFailed && sessions.length === 0) {
    return (
      <section className="w-full pb-16">
        <div className="mt-5 rounded-xl border border-dashed border-transparent" aria-live="polite">
          <EmptyState
            icon={CircleAlert}
            iconWrap="tile"
            title={msg("tagger.session.load_failed")}
            description={msg("tagger.session.load_failed_body")}
            action={{ label: msg("tagger.session.retry"), onClick: fetchSessions }}
          />
        </div>
      </section>
    );
  }

  if (sessions.length === 0) {
    return (
      <section className="w-full pb-16">
        <div className="mt-5 rounded-xl border border-dashed border-transparent">
          <EmptyState
            icon={Tags}
            iconWrap="tile"
            title={msg("tagger.session.empty_title")}
            description={msg("tagger.session.empty_body")}
            action={{ label: msg("tagger.session.start_new"), onClick: onStartNew }}
          />
        </div>
      </section>
    );
  }

  return (
    <section className="w-full pb-16">
      <div className="flex items-center gap-2.5">
        <SearchField
          value={search}
          onValueChange={setSearch}
          placeholder={msg("tagger.session.search_placeholder")}
          className="flex-1"
        />
        <Button variant="outline" onClick={onStartNew} className="h-11 shrink-0 rounded-2xl">
          <Plus className="size-4" aria-hidden="true" />
          {msg("tagger.session.start_new")}
        </Button>
      </div>

      <div className="mt-5 rounded-xl border border-dashed border-transparent">
        {filteredSessions.length === 0 ? (
          <EmptyState icon={Search} title={msg("tagger.session.search_empty")} />
        ) : (
          <div className="flex flex-col gap-2.5 p-0.5">
            {filteredSessions.map((session) => (
              <TaggingSessionCard key={session.id} session={session} onChanged={fetchSessions} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
