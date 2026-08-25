"use client";

import * as React from "react";
import { CircleNotch, MagnifyingGlass, Plus, Tag, WarningCircle } from "@/shared/ui/icons";
import { toast } from "react-toastify";

import {
  bulkDeleteTaggerSessions,
  listTaggerSessions,
  type TaggerSessionSummary,
} from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { Button } from "@/shared/ui/primitives/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { EmptyState } from "@/shared/ui/empty-state";
import { ListPageSkeleton } from "@/shared/ui/list-page-skeleton";
import { SearchField } from "@/shared/ui/search-field";
import { SelectionBar } from "@/shared/ui/selection-bar";
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
  const [loadingSessions, setLoadingSessions] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set());
  // Last-toggled session id — the shift-click range anchor, kept as an id (not
  // an index) so it survives the search filter reordering the visible list.
  const [anchorId, setAnchorId] = React.useState<string | null>(null);
  const [bulkOpen, setBulkOpen] = React.useState(false);
  const [bulkDeleting, setBulkDeleting] = React.useState(false);

  const fetchSessions = React.useCallback(async () => {
    setLoadingSessions(true);
    try {
      const res = await listTaggerSessions({ limit: 200 });
      const items = res.items.filter((session) => session.phase !== "complete");
      setSessions(items);
      // Drop selections that no longer resolve to a listed owned session
      // (deleted elsewhere, completed since, or shared-in and therefore not
      // bulk-deletable), so the bar never counts ghosts.
      setSelectedIds((prev) => {
        const next = new Set(
          [...prev].filter((id) => items.some((s) => s.id === id && s.role === "owner")),
        );
        return next.size === prev.size ? prev : next;
      });
      setLoadFailed(false);
    } catch {
      setLoadFailed(true);
    } finally {
      setLoaded(true);
      setLoadingSessions(false);
    }
  }, []);

  const confirmBulkDelete = async () => {
    if (bulkDeleting) return;
    setBulkDeleting(true);
    try {
      const res = await bulkDeleteTaggerSessions([...selectedIds]);
      if (res.skipped.length > 0) {
        toast.warn(formatMsg("shared.selection.delete_skipped", { count: res.skipped.length }));
      }
      setBulkOpen(false);
      setSelectedIds(new Set());
      setAnchorId(null);
      window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("datasets.toast.delete_failed"));
    } finally {
      setBulkDeleting(false);
    }
  };

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

  // Same mechanism as the storage cleanup drawer: plain click toggles one row,
  // shift-click applies the clicked row's new state to the whole visible range
  // between it and the previous toggle. Shared-in rows inside a range are
  // skipped — only owned sessions can be bulk-deleted.
  const toggleSelected = React.useCallback(
    (id: string, shiftKey: boolean) => {
      const visible = filteredSessions;
      setSelectedIds((prev) => {
        const next = new Set(prev);
        const willSelect = !prev.has(id);
        const index = visible.findIndex((s) => s.id === id);
        const anchor = anchorId === null ? -1 : visible.findIndex((s) => s.id === anchorId);
        if (shiftKey && anchor !== -1 && index !== -1) {
          const [lo, hi] = anchor < index ? [anchor, index] : [index, anchor];
          for (let i = lo; i <= hi; i++) {
            const row = visible[i];
            if (!row || row.role !== "owner") continue;
            if (willSelect) next.add(row.id);
            else next.delete(row.id);
          }
        } else if (willSelect) {
          next.add(id);
        } else {
          next.delete(id);
        }
        return next;
      });
      setAnchorId(id);
    },
    [filteredSessions, anchorId],
  );

  if (!loaded) {
    return (
      <section className="w-full pb-16" aria-busy="true">
        <ListPageSkeleton />
        <span className="sr-only" role="status">
          {msg("tagger.session.loading")}
        </span>
      </section>
    );
  }

  if (loadFailed && sessions.length === 0) {
    return (
      <section className="w-full pb-16">
        <div className="mt-5 rounded-xl border border-dashed border-transparent" aria-live="polite">
          <EmptyState
            icon={WarningCircle}
            iconWrap="tile"
            title={msg("tagger.session.load_failed")}
            description={msg("tagger.session.load_failed_body")}
            action={{
              label: msg("tagger.session.retry"),
              onClick: fetchSessions,
              iconOnly: true,
              loading: loadingSessions,
            }}
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
            icon={Tag}
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
      <div className="flex flex-col gap-2.5 sm:flex-row sm:items-center">
        <SearchField
          value={search}
          onValueChange={setSearch}
          placeholder={msg("tagger.session.search_placeholder")}
          className="flex-1 !h-[44px] [&_input]:h-full max-lg:[&_button]:!size-[44px] lg:!h-11"
        />
        <Button
          variant="outline"
          onClick={onStartNew}
          className="!h-[44px] w-full shrink-0 rounded-2xl sm:w-auto"
        >
          <Plus className="size-4" aria-hidden="true" />
          {msg("tagger.session.start_new")}
        </Button>
      </div>

      <div className="mt-5 rounded-xl border border-dashed border-transparent">
        {filteredSessions.length === 0 ? (
          <EmptyState icon={MagnifyingGlass} title={msg("tagger.session.search_empty")} />
        ) : (
          <div className="flex flex-col gap-2.5 p-0.5">
            {filteredSessions.map((session) => (
              <TaggingSessionCard
                key={session.id}
                session={session}
                onChanged={fetchSessions}
                selected={selectedIds.has(session.id)}
                onToggleSelect={(shiftKey) => toggleSelected(session.id, shiftKey)}
              />
            ))}
          </div>
        )}
      </div>

      <SelectionBar
        count={selectedIds.size}
        onClear={() => {
          setSelectedIds(new Set());
          setAnchorId(null);
        }}
        onDelete={() => setBulkOpen(true)}
      />

      <Dialog open={bulkOpen} onOpenChange={setBulkOpen}>
        <DialogContent
          className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md"
          showCloseButton={false}
        >
          <DialogHeader>
            <DialogTitle>{msg("tagger.session.bulk_delete_title")}</DialogTitle>
            <DialogDescription>
              {formatMsg("tagger.session.bulk_delete_body", { count: selectedIds.size })}{" "}
              {msg("delete.irreversible")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-2 gap-3">
            <Button
              variant="outline"
              onClick={() => setBulkOpen(false)}
              disabled={bulkDeleting}
              className="w-full justify-center"
            >
              {msg("datasets.delete.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={confirmBulkDelete}
              disabled={bulkDeleting}
              className="w-full justify-center shadow-xs"
            >
              {bulkDeleting ? (
                <CircleNotch className="size-4 animate-spin" />
              ) : (
                msg("datasets.delete.confirm")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
