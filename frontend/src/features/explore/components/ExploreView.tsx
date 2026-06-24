"use client";

import * as React from "react";
import { useSession } from "next-auth/react";
import { motion } from "framer-motion";
import { AlertTriangle, Clock, FilterX, LogIn, Plus, SearchX, Send } from "lucide-react";
import { logSearchQuery, type PublicDashboardPoint } from "@/shared/lib/api";
import { msg, formatMsg } from "@/shared/lib/messages";
import { EmptyState } from "@/shared/ui/empty-state";
import { registerTutorialHook } from "@/features/tutorial";
import { usePublicDashboard } from "../hooks/use-public-dashboard";
import { useCorpusFacets } from "../hooks/use-corpus-facets";
import { useSemanticSearch } from "../hooks/use-semantic-search";
import { useRecentQueries } from "../hooks/use-recent-queries";
import { usePopularQueries } from "../hooks/use-popular-queries";
import { useResultKeyboardNav } from "../hooks/use-result-keyboard-nav";
import { ExploreSkeleton } from "./ExploreSkeleton";
import { SearchBar } from "./SearchBar";
import { FiltersDrawer } from "./FiltersDrawer";
import { ResultsList } from "./ResultsList";
import { ResultsToolbar } from "./ResultsToolbar";
import { ResultsSkeleton } from "./ResultsSkeleton";
import { Pagination } from "./Pagination";

/**
 * Top-level /explore page rendering a single ranked-list view driven by one
 * shared search input and filter set, with corpus toggle and pagination.
 */
export function ExploreView() {
  const { data: session, status } = useSession();
  const sessionUser = session?.user?.name ?? "";
  const { points: realPoints, loading: corpusLoading, error: corpusError } = usePublicDashboard();
  const [demoPoints, setDemoPoints] = React.useState<PublicDashboardPoint[] | null>(null);
  const points = demoPoints ?? realPoints;

  React.useEffect(() => registerTutorialHook("setDemoExplorePoints", setDemoPoints), []);
  React.useEffect(() => {
    const onExit = () => setDemoPoints(null);
    window.addEventListener("tutorial-exited", onExit);
    return () => window.removeEventListener("tutorial-exited", onExit);
  }, []);

  const { query, response, actions, appliedFilterCount } = useSemanticSearch({
    sessionUser,
    sessionReady: status !== "loading",
  });
  const [drawerOpen, setDrawerOpen] = React.useState(false);

  const { recent, push: pushRecent, clear: clearRecent } = useRecentQueries();

  // A query is recorded only when it leads to an opened optimization — a result
  // row clicked, or Enter pressed on a keyboard-highlighted row — never on a
  // bare Enter-to-search or debounced typing. Tying the signal to a click-through
  // keeps idle or mistyped queries out of recent and the trending counts. Recent
  // is personal and per-device, recorded for any corpus; trending is public-corpus
  // only and logged server-side. The consecutive-dedup ref guards against a
  // keyboard-open and the row's own click handler double-counting the same query.
  const lastLoggedRef = React.useRef("");
  const commitQuery = React.useCallback(
    (raw: string) => {
      const trimmed = raw.trim();
      if (!trimmed) return;
      pushRecent(trimmed);
      if (query.corpus !== "public") return;
      const normalized = trimmed.toLowerCase();
      if (normalized.length < 2 || normalized === lastLoggedRef.current) return;
      lastLoggedRef.current = normalized;
      logSearchQuery(trimmed);
    },
    [query.corpus, pushRecent],
  );

  const { activeIndex, onInputKeyDown } = useResultKeyboardNav(
    response.results,
    () => commitQuery(query.text),
  );

  // Filter options come from a per-corpus facets fetch so each tab lists only
  // the chips it can filter to (a model private to "mine" never shows under
  // "public"). The tutorial's demo corpus has no backend scope, so there we
  // fall back to deriving options from the injected demo points.
  const facets = useCorpusFacets(query.corpus, sessionUser);
  const modelOptions = React.useMemo(
    () => (demoPoints ? collectDistinct(demoPoints, "winning_model") : facets.models),
    [demoPoints, facets.models],
  );
  const optimizerOptions = React.useMemo(
    () => (demoPoints ? collectDistinct(demoPoints, "optimizer_name") : facets.optimizers),
    [demoPoints, facets.optimizers],
  );
  const moduleOptions = React.useMemo(
    () => (demoPoints ? collectDistinct(demoPoints, "module_name") : facets.modules),
    [demoPoints, facets.modules],
  );
  // Popular searches for a blank field: real trending only — what people
  // actually searched (public corpus, logged server-side on explicit commit).
  // When the log has no data yet, this is empty and the section simply doesn't
  // render; showing nothing beats surfacing irrelevant filler.
  const trendingQueries = usePopularQueries();
  const popularSearches = React.useMemo<string[]>(
    () => trendingQueries.map((q) => q.query),
    [trendingQueries],
  );

  const corpusTotal = points.length;
  const isPublicCorpus = query.corpus === "public";
  // The dashed empty state only fires when the public corpus is genuinely
  // empty — we still want the corpus toggle visible so the user can pivot
  // to "Mine" without first creating a public job.
  const isTrulyEmpty =
    isPublicCorpus && !corpusLoading && !corpusError && corpusTotal === 0;

  // Until the session resolves we don't yet know the default corpus (mine when
  // signed in, public when anonymous); show the skeleton rather than briefly
  // mounting the wrong tab and flashing its results.
  if (status === "loading") {
    return <ExploreSkeleton />;
  }

  if (isPublicCorpus && corpusLoading && points.length === 0) {
    return <ExploreSkeleton />;
  }

  return (
    <div className="pb-16">
      <div className="flex flex-col gap-1.5">
        {isPublicCorpus && corpusError && (
          <div
            className="flex items-start gap-3 rounded-lg border border-border bg-accent-muted/50 px-4 py-3 text-xs text-foreground"
            role="status"
          >
            <AlertTriangle
              className="mt-0.5 size-4 shrink-0 text-muted-foreground"
              aria-hidden="true"
            />
            <span>{corpusError}</span>
          </div>
        )}

        <div className="flex flex-col gap-3">
          <SearchBar
            text={query.text}
            onSubmit={actions.setText}
            corpus={query.corpus}
            onCorpusChange={actions.setCorpus}
            signedIn={sessionUser.length > 0}
            filtersCount={appliedFilterCount}
            onOpenFilters={() => setDrawerOpen(true)}
            onClearFilters={actions.clearFilters}
            loading={response.loading}
            onResultKeyDown={onInputKeyDown}
            activeResultIndex={activeIndex}
            recentQueries={recent}
            onClearRecent={clearRecent}
            suggestions={isPublicCorpus ? popularSearches : []}
          />
        </div>

        {isTrulyEmpty ? (
          <EmptyState
            variant="list"
            icon={Send}
            title={msg("explore.empty.title")}
            description={msg("explore.empty.hint")}
            action={{ label: msg("explore.empty.cta"), href: "/submit", icon: Plus }}
            className="min-h-[40vh] justify-center"
          />
        ) : (
          <ListPane
            query={query}
            response={response}
            activeIndex={activeIndex}
            onSetPage={actions.setPage}
            onSetSort={actions.setSort}
            onClearAll={actions.clearAll}
            onClearQuery={() => actions.setText("")}
            onResultOpen={() => commitQuery(query.text)}
            hasFilters={appliedFilterCount > 0}
            sessionUser={sessionUser}
          />
        )}
      </div>

      <FiltersDrawer
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        modelOptions={modelOptions}
        optimizerOptions={optimizerOptions}
        moduleOptions={moduleOptions}
        selectedModels={query.models}
        selectedOptimizers={query.optimizers}
        selectedTypes={query.types}
        selectedModules={query.modules}
        dateFrom={query.dateFrom}
        dateTo={query.dateTo}
        onChangeModels={actions.setModels}
        onChangeOptimizers={actions.setOptimizers}
        onChangeTypes={actions.setTypes}
        onChangeModules={actions.setModules}
        onChangeDateRange={actions.setDateRange}
        onClearAll={actions.clearAll}
      />
    </div>
  );
}

function ListPane({
  query,
  response,
  activeIndex,
  onSetPage,
  onSetSort,
  onClearAll,
  onClearQuery,
  onResultOpen,
  hasFilters,
  sessionUser,
}: {
  query: ReturnType<typeof useSemanticSearch>["query"];
  response: ReturnType<typeof useSemanticSearch>["response"];
  activeIndex: number;
  onSetPage: ReturnType<typeof useSemanticSearch>["actions"]["setPage"];
  onSetSort: ReturnType<typeof useSemanticSearch>["actions"]["setSort"];
  onClearAll: () => void;
  onClearQuery: () => void;
  onResultOpen: () => void;
  hasFilters: boolean;
  sessionUser: string;
}) {
  if (response.error) {
    return (
      <div
        role="status"
        className="mx-auto flex max-w-2xl items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-[13px] text-destructive"
      >
        <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
        <span>{msg("explore.results.error")}</span>
      </div>
    );
  }

  if (response.loading && response.results.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <div className="border-t border-border/55">
          <ResultsSkeleton rows={4} />
        </div>
      </div>
    );
  }

  if (!response.loading && response.results.length === 0) {
    // Distinct cases to keep the empty UI honest:
    //   1. Session-scoped (Mine/Shared) + signed out — no auth, no list to show
    //   2. Mine + no filters — user has zero jobs; clear-filters is misleading
    //   3. Shared + no filters — nothing has been shared with the user yet
    //   4. Otherwise — a real "no matches" state with clear-filters affordance
    const isMine = query.corpus === "mine";
    const isShared = query.corpus === "shared";
    if ((isMine || isShared) && !sessionUser) {
      return (
        <EmptyState
          variant="list"
          icon={LogIn}
          title={msg(
            isShared ? "explore.corpus.shared.signed_out" : "explore.corpus.mine.signed_out",
          )}
        />
      );
    }
    if (isMine && !response.isActive) {
      return (
        <EmptyState
          variant="list"
          icon={Send}
          title={msg("explore.corpus.mine.empty")}
          description={msg("explore.corpus.mine.empty.hint")}
          action={{ label: msg("explore.empty.cta"), href: "/submit", icon: Plus }}
        />
      );
    }
    if (isShared && !response.isActive) {
      return (
        <EmptyState
          variant="list"
          title={msg("explore.corpus.shared.empty")}
          description={msg("explore.corpus.shared.empty.hint")}
          className="pt-4"
        />
      );
    }
    return (
      <EmptyState
        variant="list"
        icon={SearchX}
        title={formatMsg("explore.results.empty.title", { query: query.text || "—" })}
        description={msg("explore.results.empty.hint")}
      >
        {(query.text.trim().length > 0 || hasFilters) && (
          <div className="flex flex-wrap items-center justify-center gap-2">
            {query.text.trim().length > 0 && (
              <button
                type="button"
                onClick={onClearQuery}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3.5 py-2 text-[12.5px] text-foreground/75 transition-colors cursor-pointer hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
              >
                <Clock className="size-3.5" aria-hidden="true" />
                {msg("explore.results.empty.show_recent")}
              </button>
            )}
            {hasFilters && (
              <button
                type="button"
                onClick={onClearAll}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3.5 py-2 text-[12.5px] text-foreground/75 transition-colors cursor-pointer hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
              >
                <FilterX className="size-3.5" aria-hidden="true" />
                {msg("explore.results.empty.clear_filters")}
              </button>
            )}
          </div>
        )}
      </EmptyState>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <ResultsToolbar
        total={response.total}
        sort={query.sort}
        onSortChange={onSetSort}
        hasQuery={query.text.trim().length > 0}
      />
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
        className="border-t border-border/55"
      >
        <ResultsList
          results={response.results}
          highlight={query.text}
          searchType={response.searchType}
          activeIndex={activeIndex}
          onResultOpen={onResultOpen}
        />
      </motion.div>
      <Pagination
        page={query.page}
        size={query.size}
        total={response.total}
        onPageChange={onSetPage}
      />
    </div>
  );
}

function collectDistinct(
  points: PublicDashboardPoint[],
  key: "winning_model" | "optimizer_name" | "module_name",
): string[] {
  const set = new Set<string>();
  for (const p of points) {
    const v = p[key];
    if (typeof v === "string" && v.length > 0) set.add(v);
  }
  return Array.from(set).sort((a, b) => a.localeCompare(b));
}
