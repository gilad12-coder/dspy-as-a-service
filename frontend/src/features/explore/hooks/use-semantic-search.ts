"use client";

import * as React from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { searchPublicDashboard, type SearchResult, type SearchSort } from "@/shared/lib/api";

export type ExploreCorpus = "mine" | "public" | "shared";
export type SearchType = "semantic" | "lexical";

export interface SearchQueryState {
  /** The text the user is typing — updates immediately, used to drive the input value. */
  text: string;
  /** Page-size selector. */
  size: number;
  /** 1-indexed page. */
  page: number;
  /** Active categorical filter values. */
  models: string[];
  optimizers: string[];
  types: string[];
  tasks: string[];
  modules: string[];
  /** ISO date (YYYY-MM-DD), inclusive. */
  dateFrom: string | null;
  dateTo: string | null;
  /** Which corpus to search: owned, explicitly shared, or published on-premise runs. */
  corpus: ExploreCorpus;
  /**
   * Result ordering. Resolved to a concrete value (never "auto"): with a
   * query the default is "relevance"; with an empty query it falls back to
   * "recent" since there is nothing to rank by.
   */
  sort: SearchSort;
}

export interface SearchResponseState {
  results: SearchResult[];
  total: number;
  loading: boolean;
  error: string | null;
  /** True when the user has typed anything or applied any filter. */
  isActive: boolean;
  /** Which backend branch served this response — drives the per-row badge. */
  searchType: SearchType | null;
}

const VALID_SIZES = new Set([10, 30, 50]);
const VALID_CORPORA = new Set<ExploreCorpus>(["mine", "public", "shared"]);
const VALID_SORTS = new Set<SearchSort>(["relevance", "recent", "gain"]);

const DEFAULT_SIZE = 30;
const DEFAULT_PAGE = 1;

/**
 * The sort the UI defaults to for a given query. "relevance" only makes
 * sense when there's a query to rank against — the backend rejects an empty
 * query under relevance — so an empty query falls back to "recent".
 */
function autoSort(text: string): SearchSort {
  return text.trim() ? "relevance" : "recent";
}

function parseCsv(value: string | null): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function readState(
  params: URLSearchParams,
  defaultCorpus: ExploreCorpus,
): SearchQueryState {
  const sizeRaw = Number.parseInt(params.get("size") ?? "", 10);
  const pageRaw = Number.parseInt(params.get("page") ?? "", 10);
  const corpusRaw = params.get("corpus") as ExploreCorpus | null;
  const text = params.get("q") ?? "";
  const sortRaw = params.get("sort") as SearchSort | null;
  let sort = sortRaw && VALID_SORTS.has(sortRaw) ? sortRaw : autoSort(text);
  // A stale ``sort=relevance`` left over from a since-cleared query would be
  // rejected by the backend; coerce it back to recency.
  if (!text.trim() && sort === "relevance") sort = "recent";
  return {
    text,
    size: VALID_SIZES.has(sizeRaw) ? sizeRaw : DEFAULT_SIZE,
    page: Number.isFinite(pageRaw) && pageRaw > 0 ? pageRaw : DEFAULT_PAGE,
    models: parseCsv(params.get("models")),
    optimizers: parseCsv(params.get("optimizers")),
    types: parseCsv(params.get("types")),
    tasks: parseCsv(params.get("tasks")),
    modules: parseCsv(params.get("modules")),
    dateFrom: params.get("from"),
    dateTo: params.get("to"),
    corpus:
      corpusRaw && VALID_CORPORA.has(corpusRaw) ? corpusRaw : defaultCorpus,
    sort,
  };
}

function hasAnyFilter(state: SearchQueryState): boolean {
  if (state.text.trim().length > 0) return true;
  if (state.models.length > 0) return true;
  if (state.optimizers.length > 0) return true;
  if (state.types.length > 0) return true;
  if (state.tasks.length > 0) return true;
  if (state.modules.length > 0) return true;
  if (state.dateFrom) return true;
  if (state.dateTo) return true;
  return false;
}

export interface SearchActions {
  setText: (value: string) => void;
  setPage: (page: number) => void;
  setSize: (size: number) => void;
  setSort: (sort: SearchSort) => void;
  setCorpus: (corpus: ExploreCorpus) => void;
  setModels: (models: string[]) => void;
  setOptimizers: (optimizers: string[]) => void;
  setTypes: (types: string[]) => void;
  setTasks: (tasks: string[]) => void;
  setModules: (modules: string[]) => void;
  setDateRange: (from: string | null, to: string | null) => void;
  /** Clears the metadata filters but keeps the free-text query. */
  clearFilters: () => void;
  clearAll: () => void;
}

export interface UseSemanticSearchOptions {
  /** Logged-in user's name; empty string when signed out. Drives Mine corpus fetch. */
  sessionUser: string;
  /**
   * False while next-auth is still resolving the session. Gates the first
   * fetch so a signed-in user never flashes the public corpus before their
   * session-derived default ("mine") is known.
   */
  sessionReady: boolean;
}

export function useSemanticSearch(opts: UseSemanticSearchOptions): {
  query: SearchQueryState;
  response: SearchResponseState;
  actions: SearchActions;
  /** Cleared count for the "X filters active" pill — excludes free-text query. */
  appliedFilterCount: number;
} {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const sessionUser = opts.sessionUser;
  const sessionReady = opts.sessionReady;
  const defaultCorpus: ExploreCorpus = sessionUser ? "mine" : "public";

  const query = React.useMemo<SearchQueryState>(
    () =>
      readState(
        new URLSearchParams(searchParams?.toString() ?? ""),
        defaultCorpus,
      ),
    [searchParams, defaultCorpus],
  );

  const [response, setResponse] = React.useState<SearchResponseState>({
    results: [],
    total: 0,
    loading: false,
    error: null,
    isActive: false,
    searchType: null,
  });

  const updateUrl = React.useCallback(
    (mutator: (params: URLSearchParams) => void, opts: { resetPage?: boolean } = {}) => {
      const params = new URLSearchParams(
        typeof window !== "undefined" ? window.location.search : "",
      );
      mutator(params);
      if (opts.resetPage) params.delete("page");
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [pathname, router],
  );

  const writeList = React.useCallback(
    (key: string, values: string[]) => {
      updateUrl((p) => {
        if (values.length === 0) p.delete(key);
        else p.set(key, values.join(","));
      }, { resetPage: true });
    },
    [updateUrl],
  );

  const actions = React.useMemo<SearchActions>(
    () => ({
      setText: (value: string) =>
        updateUrl((p) => {
          const trimmed = value;
          if (!trimmed) p.delete("q");
          else p.set("q", trimmed);
        }, { resetPage: true }),

      setPage: (page: number) =>
        updateUrl((p) => {
          if (page <= 1) p.delete("page");
          else p.set("page", String(page));
        }),

      setSize: (size: number) =>
        updateUrl((p) => {
          if (size === DEFAULT_SIZE) p.delete("size");
          else p.set("size", String(size));
        }, { resetPage: true }),

      setSort: (sort: SearchSort) =>
        updateUrl((p) => {
          // Only persist a non-default choice so the URL stays clean and the
          // auto relevance↔recent flip keeps working when the query changes.
          if (sort === autoSort(p.get("q") ?? "")) p.delete("sort");
          else p.set("sort", sort);
        }, { resetPage: true }),

      setCorpus: (corpus: ExploreCorpus) =>
        updateUrl((p) => {
          if (corpus === defaultCorpus) p.delete("corpus");
          else p.set("corpus", corpus);
          // Each private corpus has its own filter options, so selections do
          // not carry across tabs.
          for (const key of [
            "models",
            "optimizers",
            "types",
            "tasks",
            "modules",
            "from",
            "to",
          ]) {
            p.delete(key);
          }
        }, { resetPage: true }),

      setModels: (models: string[]) => writeList("models", models),
      setOptimizers: (optimizers: string[]) => writeList("optimizers", optimizers),
      setTypes: (types: string[]) => writeList("types", types),
      setTasks: (tasks: string[]) => writeList("tasks", tasks),
      setModules: (modules: string[]) => writeList("modules", modules),

      setDateRange: (from: string | null, to: string | null) =>
        updateUrl((p) => {
          if (from) p.set("from", from);
          else p.delete("from");
          if (to) p.set("to", to);
          else p.delete("to");
        }, { resetPage: true }),

      clearFilters: () =>
        updateUrl((p) => {
          for (const key of [
            "models",
            "optimizers",
            "types",
            "tasks",
            "modules",
            "from",
            "to",
            "page",
          ]) {
            p.delete(key);
          }
        }),

      clearAll: () =>
        updateUrl((p) => {
          for (const key of [
            "q",
            "models",
            "optimizers",
            "types",
            "tasks",
            "modules",
            "from",
            "to",
            "page",
          ]) {
            p.delete(key);
          }
        }),
    }),
    [defaultCorpus, updateUrl, writeList],
  );

  const appliedFilterCount =
    query.models.length +
    query.optimizers.length +
    query.types.length +
    query.tasks.length +
    query.modules.length +
    (query.dateFrom ? 1 : 0) +
    (query.dateTo ? 1 : 0);

  React.useEffect(() => {
    const active = hasAnyFilter(query);
    setResponse((prev) => ({ ...prev, loading: true, error: null, isActive: active }));
    // Hold until NextAuth resolves so every request carries a private scope.
    if (!sessionReady) return;
    const controller = new AbortController();

    const runSearch = async (
      scope: { owner_username?: string; shared_with_username?: string },
    ) => {
      const data = await searchPublicDashboard(
        {
          query: query.text.trim() || undefined,
          models: query.models.length ? query.models : undefined,
          optimizers: query.optimizers.length ? query.optimizers : undefined,
          optimization_types: query.types.length ? query.types : undefined,
          tasks: query.tasks.length ? query.tasks : undefined,
          modules: query.modules.length ? query.modules : undefined,
          date_from: query.dateFrom ?? undefined,
          date_to: query.dateTo ?? undefined,
          sort: query.sort,
          page: query.page,
          size: query.size,
          owner_username: scope.owner_username,
          shared_with_username: scope.shared_with_username,
        },
        { signal: controller.signal },
      );
      if (controller.signal.aborted) return;
      setResponse({
        results: data.results,
        total: data.total,
        loading: false,
        error: null,
        isActive: active,
        searchType: data.search_type ?? null,
      });
    };

    const runSignedOut = () => {
      setResponse({
        results: [],
        total: 0,
        loading: false,
        error: null,
        isActive: active,
        searchType: null,
      });
    };

    const run = async () => {
      try {
        if (query.corpus === "public") {
          await runSearch({});
        } else if (!sessionUser) {
          runSignedOut();
        } else if (query.corpus === "shared") {
          await runSearch({ shared_with_username: sessionUser });
        } else {
          await runSearch({ owner_username: sessionUser });
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        const name = (err as Error | undefined)?.name;
        if (name === "AbortError") return;
        setResponse((prev) => ({
          ...prev,
          loading: false,
          error: err instanceof Error ? err.message : "search failed",
        }));
      }
    };

    // Typing is already debounced upstream in SearchBar, and every other
    // trigger (filters, corpus, sort, page) is a discrete click — so fire the
    // request immediately and rely on AbortController to drop the stale one.
    void run();
    return () => {
      controller.abort();
    };
  }, [query, sessionUser, sessionReady]);

  return { query, response, actions, appliedFilterCount };
}
