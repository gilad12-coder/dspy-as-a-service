import { useCallback, useEffect, useState, type MutableRefObject } from "react";
import { getModelCatalog, cachedCatalog } from "@/shared/lib/model-catalog";
import { profileDataset } from "@/shared/lib/api";
import type {
  ColumnMapping,
  DatasetProfile,
  ModelCatalogResponse,
  ModelConfig,
  SplitFractions,
  SplitPlan,
} from "@/shared/types/api";
import type { ParsedDataset } from "@/shared/lib/parse-dataset";
import { MAX_RECENT, RECENT_KEY } from "../constants";
import type { ColumnRole } from "../constants";

export function buildColumnMapping(columnRoles: Record<string, ColumnRole>): ColumnMapping {
  const inputs: Record<string, string> = {};
  const outputs: Record<string, string> = {};
  Object.entries(columnRoles).forEach(([col, role]) => {
    if (role === "input") inputs[col] = col;
    else if (role === "output") outputs[col] = col;
  });
  return { inputs, outputs };
}

export function useRecentModelConfigs() {
  // Initial state must be empty: this hook is client-only but Next.js still
  // renders client components on the server. Touching `localStorage` in the
  // initializer throws ReferenceError under SSR, the try/catch falls back
  // to [], and the UI hydrates without any recent configs. Hydrate via an
  // effect instead.
  const [recentConfigs, setRecentConfigs] = useState<ModelConfig[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(RECENT_KEY);
      if (stored) setRecentConfigs(JSON.parse(stored));
    } catch {
      // Private-mode Safari / disabled storage / corrupt JSON — keep [].
    }
  }, []);

  const saveToRecent = useCallback((config: ModelConfig) => {
    if (!config.name) return;
    // Legacy inline connection fields are never persisted; BYOK references
    // the encrypted Settings → Providers vault by slug instead.
    const {
      api_key: _ApiKey,
      api_base: _ApiBase,
      base_url: _ExtraBaseUrl,
      ...safeExtra
    } = config.extra ?? {};
    const { base_url: _baseUrl, ...safeFields } = config;
    const safeConfig: ModelConfig = {
      ...safeFields,
      extra: Object.keys(safeExtra).length > 0 ? safeExtra : undefined,
    };
    setRecentConfigs((prev) => {
      const deduped = prev.filter((c) => c.name !== safeConfig.name);
      const next = [safeConfig, ...deduped].slice(0, MAX_RECENT);
      try {
        localStorage.setItem(RECENT_KEY, JSON.stringify(next));
      } catch {
        // Quota exceeded or storage disabled — in-memory list still updates.
      }
      return next;
    });
  }, []);

  const clearRecentConfigs = useCallback(() => {
    setRecentConfigs([]);
    try {
      localStorage.removeItem(RECENT_KEY);
    } catch {
      // Storage disabled — already cleared in-memory.
    }
  }, []);

  const removeRecentConfig = useCallback((name: string) => {
    setRecentConfigs((prev) => {
      const next = prev.filter((c) => c.name !== name);
      try {
        if (next.length === 0) localStorage.removeItem(RECENT_KEY);
        else localStorage.setItem(RECENT_KEY, JSON.stringify(next));
      } catch {
        // Storage disabled — in-memory list is still updated.
      }
      return next;
    });
  }, []);

  return { recentConfigs, saveToRecent, clearRecentConfigs, removeRecentConfig };
}

export function useModelCatalog() {
  const [catalog, setCatalog] = useState<ModelCatalogResponse | null>(cachedCatalog);

  useEffect(() => {
    if (catalog) return;
    let cancelled = false;
    getModelCatalog()
      .then((c) => {
        if (!cancelled) setCatalog(c);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [catalog]);

  return catalog;
}

export function useDatasetProfiling({
  parsedDataset,
  columnRoles,
  splitModeRef,
  setDatasetProfile,
  setSplitPlan,
  setProfileLoading,
  setSplit,
  setShuffle,
  setSeed,
}: {
  parsedDataset: ParsedDataset | null;
  columnRoles: Record<string, ColumnRole>;
  splitModeRef: MutableRefObject<"auto" | "manual">;
  setDatasetProfile: (profile: DatasetProfile | null) => void;
  setSplitPlan: (plan: SplitPlan | null) => void;
  setProfileLoading: (loading: boolean) => void;
  setSplit: (split: SplitFractions) => void;
  setShuffle: (shuffle: boolean) => void;
  setSeed: (seed: number | undefined) => void;
}) {
  useEffect(() => {
    if (!parsedDataset || parsedDataset.rowCount === 0) return;
    const mapping = buildColumnMapping(columnRoles);
    if (Object.keys(mapping.inputs).length === 0) return;

    let cancelled = false;
    const handle = setTimeout(() => {
      setProfileLoading(true);
      profileDataset({
        dataset: parsedDataset.rows as Array<Record<string, unknown>>,
        column_mapping: mapping,
      })
        .then((response) => {
          if (cancelled) return;
          setDatasetProfile(response.profile);
          setSplitPlan(response.plan);
          if (splitModeRef.current === "auto") {
            setSplit(response.plan.fractions);
            setShuffle(response.plan.shuffle);
            setSeed(response.plan.seed);
          }
        })
        .catch(() => {
          /* non-blocking: manual controls still work */
        })
        .finally(() => {
          if (!cancelled) setProfileLoading(false);
        });
    }, 400);

    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [
    columnRoles,
    parsedDataset,
    setDatasetProfile,
    setProfileLoading,
    setSeed,
    setShuffle,
    setSplit,
    setSplitPlan,
    splitModeRef,
  ]);
}
