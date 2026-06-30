import type { ModelCatalogResponse, DiscoverModelsResponse } from "@/shared/types/api";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";

// Resolve lazily — a module-load const races the injected window.__SKYNET_ENV__
// and freezes the build-time localhost fallback. See shared/lib/api.ts.
const apiBase = () => getRuntimeEnv().apiUrl;
const LS_KEY = "skynet:model-catalog";
const LS_TTL = 10 * 60 * 1000;

const EMPTY_CATALOG: ModelCatalogResponse = { providers: [], models: [] };

function isModelCatalogResponse(value: unknown): value is ModelCatalogResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<ModelCatalogResponse>;
  return Array.isArray(candidate.providers) && Array.isArray(candidate.models);
}

let _cache: ModelCatalogResponse | null = null;
try {
  const raw = typeof window !== "undefined" ? localStorage.getItem(LS_KEY) : null;
  if (raw) {
    const parsed = JSON.parse(raw) as { ts?: unknown; data?: unknown };
    if (
      typeof parsed.ts === "number" &&
      Date.now() - parsed.ts < LS_TTL &&
      isModelCatalogResponse(parsed.data)
    ) {
      _cache = parsed.data;
    }
  }
} catch {
  /* ignore parse errors */
}

const _ready: Promise<ModelCatalogResponse> =
  typeof window === "undefined"
    ? Promise.resolve(_cache ?? EMPTY_CATALOG)
    : (async () => {
        try {
          const res = await fetch(`${apiBase()}/models`);
          if (!res.ok) throw new Error(`Server error: ${res.status}`);
          const data: unknown = await res.json();
          if (!isModelCatalogResponse(data)) throw new Error("Invalid model catalog response");
          _cache = data;
          try {
            localStorage.setItem(LS_KEY, JSON.stringify({ data, ts: Date.now() }));
          } catch {}
          return data;
        } catch {
          if (_cache) return _cache;
          _cache = EMPTY_CATALOG;
          return EMPTY_CATALOG;
        }
      })();

export function getModelCatalog(): Promise<ModelCatalogResponse> {
  return _ready;
}

/** Synchronously inspect the cached catalog (returns null only before first fetch resolves). */
export function cachedCatalog(): ModelCatalogResponse | null {
  return _cache;
}

// BYOK catalog — every offered provider's models, regardless of platform keys
// (a BYOK run pays with the user's own key). Fetched lazily on first need (only
// in BYOK mode), not at module load, and held in memory for the session.
let _byokCache: ModelCatalogResponse | null = null;
let _byokReady: Promise<ModelCatalogResponse> | null = null;

export function getByokModelCatalog(): Promise<ModelCatalogResponse> {
  if (_byokReady) return _byokReady;
  if (typeof window === "undefined") return Promise.resolve(EMPTY_CATALOG);
  _byokReady = (async () => {
    try {
      const res = await fetch(`${apiBase()}/models/byok`);
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data: unknown = await res.json();
      if (!isModelCatalogResponse(data)) throw new Error("Invalid BYOK catalog response");
      _byokCache = data;
      return data;
    } catch {
      // Reset so a transient failure can retry on the next BYOK switch.
      _byokReady = null;
      return _byokCache ?? EMPTY_CATALOG;
    }
  })();
  return _byokReady;
}

/** Synchronously inspect the cached BYOK catalog (null before its first fetch resolves). */
export function cachedByokCatalog(): ModelCatalogResponse | null {
  return _byokCache;
}

export async function discoverModels(
  baseUrl: string,
  apiKey?: string,
  signal?: AbortSignal,
): Promise<DiscoverModelsResponse> {
  const res = await fetch(`${apiBase()}/models/discover`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_url: baseUrl, api_key: apiKey || undefined }),
    signal,
  });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.json();
}
