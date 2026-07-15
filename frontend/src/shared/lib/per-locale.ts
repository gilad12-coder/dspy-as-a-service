/**
 * Locale-faithful views over i18n-derived module constants.
 *
 * A `const LABELS = { x: msg("…") }` at module scope resolves its strings while
 * the bundle evaluates — in the browser that can happen before the catalog shim
 * runs (freezing raw keys like `auto.features….literal.1` into the constant for
 * the whole session), and on the server it happens once per process with
 * whichever request's catalog was pinned at import time (freezing one request's
 * language into every later request, regardless of that request's locale).
 *
 * `perLocale` keeps the declaration shape — call sites keep reading a plain
 * object/array — but defers the build to first access and re-runs it whenever
 * the active locale or catalog changes, so reads always reflect the current
 * request/render. This is the message-catalog counterpart of the `TERMS` Proxy
 * in `terms.ts`, which fixed the same freeze for bare glossary reads.
 */

import { getActiveLocale } from "@/shared/lib/runtime-locale";
import { getActiveMessages } from "@/shared/lib/runtime-messages";

/**
 * Wrap an i18n-dependent module constant so it resolves at access time.
 *
 * The returned proxy rebuilds `build()` lazily, memoized on the active
 * (locale, catalog) pair: one build per request on the server, one per session
 * in the browser once the catalog shim has landed. Before the shim lands the
 * catalog identity changes per read, so accesses rebuild transiently instead
 * of caching raw keys — mirroring `getActiveMessages()`'s own degrade rule.
 *
 * Property reads, `in`, spread, iteration, and `Object.keys` all forward to
 * the current build. Descriptors are reported configurable so array shapes
 * (whose `length` is non-configurable) don't violate proxy invariants against
 * the throwaway target.
 */
export function perLocale<T extends object>(build: () => T): T {
  let cached: { locale: string; catalog: unknown; value: T } | null = null;
  const resolve = (): T => {
    const locale = getActiveLocale();
    const catalog = getActiveMessages();
    if (!cached || cached.locale !== locale || cached.catalog !== catalog) {
      cached = { locale, catalog, value: build() };
    }
    return cached.value;
  };
  return new Proxy({} as T, {
    get(_target, key) {
      const value = resolve();
      return Reflect.get(value, key, value);
    },
    has: (_target, key) => Reflect.has(resolve(), key),
    ownKeys: () => Reflect.ownKeys(resolve()),
    getOwnPropertyDescriptor(_target, key) {
      const desc = Object.getOwnPropertyDescriptor(resolve(), key);
      return desc ? { ...desc, configurable: true } : undefined;
    },
  });
}
