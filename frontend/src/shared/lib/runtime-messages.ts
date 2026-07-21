/**
 * Sync active-catalog resolution for `msg()`, across the RSC server/client
 * boundary — the message-catalog counterpart to `runtime-locale.ts`.
 *
 * `msg("some.key")` resolves a string from the *active* locale's catalog with no
 * argument, so the catalog has to be ambient. To ship only the active fallback
 * chain (not all locales), the catalog is delivered out of band rather than
 * statically imported into the client bundle. There are three reader contexts,
 * each fed from a single per-request catalog the root layout builds:
 *
 * - **Browser**: the `window.__SKYNET_MESSAGES__` shim the layout injects before
 *   hydration, so module-scope `msg()` calls (resolved while the bundle evaluates,
 *   ahead of any React render) can find their strings. Mirrored in a module global
 *   and refreshed whenever navigation or Fast Refresh injects a newer catalog; a
 *   call made before the shim runs degrades transiently rather than freezing `{}`.
 * - **Server Components** (RSC graph): a per-request slot held by React's
 *   `cache()`, pinned once at the top of the root layout — race-free across
 *   concurrent requests.
 * - **SSR of client components** (the *client* module graph, where `window` is
 *   undefined and the `cache()` slot lives in the other graph and reads empty):
 *   a request-scoped `globalThis` value the layout publishes. Being process-wide
 *   it carries the same concurrency caveat as `clientLocale` — a mixed-locale
 *   render could briefly read another request's catalog — but the per-request
 *   window shim corrects the browser on hydration. This replaces threading the
 *   whole catalog through a `LocaleProvider` prop, which duplicated it in the
 *   serialized Flight payload.
 *
 * The merge across the fallback chain happens once, server-side (see
 * `messages.server.ts`); the value here is already a flat, complete map.
 */

import { cache } from "react";

export type UiCatalog = Record<string, string>;

declare global {
  interface Window {
    __SKYNET_MESSAGES__?: UiCatalog;
  }
  // Request-scoped catalog for SSR of client components (see module docstring).
  var __SKYNET_REQUEST_MESSAGES__: UiCatalog | undefined;
}

let clientMessages: UiCatalog | null = null;

// Lazily created so cache() is only ever invoked during a server render, never
// when this module loads in the client bundle. Mirrors runtime-locale.ts. The
// slot defaults to null (not {}) so the *client* graph's empty slot is
// distinguishable from a populated one and falls through to the request global.
let serverSlot: (() => { current: UiCatalog | null }) | null = null;

function getServerSlot(): { current: UiCatalog | null } {
  serverSlot ??= cache((): { current: UiCatalog | null } => ({ current: null }));
  return serverSlot();
}

/**
 * Pin the active catalog for the current server request. Call once at the top of
 * the root layout (and `generateMetadata`) before any descendant resolves a
 * message. Sets both the RSC-graph `cache()` slot (read by Server Components,
 * race-free) and the `globalThis` value SSR of client components reads.
 */
export function setServerMessages(catalog: UiCatalog): void {
  getServerSlot().current = catalog;
  globalThis.__SKYNET_REQUEST_MESSAGES__ = catalog;
}

/**
 * Resolve the catalog every `msg()` call should read from.
 *
 * Branches by bundle, exactly like `getActiveLocale()`: when `window` is
 * undefined, use the per-request server slot (populated in the RSC graph) or
 * the request `globalThis` SSR fallback. In the browser, adopt the current
 * injected `window.__SKYNET_MESSAGES__` object whenever it changes.
 */
export function getActiveMessages(): UiCatalog {
  if (typeof window === "undefined") {
    const slot = getServerSlot().current;
    if (slot !== null) return slot;
    return globalThis.__SKYNET_REQUEST_MESSAGES__ ?? {};
  }
  // The beforeInteractive shim can execute *after* a module chunk has already
  // run a module-scope msg() — the webpack prod build orders some chunk <script>s
  // ahead of the inline shim, so window.__SKYNET_MESSAGES__ is still undefined at
  // that first call. Caching {} here would then freeze every later lookup to its
  // raw key across the whole app. Only cache once the shim has populated the
  // global; until then degrade transiently so the first post-hydration read
  // recovers the real catalog.
  const injected = window.__SKYNET_MESSAGES__;
  if (injected !== undefined && injected !== clientMessages) clientMessages = injected;
  return clientMessages ?? {};
}

/**
 * Inline `<script>` body that sets `window.__SKYNET_MESSAGES__` before
 * hydration. Unlike the locale shim, the catalog carries arbitrary user-facing
 * copy, so `<` is escaped to `<` to keep a value from closing the
 * surrounding `<script>` (same guard the JSON-LD block uses).
 */
export function serializeMessages(catalog: UiCatalog): string {
  return `window.__SKYNET_MESSAGES__=${JSON.stringify(catalog).replace(/</g, "\\u003c")};`;
}
