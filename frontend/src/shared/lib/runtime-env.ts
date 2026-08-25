/**
 * Runtime env shim — lets a single built image be pointed at any backend by
 * setting `API_URL` (and friends) on the running pod.
 *
 * `NEXT_PUBLIC_*` vars are inlined into the JS bundle at build time, so they
 * cannot vary per environment without rebuilding. To work around this:
 *
 * 1. The server component in `app/layout.tsx` reads the runtime env at request
 *    time via `getServerRuntimeEnv()` and injects it as a tiny `<script>` that
 *    sets `window.__SKYNET_ENV__` before hydration.
 * 2. Client code calls `getRuntimeEnv()` which prefers `window.__SKYNET_ENV__`
 *    and falls back to the build-time `NEXT_PUBLIC_*` defaults so `pnpm dev`
 *    keeps working without touching any pod-only state.
 *
 * Note: feature flags such as `NEXT_PUBLIC_FEATURE_GENERALIST_AGENT` stay as
 * `NEXT_PUBLIC_*` on purpose — they're build-time decisions about which
 * features ship in a given image, not runtime-tunable knobs.
 */

export interface RuntimeEnv {
  apiUrl: string;
  appVersion: string;
  transcriptionEnabled: boolean;
}

declare global {
  interface Window {
    __SKYNET_ENV__?: RuntimeEnv;
  }
}

const DEFAULT_API_URL = "http://localhost:8000";
const DEFAULT_APP_VERSION = "0.1.0";

function envEnabled(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes((value ?? "").trim().toLocaleLowerCase());
}

/**
 * Server-side: read `process.env.API_URL` first (the runtime override that
 * Kubernetes sets on the pod), then fall back to `NEXT_PUBLIC_API_URL` so the
 * old build-time default still works when no runtime override is provided.
 */
export function getServerRuntimeEnv(): RuntimeEnv {
  return {
    apiUrl: process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL,
    appVersion:
      process.env.APP_VERSION ?? process.env.NEXT_PUBLIC_APP_VERSION ?? DEFAULT_APP_VERSION,
    transcriptionEnabled: envEnabled(
      process.env.TRANSCRIPTION_ENABLED ?? process.env.NEXT_PUBLIC_TRANSCRIPTION_ENABLED,
    ),
  };
}

/**
 * Universal getter — safe to call from server or client components.
 *
 * On the server it delegates to `getServerRuntimeEnv()`. On the client it
 * reads `window.__SKYNET_ENV__` (injected by the layout) and falls back to
 * the build-time `NEXT_PUBLIC_*` defaults so dev mode without an injected
 * script keeps working.
 */
export function getRuntimeEnv(): RuntimeEnv {
  if (typeof window === "undefined") {
    return getServerRuntimeEnv();
  }
  const injected = window.__SKYNET_ENV__;
  return {
    apiUrl: injected?.apiUrl ?? process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL,
    appVersion:
      injected?.appVersion ?? process.env.NEXT_PUBLIC_APP_VERSION ?? DEFAULT_APP_VERSION,
    transcriptionEnabled:
      injected?.transcriptionEnabled ?? envEnabled(process.env.NEXT_PUBLIC_TRANSCRIPTION_ENABLED),
  };
}

/**
 * Produce the inline `<script>` body that sets `window.__SKYNET_ENV__`.
 *
 * Uses `JSON.parse(JSON.stringify(...))` to keep parsing fast, and escapes any
 * sequence that could prematurely terminate the surrounding `<script>` tag
 * (the same approach as `serialize-javascript` from yahoo). This guards
 * against XSS even if a runtime env value somehow contains markup. The
 * ` ` and ` ` escapes use `RegExp` constructors so the unicode
 * line/paragraph separators don't terminate the source regex literal.
 */
const LINE_SEPARATOR_RE = new RegExp(" ", "g");
const PARAGRAPH_SEPARATOR_RE = new RegExp(" ", "g");

export function serializeRuntimeEnv(env: RuntimeEnv): string {
  const json = JSON.stringify(env)
    .replace(/</g, "\\u003C")
    .replace(/>/g, "\\u003E")
    .replace(/\//g, "\\u002F")
    .replace(LINE_SEPARATOR_RE, "\\u2028")
    .replace(PARAGRAPH_SEPARATOR_RE, "\\u2029");
  return `window.__SKYNET_ENV__=JSON.parse(${JSON.stringify(json)});`;
}
