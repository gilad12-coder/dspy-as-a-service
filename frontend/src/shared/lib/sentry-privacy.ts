/** Privacy filters and bounded sampling shared by every Sentry runtime. */

interface SanitizableEvent {
  user?: unknown;
  request?: {
    url?: string;
    headers?: unknown;
    cookies?: unknown;
    data?: unknown;
  };
}

/** Remove user identity, request bodies, cookies, headers, and URL queries. */
export function scrubSentryEvent<T extends SanitizableEvent>(event: T): T {
  event.user = undefined;
  if (event.request) {
    event.request.headers = undefined;
    event.request.cookies = undefined;
    event.request.data = undefined;
    if (event.request.url) {
      try {
        const url = new URL(event.request.url);
        event.request.url = `${url.origin}${url.pathname}`;
      } catch {
        event.request.url = event.request.url.split(/[?#]/, 1)[0];
      }
    }
  }
  return event;
}

/** Parse an environment sample rate and clamp invalid values to five percent. */
export function sentryTraceSampleRate(raw: string | undefined): number {
  const parsed = Number(raw ?? "0.05");
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : 0.05;
}
