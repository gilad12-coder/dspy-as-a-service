/** Register Sentry for the active Next.js runtime and capture request errors. */

import * as Sentry from "@sentry/nextjs";

/** Load only the runtime-specific Sentry configuration. */
export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    await import("./sentry.server.config");
  }
  if (process.env.NEXT_RUNTIME === "edge") {
    await import("./sentry.edge.config");
  }
}

export const onRequestError = Sentry.captureRequestError;
